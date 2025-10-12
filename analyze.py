#!/usr/bin/env python3
"""
Audio analysis tool.

Usage:
  python analyze.py path\to\audio.mp3 [--prosody]
  python analyze.py https://example.com/audio.mp3 [--prosody]

Optional beacon analysis:
  --usebeacon                    Enable beacon-based silence measurement
  --beacon=<path.wav>            Path to beacon WAV template
  --channel=<mix|left|right>     Choose which recorded channel to analyze (dual-channel recordings)
  --asr                          Try to extract first word after each beacon (needs faster-whisper)

Advanced beacon tuning:
  --beacon_mode=<simple|strict>  Detector mode (simple = NCC only; strict = NCC + cosine)
  --cc_threshold=<float>         NCC gate (for normalized NCC in [-1,1], typical 0.50–0.70)
  --cos_threshold=<float>        Cosine verifier gate (strict mode), e.g., 0.85
  --refractory_sec=<float|-1>    Min spacing between matches; -1 = auto (~0.8× beacon length)
  --beacon_debug                 Print detector sanity + top NCC candidates
  --beacon_sweep                 Print hits across multiple cc_threshold values (debug only)

Notes:
- Requires: numpy, librosa, pydub (and ffmpeg), auditok, mutagen
- Optional: faster-whisper (for --asr)
"""

import sys, os, io, math, tempfile
from typing import Optional

import numpy as np
import librosa
from pydub import AudioSegment
import auditok

# Optional: only used if you pass HTTP URLs to analyze
try:
    import requests  # noqa: F401
except Exception:
    requests = None

# ---------------------------------------------------------------------------
# Global toggles (set by CLI)
# ---------------------------------------------------------------------------
useBeacon = False
beaconFilePath = None

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

# =========================== auditok compatibility ==========================
def _seg_bounds(seg):
    """
    Return (start_sec, end_sec) for an auditok segment across versions.
    Supports: seg.meta.start/end, seg.start/end, seg.timestamps, seg.bounds, or tuple-like.
    """
    meta = getattr(seg, "meta", None)
    if meta is not None and hasattr(meta, "start") and hasattr(meta, "end"):
        return float(meta.start), float(meta.end)
    if hasattr(seg, "start") and hasattr(seg, "end"):
        return float(seg.start), float(seg.end)
    if hasattr(seg, "timestamps"):
        st, en = seg.timestamps
        return float(st), float(en)
    if hasattr(seg, "bounds"):
        st, en = seg.bounds
        return float(st), float(en)
    try:
        st, en = seg
        return float(st), float(en)
    except Exception:
        raise AttributeError("Unsupported auditok segment object: cannot extract start/end")

# =========================== Core WAV/MP3 utilities =========================
def _export_wav_from_mp3(mp3_path, sr=16000, channel="mix"):
    """
    Decode MP3 → temp mono WAV at desired sample-rate with channel selection.
    channel: "mix" (default), "left", or "right"
    """
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    seg = AudioSegment.from_file(mp3_path)

    if channel in ("left", "right") and seg.channels == 2:
        left, right = seg.split_to_mono()
        seg = left if channel == "left" else right

    # Ensure mono after selection (or mixdown)
    seg = seg.set_channels(1).set_frame_rate(sr)
    seg.export(tmp_wav, format="wav")
    return tmp_wav

def _load_mono_wav(path, sr=16000):
    """Return mono float32 waveform and sample rate."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr

# =========================== Beacon: ZNCC (normalized) ======================
def _normxcorr_1d(x, h):
    """
    Z-normalized cross-correlation (ZNCC), robust and bounded to [-1, 1].
    - Per-window mean/variance for x
    - Zero-mean template h
    - Low-energy windows masked to 0 (avoid huge spikes on silence)
    """
    Lx, Lh = len(x), len(h)
    if Lx == 0 or Lh == 0:
        return np.array([], dtype=np.float32)

    x = x.astype(np.float32)
    h = h.astype(np.float32)

    # Zero-mean the template once; precompute its norm
    h = h - np.mean(h)
    h_norm = float(np.linalg.norm(h)) + 1e-12

    # FFT length
    n = 1
    tot = Lx + Lh - 1
    while n < tot:
        n <<= 1

    # Running sums via FFT for window stats over length Lh
    ones = np.ones(Lh, dtype=np.float32)
    X  = np.fft.rfft(x, n)
    O  = np.fft.rfft(ones, n)
    X2 = np.fft.rfft(x*x, n)

    sum_x   = np.fft.irfft(X  * O, n)[:tot]              # Σx
    sum_x2  = np.fft.irfft(X2 * O, n)[:tot]              # Σx^2
    mean_x  = sum_x / max(Lh, 1)
    var_x   = np.maximum(sum_x2 - (sum_x*sum_x)/max(Lh, 1), 0.0)  # L2 energy - μ^2 * L

    # Numerator: correlation with reversed, zero-mean template
    H = np.fft.rfft(h[::-1], n)
    num = np.fft.irfft(X * H, n)[:tot]                   # Σ x * h_rev
    # Remove mean contribution (∑h ≈ 0 after zero-meaning; kept for stability)
    sum_h = float(np.sum(h))
    num = num - mean_x * sum_h

    # Denominator: ||h|| * σ_x (per-position)
    std_x = np.sqrt(var_x + 1e-12)
    denom = h_norm * std_x

    # Mask very-low-energy windows to avoid divide-by-near-zero
    low_energy = std_x < 1e-3
    denom[low_energy] = np.inf

    ncc = (num / denom).astype(np.float32)
    ncc[low_energy] = 0.0
    return np.clip(ncc, -1.0, 1.0)

def _detect_beacons_in_wav(call_wav, beacon_wav, sr=16000,
                           cc_threshold=0.60,
                           refractory_sec=None,      # None => auto ~0.8 × beacon length
                           cos_threshold=0.85,       # <=0 or None to disable
                           merge_within_sec=0.030,
                           mode="strict",            # "strict" or "simple"
                           debug=False, topk_debug=10):
    """
    strict: NCC gate + min spacing + cosine verifier + merge.
    simple: NCC gate + min spacing + merge (no cosine).
    """
    x, _ = _load_mono_wav(call_wav, sr=sr)
    h, _ = _load_mono_wav(beacon_wav, sr=sr)

    # DC remove; leave scale/normalization to ZNCC
    x = x - np.mean(x)
    h = h - np.mean(h)

    Lx, Lh = len(x), len(h)
    if Lx == 0 or Lh == 0:
        return []

    beacon_len_sec = Lh / float(sr)
    if refractory_sec is None or refractory_sec < 0:
        refractory_sec = max(0.8 * beacon_len_sec, 0.05)  # ≥50 ms
    n_refrac = max(1, int(refractory_sec * sr))

    ncc = _normxcorr_1d(x, h)
    if ncc.size == 0:
        return []

    # Local maxima candidates (by NCC)
    cand_idx = [i for i in range(1, len(ncc)-1) if ncc[i] >= ncc[i-1] and ncc[i] >= ncc[i+1]]
    cand_idx.sort(key=lambda i: ncc[i], reverse=True)

    if debug:
        print("    top NCC candidates:")
        for i in cand_idx[:topk_debug]:
            t = (i - (Lh - 1)) / float(sr)
            print(f"      t={t:8.3f}s  ncc={ncc[i]:.3f}")

    # Non-maximum suppression using refractory window
    used = np.zeros_like(ncc, dtype=bool)
    kept = []
    for i in cand_idx:
        if used[i] or ncc[i] < cc_threshold:
            continue
        kept.append(i)
        lo = max(0, i - n_refrac)
        hi = min(len(ncc), i + n_refrac + 1)
        used[lo:hi] = True

    # Optional cosine verifier (strict mode only)
    def _cosine_slice(start_idx):
        if start_idx < 0 or start_idx + Lh > Lx:
            return 0.0
        y = x[start_idx:start_idx+Lh]
        y = y - np.mean(y)
        y = y / max(np.linalg.norm(y), 1e-9)
        hh = h - np.mean(h)
        hh = hh / max(np.linalg.norm(hh), 1e-9)
        return float(np.dot(y, hh))

    peaks = []
    for i in kept:
        start_idx = i - (Lh - 1)
        if start_idx < 0 or start_idx + Lh > Lx:
            continue
        if mode == "strict" and (cos_threshold is not None and cos_threshold > 0):
            cosv = _cosine_slice(start_idx)
            if cosv < float(cos_threshold):
                continue
        s = start_idx / float(sr)
        e = (start_idx + Lh) / float(sr)
        peaks.append((s, e))

    # Merge near-duplicates (floating jitter)
    if not peaks:
        if debug:
            print("    no peaks accepted after thresholds.")
        return []
    peaks.sort(key=lambda t: t[0])
    merged = []
    cur_s, cur_e = peaks[0]
    for s, e in peaks[1:]:
        if s - cur_s <= merge_within_sec:
            cur_s, cur_e = min(cur_s, s), max(cur_e, e)
        else:
            merged.append((cur_s, cur_e)); cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def _beacon_threshold_sweep(wav_path, beacon_wav, sr=16000, mode="simple", channel="mix"):
    """
    Debug helper: show NCC distribution + hits across cc_threshold values to tune quickly.
    """
    tmp = _export_wav_from_mp3(wav_path, sr=sr, channel=channel)
    try:
        x,_ = _load_mono_wav(tmp, sr=sr)
        h,_ = _load_mono_wav(beacon_wav, sr=sr)
        ncc = _normxcorr_1d(x, h)
        if ncc.size == 0:
            print("  NCC empty"); return
        qs = np.quantile(ncc, [0.5, 0.9, 0.95, 0.99, 0.995])
        print(f"  NCC quantiles 50/90/95/99/99.5: {qs.round(3)}")
        for cc in [0.40,0.45,0.50,0.55,0.60,0.65,0.70]:
            peaks = _detect_beacons_in_wav(tmp, beacon_wav, sr=sr,
                                           cc_threshold=cc,
                                           refractory_sec=None,
                                           cos_threshold=None if mode=="simple" else 0.85,
                                           mode=mode,
                                           debug=False)
            print(f"  cc_threshold={cc:.2f}  ->  hits={len(peaks)}")
    finally:
        try: os.remove(tmp)
        except: pass

# =========================== VAD / Prosody =================================
def _auditok_regions(wav_path, min_dur=0.2, max_dur=30.0, max_silence=0.35, energy_threshold=55):
    region = auditok.AudioRegion.load(wav_path)
    segs = region.split(min_dur=min_dur, max_dur=max_dur, max_silence=max_silence, energy_threshold=energy_threshold)
    return [_seg_bounds(s) for s in segs]

def _segment_turns_with_auditok(wav_path, min_dur=0.25, max_dur=30.0, max_silence=0.35, energy_threshold=55):
    turns = _auditok_regions(wav_path, min_dur, max_dur, max_silence, energy_threshold)
    return [("agent" if i % 2 == 0 else "customer", st, en) for i, (st, en) in enumerate(turns)]

def _prosody_features(y, sr):
    hop = 256
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=400, sr=sr, frame_length=2048, hop_length=hop)
    voiced = ~np.isnan(f0)
    voiced_ratio = float(voiced.mean()) if f0.size else 0.0
    f0_voiced = f0[voiced] if f0.size else np.array([])
    sc = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.005, wait=5)
    dur_sec = len(y) / sr
    speaking_rate_est = float(len(peaks) / dur_sec) if dur_sec > 0 else 0.0

    def _stats(a):
        if a.size == 0:
            return dict(mean=None, median=None, std=None, p10=None, p90=None, iqr=None, _range=None)
        return dict(
            mean=float(np.nanmean(a)),
            median=float(np.nanmedian(a)),
            std=float(np.nanstd(a)),
            p10=float(np.nanpercentile(a,10)),
            p90=float(np.nanpercentile(a,90)),
            iqr=float(np.nanpercentile(a,75)-np.nanpercentile(a,25)),
            _range=float(np.nanmax(a)-np.nanmin(a)),
        )

    return {
        "energy": _stats(rms),
        "pitch": _stats(f0_voiced),
        "centroid": _stats(sc),
        "voiced_ratio": voiced_ratio,
        "speaking_rate_est": speaking_rate_est,
        "duration_sec": dur_sec,
    }

def _score_from_features(feats):
    pitch_std = feats["pitch"]["std"] or 0.0
    energy_std = feats["energy"]["std"] or 0.0
    pitch_range = feats["pitch"]["_range"] or 0.0
    pitch_iqr   = feats["pitch"]["iqr"] or 0.0
    cent_std    = feats["centroid"]["std"] or 0.0

    def inv_scale(x, ref, maxv): return max(0.0, 100.0 * (1.0 - min(x/ref, maxv)))
    def scale(x, ref, maxv):     return max(0.0, 100.0 * min(x/ref, maxv))

    tone_stability = 0.6*inv_scale(pitch_std, 30.0, 2.0) + 0.4*inv_scale(energy_std, 0.08, 2.0)
    expressiveness = 0.5*scale(pitch_range, 120.0, 2.0) + 0.3*scale(pitch_iqr, 60.0, 2.0) + 0.2*scale(cent_std, 800.0, 2.0)
    overall = 0.5*expressiveness + 0.5*tone_stability
    return dict(
        tone_stability=round(float(tone_stability),1),
        expressiveness=round(float(expressiveness),1),
        overall_prosody=round(float(overall),1),
    )

def measure_agent_tone_and_expressiveness(mp3_path,
                                          sr=16000,
                                          min_dur=0.25, max_dur=30.0, max_silence=0.35, energy_threshold=55):
    """
    Returns per-turn prosody for Agent + aggregated scores (turn details suppressed in printing).
    """
    tmp_wav = _export_wav_from_mp3(mp3_path, sr=sr)
    try:
        labeled = _segment_turns_with_auditok(tmp_wav, min_dur, max_dur, max_silence, energy_threshold)
        y_full, sr = librosa.load(tmp_wav, sr=sr)
        agent_turns = []
        for spk, st, en in labeled:
            if spk != "agent": 
                continue
            s = max(0, int(st*sr)); e = max(s+1, int(en*sr))
            feats = _prosody_features(y_full[s:e], sr)
            scores = _score_from_features(feats)
            agent_turns.append({"start_sec": st, "end_sec": en, "duration_sec": feats["duration_sec"], "features": feats, "scores": scores})

        if not agent_turns:
            return {"agent_turns": [], "aggregate": None, "note": "No agent turns detected (check VAD thresholds)."}

        w = np.array([t["duration_sec"] for t in agent_turns]); w = w / w.sum()
        def wavg(getter):
            vals = np.array([getter(t) for t in agent_turns], dtype=float)
            return float(np.sum(vals*w))

        agg = {
            "pitch_mean_hz": wavg(lambda t: (t["features"]["pitch"]["mean"] or 0.0)),
            "pitch_range_hz": wavg(lambda t: (t["features"]["pitch"]["_range"] or 0.0)),
            "energy_mean": wavg(lambda t: (t["features"]["energy"]["mean"] or 0.0)),
            "energy_std": wavg(lambda t: (t["features"]["energy"]["std"] or 0.0)),
            "centroid_mean": wavg(lambda t: (t["features"]["centroid"]["mean"] or 0.0)),
            "voiced_ratio": wavg(lambda t: t["features"]["voiced_ratio"]),
            "speaking_rate_est": wavg(lambda t: t["features"]["speaking_rate_est"]),
        }
        agg_scores = {
            "tone_stability": float(np.mean([t["scores"]["tone_stability"] for t in agent_turns])),
            "expressiveness": float(np.mean([t["scores"]["expressiveness"] for t in agent_turns])),
            "overall_prosody": float(np.mean([t["scores"]["overall_prosody"] for t in agent_turns])),
        }
        return {"agent_turns": agent_turns, "aggregate": {"features": agg, "scores": agg_scores}}
    finally:
        try: os.remove(tmp_wav)
        except OSError: pass

# =========================== IO helpers ====================================
def load_bytes(path_or_url: str) -> bytes:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        if requests is None:
            raise RuntimeError("requests not installed; cannot fetch URL")
        resp = requests.get(path_or_url, timeout=60)
        resp.raise_for_status()
        return resp.content
    with open(path_or_url, 'rb') as f:
        return f.read()

def pauses_agent_to_customer(mp3_path,
                             min_dur=1,
                             max_dur=30,
                             max_silence=0.35,
                             energy_threshold=60):
    """
    Agent→Customer gap estimation using VAD turns (alternating agent/customer by order).
    """
    tmp_wav = _export_wav_from_mp3(mp3_path, sr=16000, channel="mix")
    try:
        turns = _segment_turns_with_auditok(tmp_wav, min_dur=min_dur, max_dur=max_dur,
                                            max_silence=max_silence, energy_threshold=energy_threshold)
        labeled = [("agent" if i % 2 == 0 else "customer", st, en) for i, (st,en) in enumerate(turns)]
        pauses_ms = []
        for i in range(len(labeled)-1):
            spk_i, _, end_i = labeled[i]
            spk_j, start_j, _ = labeled[i+1]
            if spk_i == "agent" and spk_j == "customer":
                pauses_ms.append(max(0.0, start_j - end_i) * 1000.0)

        if not pauses_ms:
            return {"count": 0, "avg_ms": None, "p95_ms": None, "pauses_ms": []}
        arr = np.array(pauses_ms, dtype=float)
        return {"count": int(arr.size), "avg_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr,95)), "pauses_ms": pauses_ms}
    finally:
        try: os.remove(tmp_wav)
        except OSError: pass

def analyze_with_mutagen(data: bytes) -> dict:
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(io.BytesIO(data))
        if mf is None:
            return {}
        info = getattr(mf, 'info', None)
        if not info:
            return {}
        out = {}
        dur = getattr(info, 'length', None)
        if dur is not None: out['duration_sec'] = round(float(dur), 3)
        ch = getattr(info, 'channels', None)
        if ch is not None: out['channels'] = ch
        sr = getattr(info, 'sample_rate', None)
        if sr is not None: out['sample_rate_hz'] = sr
        br = getattr(info, 'bitrate', None)
        if br is not None: out['bitrate_bps'] = br
        return out
    except Exception:
        return {}

def analyze_loudness_with_pydub(data: bytes) -> dict:
    try:
        seg = AudioSegment.from_file(io.BytesIO(data))
        duration_ms = len(seg)
        rms = seg.rms
        lufs = 20 * math.log10(max(1, rms) / 32768.0)
        from pydub.silence import detect_silence
        silences = detect_silence(seg, min_silence_len=500, silence_thresh=seg.dBFS - 16)
        total_silence_ms = sum((end - start) for start, end in silences)
        return {'duration_ms': duration_ms, 'rms': rms, 'approx_lufs': round(lufs, 2),
                'silence_regions': silences, 'total_silence_ms': int(total_silence_ms)}
    except Exception:
        return {}

def fmt_ms(v):
    if v is None: return "-"
    try: return f"{float(v):.1f} ms"
    except Exception: return str(v)

def fmt_sec(v):
    if v is None: return "-"
    try: return f"{float(v):.3f} s"
    except Exception: return str(v)

def fmt_bps(v):
    if not v: return "-"
    try: return f"{float(v)/1000.0:.0f} kbps"
    except Exception: return str(v)

_ASR_MODEL_CACHE = {}

def _get_asr_model(name="small"):
    """Cache faster-whisper models by name."""
    if name in _ASR_MODEL_CACHE:
        return _ASR_MODEL_CACHE[name]
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel(name, device="cpu", compute_type="int8")
        _ASR_MODEL_CACHE[name] = model
        print(f"    ASR: model loaded: {name}")
        return model
    except Exception:
        print(f"    ASR: model not available: {name}")
        return None

def _first_word_after(wav_path,
                      speech_start_time,
                      lookahead_sec=6.0,     # was 5.0 → give ASR a bit more to chew on
                      sr=16000,
                      onset_pad=0.25,        # was 0.150 → start a hair later after the boundary
                      language=None,         # None = auto-detect; set "en", "ru", etc. for better reliability
                      asr_model="small",
                      debug=False):
    """
    Return the first recognized word after speech_start_time.
    Strategy:
      1) Window = [onset_pad .. onset_pad + lookahead_sec]
      2) Pass A: VAD-filtered decode
      3) If empty, Pass B: no-VAD decode (helps when VAD misses soft first syllables)
    """
    model = _get_asr_model(asr_model)
    if model is None:
        if debug: print("    ASR: model not available")
        return None

    # Load window
    y, _ = _load_mono_wav(wav_path, sr=sr)
    s = max(0, int((speech_start_time + onset_pad) * sr))
    e = min(len(y), int((speech_start_time + onset_pad + lookahead_sec) * sr))
    if e <= s + 256:
        return None

    # Light normalization helps ASR on quiet snippets
    w = y[s:e].astype(np.float32)
    m = np.max(np.abs(w)) or 1.0
    w = 0.9 * (w / m)

    def _decode(vad=True):
        segs, _ = model.transcribe(
            w,
            language=language,               # None => auto
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 120} if vad else None,
            beam_size=1, best_of=1
        )
        txt = " ".join([getattr(sg, "text", "").strip() for sg in segs if getattr(sg, "text", "")]).strip()
        return txt

    # Pass A: with VAD
    text = _decode(vad=True)
    if debug: print(f"    ASR A (VAD) -> {text!r}")
    if not text:
        # Pass B: without VAD (catch soft/short first words)
        text = _decode(vad=False)
        if debug: print(f"    ASR B (no VAD) -> {text!r}")
    if not text:
        return None

    import re
    toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9'’\-]+", text)
    return toks[0] if toks else None

def silence_after_beacon(mp3_path,
                         beacon_wav_path,
                         sr=16000,
                         # VAD (speech start)
                         vad_min_dur=0.2, vad_max_dur=30.0, vad_max_silence=0.35, vad_energy_threshold=55,
                         # Detector knobs (ZNCC in [-1,1])
                         cc_threshold=0.60, refractory_sec=None, cos_threshold=0.85,
                         mode="strict", debug=False, beacon_channel="mix",
                         # ASR
                         use_asr=True, asr_lookahead=5.0):
    """
    For each beacon, measure silence until next speech start, and (optionally) first word.
    Returns: {"beacons":[{"beacon_start_sec","beacon_end_sec","next_speech_start_sec","silence_ms","first_word"}]}
    """
    tmp_call_wav = _export_wav_from_mp3(mp3_path, sr=sr, channel=beacon_channel)
    try:
        if debug:
            bx, _ = _load_mono_wav(beacon_wav_path, sr=sr)
            cx, _ = _load_mono_wav(tmp_call_wav, sr=sr)
            def _stats(x): return dict(dur_s=len(x)/sr, peak=float(np.max(np.abs(x))), rms=float(np.sqrt(np.mean(x*x))))
            bstats, cstats = _stats(bx), _stats(cx)
            print(f"    beacon sanity: dur={bstats['dur_s']:.3f}s peak={bstats['peak']:.3f} rms={bstats['rms']:.3f}")
            print(f"    call   sanity: dur={cstats['dur_s']:.3f}s peak={cstats['peak']:.3f} rms={cstats['rms']:.3f} ch={beacon_channel}")

        beacons = _detect_beacons_in_wav(tmp_call_wav, beacon_wav_path, sr=sr,
                                         cc_threshold=cc_threshold,
                                         refractory_sec=refractory_sec,
                                         cos_threshold=cos_threshold,
                                         mode=mode,
                                         debug=debug)

        speech_regions = _auditok_regions(tmp_call_wav,
                                          min_dur=vad_min_dur, max_dur=vad_max_dur,
                                          max_silence=vad_max_silence, energy_threshold=vad_energy_threshold)

        results = []
        for (b_st, b_en) in beacons:
            next_speech = None
            for (s_st, s_en) in speech_regions:
                if s_st >= b_en:
                    next_speech = s_st
                    break

            if next_speech is None:
                results.append({"beacon_start_sec": round(float(b_st),3),
                                "beacon_end_sec": round(float(b_en),3),
                                "next_speech_start_sec": None,
                                "silence_ms": None,
                                "first_word": None})
                continue

            gap_ms = max(0.0, (next_speech - b_en) * 1000.0)
            first_word = _first_word_after(tmp_call_wav, next_speech, lookahead_sec=asr_lookahead, sr=sr)
            results.append({"beacon_start_sec": round(float(b_st),3),
                            "beacon_end_sec": round(float(b_en),3),
                            "next_speech_start_sec": round(float(next_speech),3),
                            "silence_ms": round(float(gap_ms),1),
                            "first_word": first_word})
        return {"beacons": results}
    finally:
        try: os.remove(tmp_call_wav)
        except OSError: pass

# =========================== Main ==========================================
def main():
    global useBeacon, beaconFilePath

    if len(sys.argv) < 2:
        print("Usage: python analyze.py <path-or-url> [--prosody] [--usebeacon --beacon=path.wav ...]")
        sys.exit(1)

    args = sys.argv[1:]
    target = args[0]
    flags = args[1:]
    flags_lc = [a.lower() for a in flags]

    prosody_enabled = any(f in ("prosody=true", "--prosody", "--prosody=true", "prosody=1") for f in flags_lc)

    # Beacon flags
    useBeacon = False
    beaconFilePath = None
    use_asr = True

    beacon_channel = "mix"
    beacon_mode = "strict"
    beacon_debug = any(f in ("--beacon_debug", "beacon_debug=1", "beacon_debug=true") for f in flags_lc)

    def _getf(name, default):
        for raw in flags:
            if raw.lower().startswith(f"--{name}="):
                try: return float(raw.split("=",1)[1])
                except Exception: return default
        return default

    cc_threshold  = _getf("cc_threshold", 0.60)
    refractory_sec = _getf("refractory_sec", -1.0)  # -1 => auto
    cos_threshold = _getf("cos_threshold", 0.85)

    for raw in flags:
        low = raw.lower()
        if low.startswith("--beacon=") or low.startswith("beacon="):
            beaconFilePath = raw.split("=",1)[1].strip('"').strip("'"); useBeacon = True
        if low in ("--usebeacon", "--beacon", "usebeacon=1", "usebeacon=true"):
            useBeacon = True
        if low.startswith("--beacon_mode="):
            val = raw.split("=",1)[1].strip().lower()
            beacon_mode = val if val in ("simple","strict") else "strict"
        if low.startswith("--channel="):
            ch = raw.split("=",1)[1].strip().lower()
            beacon_channel = ch if ch in ("mix","left","right") else "mix"

    print(f"Processing: {target} ...")

    is_url = target.startswith("http://") or target.startswith("https://")
    if not is_url and not os.path.exists(target):
        print(f"{RED}Error:{RESET} File not found: {target}")
        if "," in target:
            print(f"{YELLOW}Tip:{RESET} The filename contains a comma. Did you mean to use a dot (.)?")
        print("Ensure the path is correct, e.g. logs/recordings/latency-check.mp3")
        sys.exit(2)

    tmp_file = None
    try:
        data = load_bytes(target)
        meta = analyze_with_mutagen(data)
        loud = analyze_loudness_with_pydub(data)

        mp3_path = target
        if is_url:
            tmp_file = tempfile.mkstemp(suffix=".mp3")[1]
            with open(tmp_file, "wb") as f: f.write(data)
            mp3_path = tmp_file

        # --- Optional: Beacon threshold sweep (debug only) ---
        if useBeacon and any(f in ("--beacon_sweep",) for f in flags_lc):
            print(f"{CYAN}{BOLD}Beacon threshold sweep{RESET}")
            _ = _beacon_threshold_sweep(mp3_path, beaconFilePath,
                                        sr=16000, mode=("simple" if beacon_mode=="simple" else "strict"),
                                        channel=beacon_channel)
            return

        # --- VAD pauses (Agent → Customer) ---
        vad = {}
        vad_error = None
        try:
            vad = pauses_agent_to_customer(mp3_path)
        except Exception as e:
            vad_error = str(e)

        # --- Output: Metadata ---
        print(f"{CYAN}{BOLD}Metadata{RESET}")
        print(f"  Duration: {fmt_sec(meta.get('duration_sec'))}")
        print(f"  Channels: {meta.get('channels', '-')}")
        print(f"  Sample Rate: {meta.get('sample_rate_hz', '-')}")
        print(f"  Bitrate: {fmt_bps(meta.get('bitrate_bps'))}")

        # --- Output: Loudness & Silence ---
        print(f"{CYAN}{BOLD}Loudness & Silence{RESET}")
        approx_lufs = loud.get('approx_lufs')
        duration_ms = loud.get('duration_ms')
        total_silence_ms = loud.get('total_silence_ms')
        silence_pct = (float(total_silence_ms)/float(duration_ms)*100.0) if (duration_ms and total_silence_ms is not None and duration_ms>0) else None
        print(f"  Approx LUFS: {approx_lufs if approx_lufs is not None else '-'}")
        print(f"  Total Silence: {fmt_ms(total_silence_ms)}" + (f" ({silence_pct:.1f}%)" if silence_pct is not None else ""))

        # --- Output: Prosody (aggregate only) ---
        if prosody_enabled:
            print(f"{CYAN}{BOLD}Agent Prosody{RESET}")
            prosody = None
            prosody_error = None
            try:
                prosody = measure_agent_tone_and_expressiveness(mp3_path)
            except Exception as e:
                prosody_error = str(e)

            if prosody and prosody.get('aggregate'):
                scores = prosody['aggregate'].get('scores', {})
                def _label_and_color(v):
                    if v is None: return "-", RESET, "—"
                    v = float(v)
                    if v >= 80: return f"{v:.1f}", GREEN, "excellent"
                    if v >= 65: return f"{v:.1f}", CYAN,  "strong"
                    if v >= 50: return f"{v:.1f}", YELLOW,"okay"
                    return f"{v:.1f}", RED, "needs work"
                o_val = scores.get('overall_prosody'); t_val = scores.get('tone_stability'); e_val = scores.get('expressiveness')
                o_txt,o_col,o_lbl = _label_and_color(o_val); t_txt,t_col,t_lbl = _label_and_color(t_val); e_txt,e_col,e_lbl = _label_and_color(e_val)
                print(f"  {BOLD}Aggregate Scores{RESET}")
                print(f"    Overall: {o_col}{o_txt}{RESET} / 100   ({o_lbl})")
                print(f"    Tone:    {t_col}{t_txt}{RESET} / 100   ({t_lbl})")
                print(f"    Expr.:   {e_col}{e_txt}{RESET} / 100   ({e_lbl})")
                def _bar(v, width=20):
                    if v is None: return "·"*width
                    n = int(round((float(v)/100.0)*width))
                    return "█"*n + "·"*(width-n)
                print(f"    Bars:    O [{_bar(o_val)}]  T [{_bar(t_val)}]  E [{_bar(e_val)}]")
            else:
                print("  Not available")
            if prosody_error:
                print(f"{YELLOW}  Note:{RESET} {prosody_error}")

        # --- Output: Beacon analysis ---
        if useBeacon:
            print(f"{CYAN}{BOLD}Beacon → First Speech{RESET}")
            if not beaconFilePath or not os.path.exists(beaconFilePath):
                print(f"{YELLOW}  Note:{RESET} Beacon is enabled but beaconFilePath is missing or invalid. Use --beacon=path.wav")
            else:
                try:
                    bres = silence_after_beacon(
                        mp3_path,
                        beaconFilePath,
                        sr=16000,
                        use_asr=use_asr,
                        cc_threshold=cc_threshold,
                        refractory_sec=(None if refractory_sec < 0 else refractory_sec),
                        cos_threshold=cos_threshold,
                        mode=beacon_mode,
                        debug=beacon_debug,
                        beacon_channel=beacon_channel
                    )
                    items = bres.get("beacons", [])
                    if not items:
                        print("  No beacons detected (consider lowering cc_threshold or verify the WAV).")
                    else:
                        for i, item in enumerate(items, 1):
                            st = item["beacon_start_sec"]; en = item["beacon_end_sec"]
                            ns = item["next_speech_start_sec"]; gap = item["silence_ms"]
                            fw = item["first_word"]
                            fw_txt = fw if (fw and isinstance(fw, str) and fw.strip()) else "—"
                            print(f"  #{i} beacon {st:.3f}→{en:.3f} | next speech: {fmt_sec(ns)} | silence: {fmt_ms(gap)} | first word: {fw_txt}")

                        silences = [x["silence_ms"] for x in items if x.get("silence_ms") is not None]
                        if silences:
                            arr = np.array(silences, dtype=float)
                            avg = arr.mean(); p95 = np.percentile(arr, 95)
                            print(f"  Summary: {avg:.1f} ms (avg), {arr.min():.1f}–{arr.max():.1f} ms, n={len(arr)}")
                            print(f"           P95: {p95:.1f} ms")
                except Exception as e:
                    print(f"{YELLOW}  Note:{RESET} Beacon analysis failed: {e}")

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        sys.exit(1)
    finally:
        try:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass

if __name__ == '__main__':
    main()
