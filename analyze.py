#!/usr/bin/env python3
"""
Audio analysis tool.

Usage:
  python analyze.py path\to\audio.mp3 [--prosody] [--usebeacon --beacon=audio/click_beacon.wav [--asr]]
  python analyze.py https://example.com/audio.mp3 [--prosody] [--usebeacon --beacon=audio/click_beacon.wav [--asr]]

Prints duration, channels, sample rate, bitrate (if available), and basic loudness/silence stats.
Optionally:
  --prosody             : compute agent tone/expressiveness (librosa + pyin)
  --usebeacon           : enable beacon-based silence measurement
  --beacon=<path.wav>   : path to the beacon WAV template
  --asr                 : try to extract the first word after each beacon (needs faster-whisper)
"""

import sys
import os
import io
import json
import math
from typing import Optional
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment
import auditok

# Optional import for URL fetching happens in load_bytes()
import requests  # noqa: F401 (kept for users who use requests directly elsewhere)

# ---------------------------------------------------------------------------
# Global configuration toggles (can be overridden via CLI flags)
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


# =========================== Beacon utilities ===============================

def _load_mono_wav(path, sr=16000):
    """Return mono float32 waveform and sample rate."""
    y, srx = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr

def _normxcorr_1d(x, h):
    """
    Normalized cross-correlation of template h in signal x.
    Returns NCCF array of length len(x)+len(h)-1.
    """
    Lx, Lh = len(x), len(h)
    if Lx == 0 or Lh == 0:
        return np.array([], dtype=np.float32)

    # FFT length
    n = 1
    tot = Lx + Lh - 1
    while n < tot:
        n <<= 1

    X = np.fft.rfft(x, n)
    H = np.fft.rfft(h[::-1], n)  # reverse template for correlation
    num = np.fft.irfft(X * H, n)[:tot]

    # Denominator: sqrt( sum(x^2 win) * sum(h^2) )
    x2 = x * x
    w = np.ones(Lh, dtype=np.float32)
    X2 = np.fft.rfft(x2, n)
    W = np.fft.rfft(w, n)
    denom_x = np.fft.irfft(X2 * W, n)[:tot]
    denom_h = np.sum(h * h)
    denom = np.sqrt(np.maximum(denom_x * denom_h, 1e-12)).astype(np.float32)

    ncc = (num / denom).astype(np.float32)
    return ncc

def _detect_beacons_in_wav(call_wav, beacon_wav, sr=16000,
                           cc_threshold=0.55,          # stricter default
                           refractory_sec=None,        # if None → 0.8 * beacon_len
                           cos_threshold=0.82,         # 2nd-stage verifier
                           merge_within_sec=0.030      # merge dup peaks within 30 ms
                           ):
    """
    Return list of (start_sec, end_sec) for beacon matches with stricter criteria:
    - NCC threshold (cc_threshold)
    - Min spacing (~ beacon length)
    - Second-stage cosine similarity check (cos_threshold)
    - Merge very-close duplicates
    """
    x, _ = _load_mono_wav(call_wav, sr=sr)
    h, _ = _load_mono_wav(beacon_wav, sr=sr)

    # Normalize / DC remove
    x = x - np.mean(x); h = h - np.mean(h)
    x = x / max(np.max(np.abs(x)), 1e-6)
    h = h / max(np.max(np.abs(h)), 1e-6)

    Lx, Lh = len(x), len(h)
    if Lx == 0 or Lh == 0:
        return []

    # If user didn't set refractory, tie it to beacon length (robust)
    beacon_len_sec = Lh / float(sr)
    if refractory_sec is None:
        refractory_sec = max(0.5 * beacon_len_sec, 0.8 * beacon_len_sec)

    # 1) NCC pass
    ncc = _normxcorr_1d(x, h)
    if ncc.size == 0:
        return []

    # 2) Peak selection with threshold + min spacing
    # Local maxima where ncc >= neighbors and >= threshold
    cand_idx = []
    for i in range(1, len(ncc) - 1):
        v = ncc[i]
        if v >= cc_threshold and v >= ncc[i-1] and v >= ncc[i+1]:
            cand_idx.append(i)

    # Non-maximum suppression with refractory window
    cand_idx.sort(key=lambda i: ncc[i], reverse=True)
    used = np.zeros_like(ncc, dtype=bool)
    peaks_idx = []
    n_suppress = max(1, int(refractory_sec * sr))
    for i in cand_idx:
        if used[i]:
            continue
        peaks_idx.append(i)
        lo = max(0, i - n_suppress)
        hi = min(len(ncc), i + n_suppress + 1)
        used[lo:hi] = True

    # 3) Second-stage verifier: cosine similarity in time domain
    def _cosine_slice(start_idx):
        if start_idx < 0 or start_idx + Lh > Lx:
            return 0.0
        y = x[start_idx:start_idx+Lh]
        y = y - np.mean(y)
        y = y / max(np.linalg.norm(y), 1e-9)
        return float(np.dot(y, h / max(np.linalg.norm(h), 1e-9)))

    peaks = []
    for i in peaks_idx:
        start_idx = i - (Lh - 1)
        if start_idx < 0 or start_idx + Lh > Lx:
            continue
        cos = _cosine_slice(start_idx)
        if cos < cos_threshold:
            continue  # reject weak lookalikes
        start_sec = start_idx / float(sr)
        end_sec = (start_idx + Lh) / float(sr)
        peaks.append((start_sec, end_sec))

    # 4) Merge near-duplicates (floating-point jitter)
    if not peaks:
        return []
    peaks.sort(key=lambda t: t[0])
    merged = []
    cur_s, cur_e = peaks[0]
    for s, e in peaks[1:]:
        if s - cur_s <= merge_within_sec:
            # keep earliest start / latest end
            cur_s = min(cur_s, s)
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged

def _auditok_regions(wav_path, min_dur=0.2, max_dur=30.0, max_silence=0.35, energy_threshold=55):
    region = auditok.AudioRegion.load(wav_path)
    segs = region.split(min_dur=min_dur, max_dur=max_dur, max_silence=max_silence, energy_threshold=energy_threshold)
    return [_seg_bounds(s) for s in segs]

def _first_word_after(wav_path, start_time, end_time, lookahead_sec=3.0, sr=16000):
    """
    Optional: use local ASR if available to get first word within [speech_start, speech_start+lookahead].
    Tries faster_whisper; returns None if unavailable.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        return None  # no ASR

    try:
        model = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception:
        return None

    y, sr = _load_mono_wav(wav_path, sr=sr)
    s = max(0, int(start_time * sr))
    e = min(len(y), int((start_time + lookahead_sec) * sr))
    if e <= s + 256:
        return None

    segments, _ = model.transcribe(
        y[s:e],
        language="en",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 200}
    )
    text = ""
    for seg in segments:
        if seg.text:
            text += " " + seg.text
    text = text.strip()
    if not text:
        return None

    import re
    toks = re.findall(r"[A-Za-z0-9'’\-]+", text)
    return toks[0] if toks else None

def silence_after_beacon(mp3_path,
                         beacon_wav_path,
                         sr=16000,
                         vad_min_dur=0.2,
                         vad_max_dur=30.0,
                         vad_max_silence=0.35,
                         vad_energy_threshold=55,
                         cc_threshold=0.45,
                         refractory_sec=0.15,
                         use_asr=False):
    """
    For each beacon occurrence in the call, measure the silence until the next speech start,
    and (optionally) extract the first word spoken after that silence.

    Returns:
      {"beacons": [{"beacon_start_sec", "beacon_end_sec", "next_speech_start_sec", "silence_ms", "first_word"}]}
    """
    tmp_call_wav = _export_mono_wav_from_mp3(mp3_path, sr=sr)
    try:
        beacons = _detect_beacons_in_wav(tmp_call_wav, beacon_wav_path, sr=sr,
                                         cc_threshold=cc_threshold, refractory_sec=refractory_sec)
        speech_regions = _auditok_regions(tmp_call_wav,
                                          min_dur=vad_min_dur, max_dur=vad_max_dur,
                                          max_silence=vad_max_silence, energy_threshold=vad_energy_threshold)

        results = []
        for (b_st, b_en) in beacons:
            # Find first region that begins strictly after the beacon ends
            next_speech = None
            for (s_st, s_en) in speech_regions:
                if s_st >= b_en:
                    next_speech = s_st
                    break
            if next_speech is None:
                results.append({
                    "beacon_start_sec": round(float(b_st), 3),
                    "beacon_end_sec": round(float(b_en), 3),
                    "next_speech_start_sec": None,
                    "silence_ms": None,
                    "first_word": None
                })
                continue

            gap_ms = max(0.0, (next_speech - b_en) * 1000.0)
            first_word = _first_word_after(tmp_call_wav, next_speech, next_speech + 3.0, sr=sr) if use_asr else None

            results.append({
                "beacon_start_sec": round(float(b_st), 3),
                "beacon_end_sec": round(float(b_en), 3),
                "next_speech_start_sec": round(float(next_speech), 3),
                "silence_ms": round(float(gap_ms), 1),
                "first_word": first_word
            })

        return {"beacons": results}
    finally:
        try:
            os.remove(tmp_call_wav)
        except OSError:
            pass


# =========================== Core analysis utils ============================

def _export_mono_wav_from_mp3(mp3_path, sr=16000):
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    AudioSegment.from_file(mp3_path).set_channels(1).set_frame_rate(sr).export(tmp_wav, format="wav")
    return tmp_wav

def _segment_turns_with_auditok(wav_path, min_dur=0.25, max_dur=30.0, max_silence=0.35, energy_threshold=55):
    region = auditok.AudioRegion.load(wav_path)
    segs = region.split(min_dur=min_dur, max_dur=max_dur, max_silence=max_silence, energy_threshold=energy_threshold)
    turns = [_seg_bounds(s) for s in segs]
    labeled = [("agent" if i % 2 == 0 else "customer", st, en) for i, (st, en) in enumerate(turns)]
    return labeled

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

    feats = {
        "energy": _stats(rms),
        "pitch": _stats(f0_voiced),
        "centroid": _stats(sc),
        "voiced_ratio": voiced_ratio,
        "speaking_rate_est": speaking_rate_est,
        "duration_sec": dur_sec,
    }
    return feats

def _score_from_features(feats):
    pitch_std = feats["pitch"]["std"] or 0.0
    energy_std = feats["energy"]["std"] or 0.0
    pitch_range = feats["pitch"]["_range"] or 0.0
    pitch_iqr   = feats["pitch"]["iqr"] or 0.0
    cent_std    = feats["centroid"]["std"] or 0.0

    def inv_scale(x, ref, maxv):
        return max(0.0, 100.0 * (1.0 - min(x/ref, maxv)))
    def scale(x, ref, maxv):
        return max(0.0, 100.0 * min(x/ref, maxv))

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
    Returns per-turn prosody for Agent + aggregated scores.
    """
    tmp_wav = _export_mono_wav_from_mp3(mp3_path, sr=sr)
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
            agent_turns.append({
                "start_sec": st, "end_sec": en, "duration_sec": feats["duration_sec"],
                "features": feats, "scores": scores
            })

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


# =========================== Helpers ============================

def load_bytes(path_or_url: str) -> bytes:
    try:
        import requests  # type: ignore
    except Exception:
        requests = None
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
    if auditok is None or np is None:
        raise RuntimeError("'auditok' and 'numpy' are required for pauses_agent_to_customer. Install with: pip install auditok numpy")

    audio = AudioSegment.from_file(mp3_path).set_channels(1).set_frame_rate(16000)
    tmp_wav = tempfile.mktemp(suffix=".wav")
    audio.export(tmp_wav, format="wav")

    try:
        region = auditok.AudioRegion.load(tmp_wav)
        segments = region.split(
            min_dur=min_dur,
            max_dur=max_dur,
            max_silence=max_silence,
            energy_threshold=energy_threshold,
        )

        # Collect (start, end) in seconds using version-safe accessor
        turns = [_seg_bounds(s) for s in segments]

        # Assign speakers by order (Agent first, then Customer, alternating)
        labeled = [("agent" if i % 2 == 0 else "customer", t[0], t[1]) for i, t in enumerate(turns)]

        # Compute gaps: Agent end → next Customer start
        pauses_ms = []
        for i in range(len(labeled) - 1):
            spk_i, start_i, end_i = labeled[i]
            spk_j, start_j, end_j = labeled[i+1]
            if spk_i == "agent" and spk_j == "customer":
                gap = max(0.0, start_j - end_i)
                pauses_ms.append(gap * 1000.0)

        if not pauses_ms:
            return {"count": 0, "avg_ms": None, "p95_ms": None, "pauses_ms": []}

        arr = np.array(pauses_ms)
        return {
            "count": int(arr.size),
            "avg_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "pauses_ms": pauses_ms,
        }
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

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
        if dur is not None:
            out['duration_sec'] = round(float(dur), 3)
        ch = getattr(info, 'channels', None)
        if ch is not None:
            out['channels'] = ch
        sr = getattr(info, 'sample_rate', None)
        if sr is not None:
            out['sample_rate_hz'] = sr
        br = getattr(info, 'bitrate', None)
        if br is not None:
            out['bitrate_bps'] = br
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
        return {
            'duration_ms': duration_ms,
            'rms': rms,
            'approx_lufs': round(lufs, 2),
            'silence_regions': silences,
            'total_silence_ms': int(total_silence_ms),
        }
    except Exception:
        return {}

def fmt_ms(v):
    if v is None:
        return "-"
    try:
        return f"{float(v):.1f} ms"
    except Exception:
        return str(v)

def fmt_sec(v):
    if v is None:
        return "-"
    try:
        return f"{float(v):.3f} s"
    except Exception:
        return str(v)

def fmt_bps(v):
    if not v:
        return "-"
    try:
        kbps = float(v) / 1000.0
        return f"{kbps:.0f} kbps"
    except Exception:
        return str(v)


# =========================== Main ============================

def main():
    global useBeacon, beaconFilePath

    if len(sys.argv) < 2:
        print("Usage: python analyze.py <path-or-url> [--prosody] [--usebeacon --beacon=path.wav [--asr]]")
        sys.exit(1)

    # Parse args
    args = sys.argv[1:]
    target = args[0]
    flags = args[1:]
    flags_lc = [a.lower() for a in flags]

    prosody_enabled = any(f in ("prosody=true", "--prosody", "--prosody=true", "prosody=1") for f in flags_lc)

    # New beacon flags (populate globals)
    useBeacon = False
    beaconFilePath = None
    use_asr = any(f in ("--asr", "asr=1", "asr=true") for f in flags_lc)
    for f in flags:
        fl = f.lower()
        if fl.startswith("--beacon=") or fl.startswith("beacon="):
            beaconFilePath = f.split("=", 1)[1].strip('"').strip("'")
            useBeacon = True
        if fl in ("--usebeacon", "--beacon", "usebeacon=1", "usebeacon=true"):
            useBeacon = True

    # Show processing message
    print(f"Processing: {target} ...")

    # Validate path if local
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

        # Prepare a local path for downstream analyses
        mp3_path = target
        if is_url:
            tmp_file = tempfile.mkstemp(suffix=".mp3")[1]
            with open(tmp_file, "wb") as f:
                f.write(data)
            mp3_path = tmp_file

        # --- VAD pauses (Agent → Customer) ---
        vad = {}
        vad_error = None
        try:
            vad = pauses_agent_to_customer(mp3_path)
        except Exception as e:
            vad_error = str(e)

        # ===================== OUTPUT =====================
        # Metadata
        print(f"{CYAN}{BOLD}Metadata{RESET}")
        print(f"  Duration: {fmt_sec(meta.get('duration_sec'))}")
        print(f"  Channels: {meta.get('channels', '-')}")
        print(f"  Sample Rate: {meta.get('sample_rate_hz', '-')}")
        print(f"  Bitrate: {fmt_bps(meta.get('bitrate_bps'))}")

        # Loudness / Silence
        print(f"{CYAN}{BOLD}Loudness & Silence{RESET}")
        approx_lufs = loud.get('approx_lufs')
        duration_ms = loud.get('duration_ms')
        total_silence_ms = loud.get('total_silence_ms')
        silence_pct = None
        if duration_ms and total_silence_ms is not None and duration_ms > 0:
            silence_pct = (float(total_silence_ms) / float(duration_ms)) * 100.0
        print(f"  Approx LUFS: {approx_lufs if approx_lufs is not None else '-'}")
        print(f"  Total Silence: {fmt_ms(total_silence_ms)}" + (f" ({silence_pct:.1f}%)" if silence_pct is not None else ""))

        # VAD Pauses (Agent → Customer)
        print(f"{CYAN}{BOLD}VAD (Agent → Customer pauses){RESET}")
        if vad:
            print(f"  Count: {vad.get('count', 0)}")
            print(f"  Avg: {fmt_ms(vad.get('avg_ms'))}")
            print(f"  P95: {fmt_ms(vad.get('p95_ms'))}")
        else:
            print("  No pauses computed")
        if vad_error:
            print(f"{YELLOW}  Note: {vad_error}{RESET}")

        # Agent Prosody (tone/expressiveness) - optional
        if prosody_enabled:
            print(f"{CYAN}{BOLD}Agent Prosody:{RESET}")
            prosody = None
            prosody_error = None
            try:
                prosody = measure_agent_tone_and_expressiveness(mp3_path)
            except Exception as e:
                prosody_error = str(e)

            if prosody and prosody.get('aggregate'):
                scores = prosody['aggregate'].get('scores', {})
                overall = scores.get('overall_prosody')
                tone = scores.get('tone_stability')
                def _label_and_color(v):
                    if v is None:
                        return "-", RESET, "—"
                    v = float(v)
                    if v >= 80:
                        return f"{v:.1f}", GREEN, "excellent"
                    elif v >= 65:
                        return f"{v:.1f}", CYAN, "strong"
                    elif v >= 50:
                        return f"{v:.1f}", YELLOW, "okay"
                    else:
                        return f"{v:.1f}", RED, "needs work"

                o_val = scores.get('overall_prosody')
                t_val = scores.get('tone_stability')
                e_val = scores.get('expressiveness')

                o_txt, o_col, o_lbl = _label_and_color(o_val)
                t_txt, t_col, t_lbl = _label_and_color(t_val)
                e_txt, e_col, e_lbl = _label_and_color(e_val)

                print(f"  {BOLD}Aggregate Scores{RESET}")
                print(f"    Overall: {o_col}{o_txt}{RESET} / 100   ({o_lbl})")
                print(f"    Tone:    {t_col}{t_txt}{RESET} / 100   ({t_lbl})")
                print(f"    Expr.:   {e_col}{e_txt}{RESET} / 100   ({e_lbl})")

                # Optional: quick human comparison bars (0–100)
                def _bar(v, width=20):
                    if v is None: return "·" * width
                    n = int(round((float(v)/100.0)*width))
                    return "█" * n + "·" * (width - n)

                print(f"    Bars:    O [{_bar(o_val)}]  T [{_bar(t_val)}]  E [{_bar(e_val)}]")
            else:
                print("  Not available")
            if prosody_error:
                print(f"{YELLOW}  Note:{RESET} {prosody_error}")

        # Beacon-based silence (global toggle)
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
                        use_asr=use_asr
                    )
                    items = bres.get("beacons", [])
                    if not items:
                        print("  No beacons detected (consider lowering cc_threshold or verify the WAV).")
                    else:
                        # Per-beacon lines
                        for i, item in enumerate(items, 1):
                            st = item["beacon_start_sec"]; en = item["beacon_end_sec"]
                            ns = item["next_speech_start_sec"]; gap = item["silence_ms"]
                            fw = item["first_word"]
                            print(f"  #{i} beacon {st:.3f}→{en:.3f} | next speech: {fmt_sec(ns)} | silence: {fmt_ms(gap)}"
                                  + (f" | first word: {fw}" if fw else ""))

                        # Summary stats
                        silences = [x["silence_ms"] for x in items if x.get("silence_ms") is not None]
                        if silences:
                            arr = np.array(silences, dtype=float)
                            avg = arr.mean()
                            p95 = np.percentile(arr, 95)
                            print(f"  Summary: {avg:.1f} ms (avg), {arr.min():.1f}–{arr.max():.1f} ms, n={len(arr)}")
                            print(f"           P95: {p95:.1f} ms")
                except Exception as e:
                    print(f"{YELLOW}  Note:{RESET} Beacon analysis failed: {e}")

        # (Customer → Agent) analysis removed as requested

    except Exception as e:
        print(f"{RED}Error:{RESET} {e}")
        sys.exit(1)
    finally:
        # Clean up temp mp3 if created
        try:
            if tmp_file and os.path.exists(tmp_file):
                os.remove(tmp_file)
        except Exception:
            pass


if __name__ == '__main__':
    main()
