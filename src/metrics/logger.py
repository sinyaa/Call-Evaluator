import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def now_ms() -> int:
    """
    Get current timestamp in milliseconds.
    
    Returns:
        Current timestamp in milliseconds since epoch
    """
    return int(time.time() * 1000)

def log_event(log_file: str, event_data: Dict[str, Any]) -> None:
    """
    Log an event to a JSONL file.
    
    Args:
        log_file: Path to the log file
        event_data: Event data to log
    """
    try:
        # Ensure logs directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Add timestamp if not present
        if 'ts' not in event_data:
            event_data['ts'] = now_ms()
        
        # Add human-readable timestamp
        event_data['timestamp'] = datetime.now().isoformat()
        
           
    except Exception as e:
        logger.error(f"Error logging event: {e}")

def log_call_start(call_sid: str, to_number: str, from_number: str, 
                  agent_prompt: str, log_file: str) -> None:
    """
    Log the start of a call.
    
    Args:
        call_sid: Twilio call SID
        to_number: Destination phone number
        from_number: Source phone number
        agent_prompt: Agent system prompt
        log_file: Path to the log file
    """
    event_data = {
        'event': 'call.start',
        'call_sid': call_sid,
        'to_number': to_number,
        'from_number': from_number,
        'agent_prompt': agent_prompt[:100] + '...' if len(agent_prompt) > 100 else agent_prompt
    }
    log_event(log_file, event_data)

def log_call_end(call_sid: str, duration_ms: int, log_file: str) -> None:
    """
    Log the end of a call.
    
    Args:
        call_sid: Twilio call SID
        duration_ms: Call duration in milliseconds
        log_file: Path to the log file
    """
    event_data = {
        'event': 'call.end',
        'call_sid': call_sid,
        'duration_ms': duration_ms
    }
    log_event(log_file, event_data)

def log_audio_received(call_sid: str, audio_length: int, log_file: str) -> None:
    """
    Log received audio data.
    
    Args:
        call_sid: Twilio call SID
        audio_length: Length of audio data in bytes
        log_file: Path to the log file
    """
    event_data = {
        'event': 'audio.received',
        'call_sid': call_sid,
        'audio_length': audio_length
    }
    log_event(log_file, event_data)

def log_audio_sent(call_sid: str, audio_length: int, log_file: str) -> None:
    """
    Log sent audio data.
    
    Args:
        call_sid: Twilio call SID
        audio_length: Length of audio data in bytes
        log_file: Path to the log file
    """
    event_data = {
        'event': 'audio.sent',
        'call_sid': call_sid,
        'audio_length': audio_length
    }
    log_event(log_file, event_data)

def log_latency_metric(call_sid: str, metric_name: str, value_ms: int, log_file: str) -> None:
    """
    Log a latency metric.
    
    Args:
        call_sid: Twilio call SID
        metric_name: Name of the metric
        value_ms: Metric value in milliseconds
        log_file: Path to the log file
    """
    event_data = {
        'event': 'metric.latency',
        'call_sid': call_sid,
        'metric_name': metric_name,
        'value_ms': value_ms
    }
    log_event(log_file, event_data)

def log_error(call_sid: str, error_type: str, error_message: str, 
             error_details: Optional[Dict[str, Any]] = None, log_file: str = None) -> None:
    """
    Log an error event.
    
    Args:
        call_sid: Twilio call SID
        error_type: Type of error
        error_message: Error message
        error_details: Additional error details
        log_file: Path to the log file
    """
    event_data = {
        'event': 'error',
        'call_sid': call_sid,
        'error_type': error_type,
        'error_message': error_message
    }
    
    if error_details:
        event_data['error_details'] = error_details
    
    if log_file:
        log_event(log_file, event_data)
    else:
        # If no log file specified, use standard logging
        logger.error(f"Error [{error_type}]: {error_message}", extra=error_details)

def read_log_file(log_file: str) -> list:
    """
    Read and parse a JSONL log file.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        List of parsed log entries
    """
    try:
        if not os.path.exists(log_file):
            return []
        
        entries = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in log file: {line}")
        
        return entries
        
    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return []

def get_call_metrics(log_file: str) -> Dict[str, Any]:
    """
    Extract metrics from a call log file.
    
    Args:
        log_file: Path to the log file
        
    Returns:
        Dictionary containing call metrics
    """
    try:
        entries = read_log_file(log_file)
        
        metrics = {
            'total_events': len(entries),
            'call_start_time': None,
            'call_end_time': None,
            'total_duration_ms': None,
            'audio_events': 0,
            'latency_metrics': [],
            'errors': []
        }
        
        for entry in entries:
            event_type = entry.get('event', '')
            
            if event_type == 'call.start':
                metrics['call_start_time'] = entry.get('ts')
            elif event_type == 'call.end':
                metrics['call_end_time'] = entry.get('ts')
                metrics['total_duration_ms'] = entry.get('duration_ms')
            elif 'audio' in event_type:
                metrics['audio_events'] += 1
            elif event_type == 'metric.latency':
                metrics['latency_metrics'].append({
                    'name': entry.get('metric_name'),
                    'value_ms': entry.get('value_ms'),
                    'timestamp': entry.get('ts')
                })
            elif event_type == 'error':
                metrics['errors'].append({
                    'type': entry.get('error_type'),
                    'message': entry.get('error_message'),
                    'timestamp': entry.get('ts')
                })
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error extracting metrics: {e}")
        return {'error': str(e)}

