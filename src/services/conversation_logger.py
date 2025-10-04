"""
Centralized conversation logging system for voice evaluations
"""

import os
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConversationLogger:
    """Handles logging of conversations and call events"""
    
    def __init__(self, logs_dir: str = "logs"):
        self.logs_dir = logs_dir
        self.ensure_logs_directory()
        
        
    
    
    def ensure_logs_directory(self):
        """Ensure the logs directory exists"""
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
    
    def get_conversation_log_file(self, call_sid: str) -> str:
        """Get the conversation log file path for a call"""
        return os.path.join(self.logs_dir, f"conversation-{call_sid}.txt")
    
    def get_call_events_log_file(self, call_sid: str) -> str:
        """Get the call events log file path for a call"""
        return os.path.join(self.logs_dir, f"call-events-{call_sid}.jsonl")
    
    def log_call_start(self, call_sid: str, to_number: str, from_number: str, 
                      agent_config: Dict[str, Any]) -> None:
        """Log call start event"""
        timestamp = datetime.now()
        
        event_data = {
            "timestamp": timestamp.isoformat(),
            "event": "call_start",
            "call_sid": call_sid,
            "to_number": to_number,
            "from_number": from_number,
            "agent_config": agent_config
        }
        
        # Log to call events file
        call_events_file = self.get_call_events_log_file(call_sid)
        with open(call_events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data) + "\n")
        
        # Log to conversation file
        conversation_file = self.get_conversation_log_file(call_sid)
        with open(conversation_file, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] CALL_START: {to_number} -> {from_number}\n")
            f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] SYSTEM_PROMPT: {agent_config.get('systemPrompt', 'Default prompt')}\n")
            f.write("-" * 80 + "\n")
        
        logger.info(f"Call started: {call_sid} ({to_number} -> {from_number})")
    
    def log_call_end(self, call_sid: str, reason: str = "unknown") -> None:
        """Log call end event"""
        timestamp = datetime.now()
        
        event_data = {
            "timestamp": timestamp.isoformat(),
            "event": "call_end",
            "call_sid": call_sid,
            "reason": reason
        }
        
        # Log to call events file
        call_events_file = self.get_call_events_log_file(call_sid)
        with open(call_events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event_data) + "\n")
        
        # Log to conversation file
        conversation_file = self.get_conversation_log_file(call_sid)
        with open(conversation_file, "a", encoding="utf-8") as f:
            f.write("-" * 80 + "\n")
            f.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] CALL_END: {reason}\n")
        
        logger.info(f"Call ended: {call_sid} (reason: {reason})")
    
    def log_agent_message(self, call_sid: str, message: str, confidence: Optional[str] = None, timestamp: Optional[datetime] = None) -> None:
        """Log agent message"""
        if timestamp is None:
            timestamp = datetime.now()

        conversation_file = self.get_conversation_log_file(call_sid)
        confidence_str = f" (confidence: {confidence})" if confidence else ""
        log_entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Agent: {message}{confidence_str}\n"
        
        with open(conversation_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        print(log_entry.strip(), flush=True)
        logger.info(f"[{call_sid}] Agent: {message}")

    
    def log_customer_message(self, call_sid: str, message: str, timestamp: Optional[datetime] = None) -> None:
        """Log customer message"""
        if timestamp is None:
            timestamp = datetime.now()

        conversation_file = self.get_conversation_log_file(call_sid)
        log_entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Customer: {message}\n"
        
        with open(conversation_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        print(log_entry.strip(), flush=True)
        logger.info(f"[{call_sid}] Customer: {message}")

    
    def log_waiting_for_agent(self, call_sid: str) -> None:
        """Log that we're waiting for agent to answer"""
        timestamp = datetime.now()
        
        conversation_file = self.get_conversation_log_file(call_sid)
        log_entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Customer: [Waiting for agent to answer and greet...]\n"
        
        with open(conversation_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        print(log_entry.strip(), flush=True)
        logger.info(f"[{call_sid}] Customer waiting for agent")
    
    def log_error(self, call_sid: str, error_message: str) -> None:
        """Log error during conversation"""
        timestamp = datetime.now()
        
        conversation_file = self.get_conversation_log_file(call_sid)
        log_entry = f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] ERROR: {error_message}\n"
        
        with open(conversation_file, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        print(log_entry.strip(), flush=True)
        logger.error(f"[{call_sid}] Error: {error_message}")
    
    def get_conversation_summary(self, call_sid: str) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        conversation_file = self.get_conversation_log_file(call_sid)
        
        if not os.path.exists(conversation_file):
            return {"error": "Conversation log not found"}
        
        try:
            with open(conversation_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            agent_messages = 0
            customer_messages = 0
            total_chars = 0
            
            for line in lines:
                if "Agent:" in line:
                    agent_messages += 1
                    # Extract message content after "Agent: "
                    message = line.split("Agent:", 1)[1].strip()
                    total_chars += len(message)
                elif "Customer:" in line:
                    customer_messages += 1
                    # Extract message content after "Customer: "
                    message = line.split("Customer:", 1)[1].strip()
                    total_chars += len(message)
            
            return {
                "call_sid": call_sid,
                "agent_messages": agent_messages,
                "customer_messages": customer_messages,
                "total_messages": agent_messages + customer_messages,
                "total_characters": total_chars,
                "log_file": conversation_file
            }
        except Exception as e:
            return {"error": f"Failed to parse conversation: {e}"}

# Global conversation logger instance
conversation_logger = ConversationLogger()
