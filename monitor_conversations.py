#!/usr/bin/env python3
"""
Real-time conversation monitor for voice evaluations
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

def monitor_conversations():
    """Monitor conversation logs in real-time"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("Logs directory not found. Make sure the app is running.")
        return
    
    print("=== Conversation Monitor ===")
    print("Monitoring conversation logs in real-time...")
    print("Press Ctrl+C to stop")
    print("-" * 60)
    
    # Track which files we've already seen
    seen_files = set()
    
    try:
        while True:
            # Find all conversation log files
            conversation_files = list(logs_dir.glob("conversation-*.txt"))
            
            for log_file in conversation_files:
                if str(log_file) not in seen_files:
                    print(f"\n📞 NEW CALL DETECTED: {log_file.name}")
                    seen_files.add(str(log_file))
                
                # Read the file and print new content
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Print the content (you might want to implement more sophisticated
                    # logic to only show new lines)
                    if content.strip():
                        lines = content.strip().split('\n')
                        for line in lines:
                            if line.strip():
                                print(f"[{log_file.name}] {line}")
                
                except Exception as e:
                    print(f"Error reading {log_file}: {e}")
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

def list_recent_conversations():
    """List recent conversation logs"""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("Logs directory not found.")
        return
    
    conversation_files = list(logs_dir.glob("conversation-*.txt"))
    
    if not conversation_files:
        print("No conversation logs found.")
        return
    
    print("=== Recent Conversations ===")
    for log_file in sorted(conversation_files, key=lambda x: x.stat().st_mtime, reverse=True):
        stat = log_file.stat()
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        size = stat.st_size
        
        print(f"📞 {log_file.name}")
        print(f"   Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Size: {size} bytes")
        print()

def show_conversation_summary(call_sid: str):
    """Show summary of a specific conversation"""
    try:
        from src.services.conversation_logger import conversation_logger
        summary = conversation_logger.get_conversation_summary(call_sid)
        
        if "error" in summary:
            print(f"Error: {summary['error']}")
            return
        
        print(f"=== Conversation Summary: {call_sid} ===")
        print(f"Agent messages: {summary['agent_messages']}")
        print(f"Customer messages: {summary['customer_messages']}")
        print(f"Total messages: {summary['total_messages']}")
        print(f"Total characters: {summary['total_characters']}")
        print(f"Log file: {summary['log_file']}")
        
    except Exception as e:
        print(f"Error getting summary: {e}")

def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "monitor":
            monitor_conversations()
        elif command == "list":
            list_recent_conversations()
        elif command == "summary" and len(sys.argv) > 2:
            show_conversation_summary(sys.argv[2])
        else:
            print("Usage:")
            print("  python monitor_conversations.py monitor    # Monitor conversations in real-time")
            print("  python monitor_conversations.py list       # List recent conversations")
            print("  python monitor_conversations.py summary <call_sid>  # Show conversation summary")
    else:
        print("Conversation Monitor")
        print("Usage:")
        print("  python monitor_conversations.py monitor    # Monitor conversations in real-time")
        print("  python monitor_conversations.py list       # List recent conversations")
        print("  python monitor_conversations.py summary <call_sid>  # Show conversation summary")

if __name__ == "__main__":
    main()
