#!/usr/bin/env python3
"""
Call runner for the Voice Evaluation Framework.
Usage: python eval.py coffee
"""
import requests
import json
import time
import sys
import os
import re
from typing import Optional, Dict, Any
from datetime import datetime

def load_config() -> Dict[str, Any]:
    """Load configuration from hardcoded test.config.json."""
    config_file = "test.config.json"
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_file}' not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file: {e}")
        sys.exit(1)

def get_available_scenarios(config: Dict[str, Any]) -> list:
    """Get list of available scenarios from config."""
    return list(config.get("phoneNumbers", {}).keys())

def get_scenario_config(config: Dict[str, Any], scenario_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific scenario."""
    return config.get("phoneNumbers", {}).get(scenario_name)

def create_agent_config(scenario_config: Dict[str, Any], scenario_name: str) -> Dict[str, Any]:
    """Create agent configuration from scenario config."""
    # Use systemPrompt from config if available, otherwise fall back to generic prompt
    system_prompt = scenario_config.get("systemPrompt", 
        f"You are a customer calling for {scenario_config.get('description', 'support')}. Wait for the person to answer and greet you, then explain your issue naturally and briefly. Keep responses conversational and realistic.")
    
    return {
        "systemPrompt": system_prompt,
        "scenario": scenario_name,
        "description": scenario_config.get("description", ""),
        "name": scenario_config.get("name", scenario_name)
    }

def test_health_check(base_url: str) -> bool:
    """Test if the server is running."""
    try:
        response = requests.get(f"{base_url}/")
        print(f"Health check: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def start_call(base_url: str, to_number: str, agent_config: Dict[str, Any]) -> Optional[str]:
    """Start an outbound call."""
    try:
        payload = {
            "to_number": to_number,
            "agent_config": agent_config
        }
        
        response = requests.post(
            f"{base_url}/call/start",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Start call response: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            data = response.json()
            return data.get('call_sid')
        
        return None
        
    except Exception as e:
        print(f"Error starting call: {e}")
        return None

def check_call_status(base_url: str, call_sid: str) -> Optional[Dict[str, Any]]:
    """Check the status of a call."""
    try:
        response = requests.get(f"{base_url}/call/status/{call_sid}")
        if response.status_code == 200:
            status_data = response.json()
            if 'error' in status_data:
                print(f"Error in call status: {status_data.get('error')}")
                return None
            elif status_data.get('status', 'Unknown') not in ['in-progress']:
                print(f"Call status: {status_data.get('status', 'Unknown')}")
                return status_data
            else:
                # Status is 'in-progress', don't show it
                return status_data
        else:
            print(f"Call status check failed: {response.status_code}")
            return None
        
    except Exception as e:
        print(f"Error checking call status: {e}")
        return None

def parse_conversation_log(log_file: str) -> list:
    """Parse conversation log file and extract meaningful conversation entries."""
    conversation_entries = []
    
    if not os.path.exists(log_file):
        return conversation_entries
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse timestamp and speaker - handle new format with Agent/Customer
                match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] (\w+): (.+)', line)
                if match:
                    timestamp, speaker, message = match.groups()
                    
                    # Filter out technical messages and keep only meaningful conversation
                    # Include Agent and Customer messages, exclude system messages and timing entries
                    if (speaker in ['Agent', 'Customer'] and 
                        not message.startswith('CALL_START:') and
                        not message.startswith('SYSTEM_PROMPT:') and
                        not message.startswith('CALL_END:') and
                        not message.startswith('ERROR:') and
                        not message.startswith('finished speaking') and  # Exclude timing entries
                        not message.startswith('[') and 
                        not message.startswith('Audio level:') and
                        not message.startswith('Audio received') and
                        not message.startswith('Speech detected') and
                        not message.startswith('Generating response') and
                        not message.startswith('Speaking') and
                        not message.startswith('Response completed') and
                        not message.startswith('Error:')):
                        
                        conversation_entries.append({
                            'timestamp': timestamp,
                            'speaker': speaker,
                            'message': message
                        })

                        print(f"Conversation log: {message}")
                
    
    except Exception as e:
        print(f"Error parsing conversation log: {e}")
    
    return conversation_entries

def find_conversation_log_file(call_sid: str) -> str:
    """Find the conversation log file for a given call SID."""
    import glob
    
    # Look for conversation log files with the new naming pattern
    pattern = f"logs/conversation-{call_sid}-*.txt"
    matching_files = glob.glob(pattern)
    
    if matching_files:
        # Return the most recent file (in case there are multiple)
        return max(matching_files, key=os.path.getctime)
    else:
        # Fallback to old naming pattern for backward compatibility
        return f"logs/conversation-{call_sid}.txt"

def display_conversation(conversation_entries: list):
    """Display conversation in the requested format."""
    if not conversation_entries:
        print("No conversation entries found.")
        return
    
    for entry in conversation_entries:
        speaker = entry['speaker']
        timestamp = entry['timestamp']
        message = entry['message']
        
        # Format speaker names consistently
        speaker_name = speaker.title()
        if speaker_name == 'Customer':
            speaker_name = 'Customer'
        elif speaker_name == 'Agent':
            speaker_name = 'Agent'
        
        # Display as long readable lines
        print(f"[{timestamp}] {speaker_name}: {message}", flush=True)

def check_conversation_updates(call_sid: str, last_entries_count: int = 0):
    """Check for new conversation entries and return updated count."""
    log_file = find_conversation_log_file(call_sid)
    
    if not os.path.exists(log_file):
        return last_entries_count, []
    
    entries = parse_conversation_log(log_file)
    if len(entries) > last_entries_count:
        new_entries = entries[last_entries_count:]
        return len(entries), new_entries
    
    return last_entries_count, []


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python eval.py coffee")
        sys.exit(1)
    
    scenario_name = sys.argv[1]
    
    # Load configuration from hardcoded test.config.json
    print("Loading configuration from: test.config.json")
    config = load_config()
    
    # Get scenario configuration
    scenario_config = get_scenario_config(config, scenario_name)
    if not scenario_config:
        available_scenarios = get_available_scenarios(config)
        print(f"Error: Scenario '{scenario_name}' not found in config")
        print(f"Available scenarios: {', '.join(available_scenarios)}")
        sys.exit(1)
    
    # Get target number from scenario config
    to_number = scenario_config.get("number")
    if not to_number:
        print(f"Error: No phone number specified in scenario '{scenario_name}'")
        sys.exit(1)
    
    # Get base URL from config
    base_url = f"http://localhost:{config.get('testSettings', {}).get('webhookPort', 8080)}"
    
    print(f"=== Voice Evaluation Framework Call ===")
    print(f"Config file: test.config.json")
    print(f"Scenario: {scenario_name}")
    print(f"Target number: {to_number}")
    print(f"Server URL: {base_url}")
    
    # Test health check
    print("\n1. Testing health check...")
    if not test_health_check(base_url):
        print("Server is not running. Please start it with:")
        print("  python app.py")
        return
    
    # Create agent configuration
    agent_config = create_agent_config(scenario_config, scenario_name)
    print(f"\n2. Agent config: {json.dumps(agent_config, indent=2)}")
    
    # Start a call
    print(f"\n3. Starting call to {to_number}...")
    call_sid = start_call(base_url, to_number, agent_config)
    
    if not call_sid:
        print("Failed to start call")
        return
    
    print(f"Call started with SID: {call_sid}")
    
    # Keep he console clean with no static messages
    
    consecutive_errors = 0
    max_errors = 3
    call_ended = False
    
    # Initialize conversation monitoring variables
    conversation_started = False
    last_entries_count = 0
    
    
    for i in range(120):  # Check for up to 120 iterations (120 seconds)
        status = check_call_status(base_url, call_sid)
        
        if status is None:
            consecutive_errors += 1
            print(f"Error count: {consecutive_errors}/{max_errors}")
            if consecutive_errors >= max_errors:
                print("Too many consecutive errors. Call may have ended.")
                break
        else:
            consecutive_errors = 0  # Reset error count on successful status check
            call_status = status.get('status', 'Unknown')
            
            # Check for terminal call states - only show when call actually ends
            if call_status in ['completed', 'busy', 'no-answer', 'failed', 'canceled', 'completed']:
                print(f"\nCall ended with status: {call_status}")
                if call_status == 'completed':
                    print("Call completed successfully!")
                elif call_status in ['busy', 'no-answer']:
                    print("Call was not answered.")
                elif call_status == 'failed':
                    print("Call failed.")
                elif call_status == 'canceled':
                    print("Call was canceled.")
                call_ended = True
                break
            # Don't show any status messages for active calls - only show conversation
        
        # Check for conversation updates
        last_entries_count, new_entries = check_conversation_updates(call_sid, last_entries_count)
        
        if new_entries:
            if not conversation_started:
                print("\n" + "="*80)
                print("CONVERSATION STARTED - LIVE TRANSCRIPT")
                print("="*80)
                conversation_started = True
            
            for entry in new_entries:
                speaker = entry['speaker']
                timestamp = entry['timestamp']
                message = entry['message']
                
                
                # Format speaker name consistently
                speaker_name = speaker.title()
                if speaker_name == 'Customer':
                    speaker_name = 'Customer'
                elif speaker_name == 'Agent':
                    speaker_name = 'Agent'
                
                # Display as long readable lines
                print(f"[{timestamp}] {speaker_name}: {message}", flush=True)
        
        time.sleep(1)  # Wait 1 second between checks
    
    # Display final conversation summary
    if not call_ended:
        print("Call monitoring timeout reached.")
    
    if conversation_started:
        print("\n" + "="*80)
        print("CONVERSATION MONITORING COMPLETED")
        print("="*80)
        
        # Get final conversation entries
        log_file = find_conversation_log_file(call_sid)
        if os.path.exists(log_file):
            final_entries = parse_conversation_log(log_file)
            print(f"\nFinal conversation transcript ({len(final_entries)} entries):")
            print("-" * 80)
            display_conversation(final_entries)
            
            
        else:
            print("No conversation log file found.")
    else:
        print("\nNo conversation detected during monitoring period.")
    
    print("\nCall evaluation completed!")

if __name__ == "__main__":
    main()
