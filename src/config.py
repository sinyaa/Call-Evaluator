import json
import os
from typing import Dict, Any

class Settings:
    """Configuration manager that reads from any config.json file"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json"""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {self.config_file} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_file}: {e}")
    
    @property
    def TWILIO_ACCOUNT_SID(self) -> str:
        return self._config["twilio"]["accountSid"]
    
    @property
    def TWILIO_AUTH_TOKEN(self) -> str:
        return self._config["twilio"]["authToken"]
    
    @property
    def TWILIO_CALLER_NUMBER(self) -> str:
        return self._config["testSettings"]["fromNumber"]
    
    @property
    def PUBLIC_BASE_URL(self) -> str:
        return self._config["testSettings"]["ngrokUrl"]
    
    @property
    def OPENAI_API_KEY(self) -> str:
        return self._config["openai"]["apiKey"]
    
    @property
    def OPENAI_MODEL(self) -> str:
        return self._config["openai"]["model"]
    
    @property
    def OPENAI_REALTIME_MODEL(self) -> str:
        return self._config["openai"]["model"]
    
    @property
    def OPENAI_MAX_TURNS(self) -> int:
        return self._config["openai"]["maxTurns"]
    
    @property
    def SILENCE_THRESHOLD(self) -> int:
        return self._config["testSettings"]["silenceThreshold"]
    
    @property
    def SILENCE_WAIT_DURATION(self) -> int:
        return self._config["testSettings"]["silenceWaitDuration"]
    
    @property
    def RECORDING_ENABLED(self) -> bool:
        return self._config["testSettings"]["recordingEnabled"]
    
    @property
    def VERBOSE(self) -> bool:
        return self._config["testSettings"]["verbose"]
    
    def get_phone_number(self, scenario: str) -> Dict[str, Any]:
        """Get phone number configuration for a scenario"""
        return self._config["phoneNumbers"].get(scenario, {})
    
    def get_test_settings(self) -> Dict[str, Any]:
        """Get test settings configuration"""
        return self._config["testSettings"]
