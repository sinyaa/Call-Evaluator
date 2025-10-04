# Voice Evaluation Framework - Setup Guide

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configuration Setup
The application uses `test.config.json` for all configuration. Update the following values in your `test.config.json`:

```json
{
  "phoneNumbers": {
    "coffee": {
      "name": "Contoso Coffee",
      "number": "+1234567890",
      "description": "Coffee machine support and troubleshooting"
    }
  },
  "testSettings": {
    "fromNumber": "+1234567890",
    "ngrokUrl": "https://your-ngrok-url.ngrok-free.app",
    "webhookPort": 8080,
    "silenceThreshold": 700,
    "silenceWaitDuration": 600,
    "recordingEnabled": true,
    "verbose": false
  },
  "twilio": {
    "accountSid": "your_twilio_account_sid",
    "authToken": "your_twilio_auth_token"
  },
  "openai": {
    "apiKey": "your_openai_api_key",
    "model": "gpt-4",
    "maxTurns": 10
  }
}
```

### 3. Start the Application
```bash
make dev

# Or directly with python
python app.py
```

### 4. Expose with ngrok
```bash
make ngrok
```

Update your `test.config.json` file with the ngrok URL in the `testSettings.ngrokUrl` field.

## 📞 Making Your First Call

### Option 1: Using the Call Runner
```bash
# Using make command
make call

# Or directly with python
python run_call.py coffee
```

### Option 2: Using the API Directly
```bash
curl -X POST http://localhost:8080/call/start \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+1234567890",
    "agent_config": {
      "systemPrompt": "You are a helpful voice assistant."
    }
  }'
```

## 🏗️ Architecture Overview

### Core Components

1. **TwiML Webhook** (`/twiml/voice`)
   - Receives incoming calls from Twilio
   - Returns TwiML to establish WebSocket connection

2. **WebSocket Handler** (`/websocket/stream`)
   - Handles real-time bidirectional audio streaming
   - Manages conversation flow with OpenAI

3. **Session Management** (`TwilioSession`)
   - Coordinates audio streaming between Twilio and OpenAI
   - Implements Voice Activity Detection (VAD)
   - Logs detailed metrics and events

4. **Audio Processing** (`audio.py`)
   - Converts between Twilio μ-law and OpenAI PCM16 formats
   - Handles sample rate conversion (8kHz ↔ 16kHz)

5. **OpenAI Integration** (`openai_realtime.py`)
   - Bridges to OpenAI's Realtime API
   - Manages conversation state and audio streaming

## 🔧 API Endpoints

- `GET /` - Health check
- `POST /twiml/voice` - Twilio webhook for call initiation
- `WebSocket /websocket/stream` - Real-time audio streaming
- `POST /call/start` - Start an outbound call
- `GET /call/status/{call_sid}` - Get call status

## 📊 Monitoring & Logging

- All events are logged to `logs/session-{call_sid}.jsonl`
- Metrics include latency, audio events, and errors
- Use the logger module to extract call analytics

## 🛠️ Configuration

### Twilio Setup
1. Get your Account SID and Auth Token from Twilio Console
2. Update `test.config.json` with your Twilio credentials in the `twilio` section
3. Update the `fromNumber` in `testSettings` with your Twilio phone number
4. Purchase a phone number for outbound calls

### OpenAI Setup
1. Get an API key from OpenAI
2. Update `test.config.json` with your API key in the `openai` section
3. Ensure you have access to the Realtime API
4. Update the model name if needed

## 🐛 Troubleshooting

### Common Issues

1. **WebSocket Connection Failed**
   - Check ngrok URL is correct in `.env`
   - Ensure Twilio webhooks point to your ngrok URL

2. **Audio Not Working**
   - Verify OpenAI API key is valid
   - Check audio format conversion functions

3. **Call Not Starting**
   - Verify Twilio credentials
   - Check phone number format (+1XXXXXXXXXX)

### Debug Mode
Set `"verbose": true` in your `test.config.json` testSettings for detailed logging.

## 📝 Example Usage

### Simple Usage

The framework is designed to be simple and focused:

**test.config.json:**
```json
{
  "phoneNumbers": {
    "support": {
      "name": "Tech Support",
      "number": "+1234567890",
      "description": "Technical support line"
    }
  },
  "testSettings": {
    "fromNumber": "+1234567891",
    "ngrokUrl": "https://your-ngrok-url.ngrok-free.app",
    "silenceThreshold": 500,
    "recordingEnabled": true
  },
  "twilio": { ... },
  "openai": { ... }
}
```

**Usage:**
```bash
# Start server
python app.py

# Make call using coffee scenario
python run_call.py coffee
```

### Framework Benefits

The framework is now streamlined and simple:
- ✅ **Single config file** - Uses `test.config.json` only
- ✅ **Scenario parameter** - Pass scenario name like `coffee`
- ✅ **Hardcoded config** - No config file parameter needed
- ✅ **Easy to use** - Just pass scenario name
