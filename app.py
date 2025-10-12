from fastapi import FastAPI, WebSocket, Request, Query
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
import base64
import json
import logging
from typing import Dict, Any
import uvicorn
import uuid
import os
from datetime import datetime, timedelta

from src.config import Settings
from src.services.twilio_client import make_outbound_call
import sys
import os
import asyncio
import requests

# Configure logging (suppress debug output in server logs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Reduce verbosity of noisy third-party loggers
for noisy_logger_name in [
    'twilio',
    'twilio.http_client',
    'urllib3',
    'multipart',
    'multipart.multipart'
]:
    logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

app = FastAPI(title='Voice Evaluation Framework')

# Ensure static audio directory exists and expose it at /audio
try:
    os.makedirs('audio', exist_ok=True)
except Exception:
    pass
app.mount('/audio', StaticFiles(directory='audio'), name='audio')

# Load settings from hardcoded test.config.json
config_file = "test.config.json"
settings = Settings(config_file)

# In-memory store for call configurations
call_configs = {}

@app.get('/')
async def root():
    return {'status': 'ok', 'message': 'Voice Evaluation Framework is running'}

@app.post('/twiml/voice')
async def twiml_voice(request: Request, cfg: str = Query(...)):
    """
    TwiML webhook: plays a greeting, then opens a *bidirectional* media stream.
    """
    try:
        
        # Decode the configuration
        config_data = json.loads(base64.urlsafe_b64decode(cfg).decode())
        
        # Get the call SID from Twilio headers or form data
        call_sid = request.headers.get('X-Twilio-CallSid')
        if not call_sid:
            # Try to get it from form data
            try:
                form_data = await request.form()
                call_sid = form_data.get('CallSid', 'default_call')
                logger.info(f"Form data: {dict(form_data)}")
            except Exception as e:
                logger.warning(f"Could not get form data: {e}")
                call_sid = 'default_call'
        if not call_sid:
            call_sid = 'default_call'
        
        logger.info(f"Call SID: {call_sid}")
        
        # Store the configuration with the call SID
        call_configs[call_sid] = config_data
        
        # Initialize conversation manager
        from src.services.conversation_manager import get_conversation_manager
        system_prompt = config_data.get('systemPrompt', 'You are a customer calling for support. Wait for the person to answer and greet you, then explain your issue naturally and briefly. Keep responses conversational and realistic.')
        conversation_manager = get_conversation_manager()
        conversation_manager.start_conversation(call_sid, system_prompt)
        
        # Log call start using centralized logger
        from src.services.conversation_logger import conversation_logger
        
        # Extract phone numbers from request headers
        to_number = request.headers.get('X-Twilio-To', 'Unknown')
        from_number = request.headers.get('X-Twilio-From', 'Unknown')
        
        conversation_logger.log_call_start(call_sid, to_number, from_number, config_data)
        conversation_logger.log_waiting_for_agent(call_sid)
        
        twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Record recordingStatusCallback="{settings.PUBLIC_BASE_URL}/webhook/call-status" />
  </Start>
  <Gather input="speech" language="en-US" timeout="5" speechTimeout="1" actionOnEmptyResult="true" enhanced="true" bargeIn="true" action="/twiml/gather" method="POST">
  </Gather>
  <Say voice="Polly.Joanna-Neural" language="en-US">
I didn't hear anything. Let me try calling back later. Goodbye!
  </Say>
</Response>""".strip()
        
        logger.info(f"TwiML Response: {twiml_response}")
        return Response(content=twiml_response, media_type="application/xml")
    
    except Exception as e:
        logger.exception("Error in twiml_voice")
        twiml_error = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna-Neural" language="en-US">
Sorry, there was an error connecting. Please try again later.
  </Say>
</Response>"""
        return Response(content=twiml_error, media_type="application/xml")

@app.post('/twiml/gather')
async def twiml_gather(request: Request):
    """Handle speech input from Twilio Gather verb"""
    try:
        # Capture the timestamp when this webhook was received
        webhook_received_time = datetime.now()
        
        logger.info("=== TwiML Gather Webhook Called ===")
        
        # Get form data from Twilio
        form_data = await request.form()
        logger.info(f"Gather form data: {dict(form_data)}")
        
        call_sid = form_data.get('CallSid', 'unknown')
        speech_result = form_data.get('SpeechResult', '')
        confidence = form_data.get('Confidence', '0')
        
        logger.info(f"Call SID: {call_sid}")
        logger.info(f"Speech Result: {speech_result}")
        logger.info(f"Confidence: {confidence}")
        
        # Log agent message when webhook is received from Twilio (simple and accurate)
        from src.services.conversation_logger import conversation_logger
        conversation_logger.log_agent_message(call_sid, speech_result, confidence, webhook_received_time)
        
        # Generate customer response using conversation manager
        from src.services.conversation_manager import get_conversation_manager
        conversation_manager = get_conversation_manager()
        
        # Get the stored configuration for this call
        config_data = call_configs.get(call_sid, {})
        system_prompt = config_data.get('systemPrompt', 'You are a customer calling for support. Wait for the person to answer and greet you, then explain your issue naturally and briefly. Keep responses conversational and realistic.')
        
        # Ensure conversation exists
        conversation = conversation_manager.get_conversation(call_sid)
        if not conversation:
            conversation = conversation_manager.start_conversation(call_sid, system_prompt)
        
        # Add agent message to conversation first
        conversation.add_agent_message(speech_result)
        
        # Generate response using the conversation manager
        try:
            customer_response = conversation_manager.generate_customer_response(call_sid, speech_result)
            
        except Exception as e:
            logger.error(f"Failed to generate customer response for {call_sid}: {e}")
            logger.exception("Full traceback for conversation error:")
            
            # Log the error
            conversation_logger.log_error(call_sid, f"OpenAI response generation failed: {e}")
            
            # End the call with an error message
            twiml_error = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna-Neural" language="en-US">I'm sorry, I'm experiencing technical difficulties. Goodbye!</Say>
  <Hangup/>
</Response>"""
            logger.error(f"Call {call_sid} failed due to OpenAI error, ending call")
            return Response(content=twiml_error, media_type="application/xml")
        
        # Log customer message right before sending to Twilio
        customer_send_time = datetime.now()
        
        # Log customer message
        conversation_logger.log_customer_message(call_sid, customer_response)
        
        # Escape XML special characters in customer response
        import html
        escaped_response = html.escape(customer_response)

        # Debounce before speaking to reduce overlap when agent continues talking
        try:
            await asyncio.sleep(max(0, settings.SILENCE_WAIT_DURATION) / 1000.0)
        except Exception:
            pass

        # End the call after max turns; otherwise continue gathering
        if conversation.turn_count >= settings.OPENAI_MAX_TURNS:
            twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna-Neural" language="en-US">{escaped_response}</Say>
  <Say voice="Polly.Joanna-Neural" language="en-US">Thank you for your time. Goodbye!</Say>
  <Hangup/>
</Response>""".strip()
        else:
            twiml_response = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" language="en-US" timeout="5" speechTimeout="1" actionOnEmptyResult="true" enhanced="true" bargeIn="true" action="/twiml/gather" method="POST">
    <Say voice="Polly.Joanna-Neural" language="en-US">{escaped_response}</Say>
    <Play>{settings.PUBLIC_BASE_URL}/audio/click.wav</Play>
  </Gather>
</Response>""".strip()
        logger.info(f"TwiML Response: {twiml_response}")
        return Response(content=twiml_response, media_type="application/xml")
        
    except Exception as e:
        logger.exception("Error in twiml_gather")
        twiml_error = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna-Neural" language="en-US">Sorry, there was an error processing your response. </Say>
</Response>"""
        return Response(content=twiml_error, media_type="application/xml")

@app.post('/call/start')
async def start_call(request: Request):
    """
    Start an outbound call to a phone number.
    """
    try:
        logger.info("=== Call Start Request ===")
        body = await request.json()
        to_number = body.get('to_number')
        agent_config = body.get('agent_config', {})
        
        logger.info(f"To number: {to_number}")
        logger.info(f"Agent config: {agent_config}")
        
        if not to_number:
            logger.error("No to_number provided")
            return {'error': 'to_number is required'}
        
        # Make the outbound call
        logger.info(f"Making outbound call to {to_number}")
        call_sid = make_outbound_call(
            to_number=to_number,
            agent_cfg=agent_config,
            public_base_url=settings.PUBLIC_BASE_URL,
            from_number=settings.TWILIO_CALLER_NUMBER
        )
        
        logger.info(f"Call SID returned: {call_sid}")
        
        # Store the call_sid with the agent_config for later retrieval
        # We'll use the call_sid as the key since it's unique
        call_configs[call_sid] = agent_config
        logger.info(f"Stored config for call_sid {call_sid}")
        
        return {
            'status': 'success',
            'call_sid': call_sid,
            'message': f'Call initiated to {to_number}'
        }
    
    except Exception as e:
        logger.error(f"Error starting call: {e}")
        logger.exception("Full traceback:")
        return {'error': str(e)}

@app.get('/call/status/{call_sid}')
async def get_call_status(call_sid: str):
    """
    Get the status of a call.
    """
    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        call = client.calls(call_sid).fetch()
        
        # Handle different Twilio API response formats
        from_number = getattr(call, 'from_', None) or getattr(call, 'from', None)
        to_number = getattr(call, 'to', None)
        
        return {
            'call_sid': call_sid,
            'status': call.status,
            'direction': call.direction,
            'from': from_number,
            'to': to_number,
            'start_time': call.start_time.isoformat() if call.start_time else None,
            'end_time': call.end_time.isoformat() if call.end_time else None,
            'duration': call.duration
        }
    
    except Exception as e:
        logger.error(f"Error getting call status: {e}")
        return {'error': str(e)}

@app.post('/webhook/call-status')
async def call_status_webhook(request: Request):
    """Handle Twilio call status webhooks"""
    try:
        logger.info("=== Call Status Webhook ===")
        
        # Get form data from Twilio
        form_data = await request.form()
        logger.info(f"Call status form data: {dict(form_data)}")
        
        call_sid = form_data.get('CallSid', 'unknown')
        call_status = form_data.get('CallStatus', 'unknown')
        call_duration = form_data.get('CallDuration', '0')
        
        logger.info(f"Call SID: {call_sid}")
        logger.info(f"Call Status: {call_status}")
        logger.info(f"Call Duration: {call_duration}")

        # Handle Recording Status Callback (download recording when available)
        recording_sid = form_data.get('RecordingSid')
        recording_url = form_data.get('RecordingUrl')
        recording_status = form_data.get('RecordingStatus')
        if recording_sid and recording_url and recording_status in ['completed', 'available']:
            try:
                # Twilio requires auth to download recording; append .mp3 to URL
                mp3_url = f"{recording_url}.mp3"
                auth = (settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
                resp = requests.get(mp3_url, auth=auth, timeout=30)
                resp.raise_for_status()
                recordings_dir = os.path.join('logs', 'recordings')
                os.makedirs(recordings_dir, exist_ok=True)
                out_path = os.path.join(recordings_dir, f"{call_sid}-{recording_sid}.mp3")
                with open(out_path, 'wb') as f:
                    f.write(resp.content)
                logger.info(f"Saved recording to {out_path}")
            except Exception as rec_err:
                logger.error(f"Failed to download recording {recording_sid} for {call_sid}: {rec_err}")
        
        # Log call end if status indicates call ended
        if call_status in ['completed', 'failed', 'busy', 'no-answer']:
            from src.services.conversation_logger import conversation_logger
            conversation_logger.log_call_end(call_sid, call_status)
        
        return {"status": "success", "message": "Call status logged"}
        
    except Exception as e:
        logger.error(f"Error in call status webhook: {e}")
        logger.exception("Full traceback:")
        return {"status": "error", "message": str(e)}

@app.get('/conversation/summary/{call_sid}')
async def get_conversation_summary(call_sid: str):
    """Get conversation summary for a call"""
    try:
        from src.services.conversation_logger import conversation_logger
        summary = conversation_logger.get_conversation_summary(call_sid)
        return summary
    except Exception as e:
        logger.error(f"Error getting conversation summary: {e}")
        return {"error": str(e)}

@app.get('/conversation/logs/{call_sid}')
async def get_conversation_log(call_sid: str):
    """Get full conversation log for a call"""
    try:
        from src.services.conversation_logger import conversation_logger
        log_file = conversation_logger.get_conversation_log_file(call_sid)
        
        if not os.path.exists(log_file):
            return {"error": "Conversation log not found"}
        
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        return {"call_sid": call_sid, "log_content": content}
    except Exception as e:
        logger.error(f"Error getting conversation log: {e}")
        return {"error": str(e)}

@app.post('/conversation/cleanup')
async def cleanup_conversations():
    """Clean up old conversations"""
    try:
        from src.services.conversation_manager import get_conversation_manager
        conversation_manager = get_conversation_manager()
        conversation_manager.cleanup_old_conversations(max_age_hours=1)
        return {"status": "success", "message": "Old conversations cleaned up"}
    except Exception as e:
        logger.error(f"Error cleaning up conversations: {e}")
        return {"status": "error", "message": str(e)}

@app.post('/error/')
async def twilio_error(request: Request):
    """Handle Twilio error events and log them for debugging."""
    try:
        logger.error("=== Twilio Error Event ===")
        
        # Get the raw body to see what Twilio is sending
        body = await request.body()
        logger.error(f"Raw error body: {body}")
        
        # Try to parse as JSON
        try:
            error_data = await request.json()
            logger.error(f"Parsed error data: {error_data}")
        except Exception as json_error:
            logger.error(f"Could not parse error data as JSON: {json_error}")
            # Try to parse as form data
            try:
                form_data = await request.form()
                error_data = dict(form_data)
                logger.error(f"Parsed error data as form: {error_data}")
            except Exception as form_error:
                logger.error(f"Could not parse error data as form: {form_error}")
                error_data = {"raw_body": body.decode('utf-8', errors='ignore')}
        
        # Log error details
        logger.error(f"Error timestamp: {datetime.now().isoformat()}")
        logger.error(f"Request headers: {dict(request.headers)}")
        logger.error(f"Request method: {request.method}")
        logger.error(f"Request URL: {request.url}")
        
        # Store error in a separate log file for easier debugging
        error_log_file = f"logs/twilio-errors-{datetime.now().strftime('%Y%m%d')}.log"
        with open(error_log_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== Twilio Error - {datetime.now().isoformat()} ===\n")
            f.write(f"Headers: {dict(request.headers)}\n")
            f.write(f"Body: {error_data}\n")
            f.write("=" * 50 + "\n")
        
        logger.error(f"Error logged to: {error_log_file}")
        
        return {"status": "error_received", "message": "Error logged successfully"}
        
    except Exception as e:
        logger.error(f"Error handling Twilio error event: {e}")
        logger.exception("Full traceback:")
        return {"status": "error", "message": str(e)}

@app.get('/recordings/{call_sid}')
async def download_recordings(call_sid: str):
    """Return list of local recording files for a given Call SID."""
    try:
        recordings_dir = os.path.join('logs', 'recordings')
        if not os.path.isdir(recordings_dir):
            return {"recordings": []}
        files = [
            f for f in os.listdir(recordings_dir)
            if f.startswith(call_sid + '-') and f.endswith('.mp3')
        ]
        return {"recordings": files}
    except Exception as e:
        logger.error(f"Error listing recordings: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
