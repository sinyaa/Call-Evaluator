import base64, json
from twilio.rest import Client
from ..config import Settings

settings = Settings("test.config.json")
client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)

def make_outbound_call(to_number: str, agent_cfg: dict, public_base_url: str, from_number: str) -> str:
    cfg_b64 = base64.urlsafe_b64encode(json.dumps(agent_cfg).encode()).decode()
    twiml_url = f"{public_base_url}/twiml/voice?cfg={cfg_b64}"
    status_callback_url = f"{public_base_url}/webhook/call-status"
    # Recording status callback may reuse call-status webhook for simplicity
    recording_status_callback_url = status_callback_url
    
    call = client.calls.create(
        to=to_number, 
        from_=from_number, 
        url=twiml_url,
        status_callback=status_callback_url,
        status_callback_event=['completed', 'failed', 'canceled'],
        # Enable/disable recording from config
        record=bool(settings.RECORDING_ENABLED),
        recording_status_callback=recording_status_callback_url,
        recording_status_callback_event=['completed']
    )
    return call.sid
