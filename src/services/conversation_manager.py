"""
Conversation Manager with State Machine for Voice Evaluations
Handles OpenAI-powered customer responses with conversation tracking
"""

import openai
import logging
from typing import Dict, List, Optional
from datetime import datetime
from src.config import Settings

logger = logging.getLogger(__name__)

class ConversationState:
    """Represents the current state of a conversation"""
    def __init__(self, call_sid: str, system_prompt: str):
        self.call_sid = call_sid
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        self.turn_count = 0
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def add_agent_message(self, message: str):
        """Add an agent message to the conversation"""
        self.messages.append({"role": "assistant", "content": message})
        self.last_activity = datetime.now()
        logger.info(f"[{self.call_sid}] Agent: {message}")
    
    def add_customer_message(self, message: str):
        """Add a customer message to the conversation"""
        self.messages.append({"role": "user", "content": message})
        self.turn_count += 1
        self.last_activity = datetime.now()
        logger.info(f"[{self.call_sid}] Customer: {message}")
    
    def get_recent_context(self, max_messages: int = 6) -> List[Dict[str, str]]:
        """Get recent conversation context for better response generation"""
        # Return last max_messages from conversation (excluding system message)
        recent = self.messages[1:] if len(self.messages) > 1 else []
        return recent[-max_messages:] if len(recent) > max_messages else recent
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for debugging"""
        return f"Call: {self.call_sid}, Turns: {self.turn_count}, Messages: {len(self.messages)}"
    
    def is_repetitive_response(self, new_message: str) -> bool:
        """Check if the new message is too similar to recent customer messages"""
        # Get recent customer messages (user role)
        recent_customer_messages = [
            msg["content"] for msg in self.messages[1:] 
            if msg["role"] == "user"
        ]
        
        if len(recent_customer_messages) < 2:
            return False
        
        # Check if the new message is very similar to the last customer message
        last_message = recent_customer_messages[-1].lower().strip()
        new_message_lower = new_message.lower().strip()
        
        # Simple similarity check - if more than 80% of words are the same
        last_words = set(last_message.split())
        new_words = set(new_message_lower.split())
        
        if len(last_words) > 0 and len(new_words) > 0:
            common_words = last_words.intersection(new_words)
            similarity = len(common_words) / max(len(last_words), len(new_words))
            return similarity > 0.8
        
        return False

class ConversationManager:
    """Manages conversation states and OpenAI interactions"""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        # Initialize with default config file
        settings = Settings("test.config.json")
        # Create OpenAI client - delay initialization to avoid import issues
        self.api_key = settings.OPENAI_API_KEY
        self.openai_client = None
        self.settings = settings
        logger.info("ConversationManager initialized")
    
    def start_conversation(self, call_sid: str, system_prompt: str) -> ConversationState:
        """Start a new conversation"""
        if call_sid in self.conversations:
            logger.warning(f"Conversation {call_sid} already exists, replacing it")
        
        conversation = ConversationState(call_sid, system_prompt)
        self.conversations[call_sid] = conversation
        logger.info(f"Started new conversation: {call_sid}")
        return conversation
    
    def get_conversation(self, call_sid: str) -> Optional[ConversationState]:
        """Get an existing conversation"""
        return self.conversations.get(call_sid)
    
    def end_conversation(self, call_sid: str):
        """End and clean up a conversation"""
        if call_sid in self.conversations:
            conversation = self.conversations[call_sid]
            logger.info(f"Ending conversation: {conversation.get_conversation_summary()}")
            del self.conversations[call_sid]
    
    def generate_customer_response(self, call_sid: str, agent_message: str) -> str:
        """Generate a customer response using OpenAI"""
        conversation = self.get_conversation(call_sid)
        if not conversation:
            logger.error(f"No conversation found for call_sid: {call_sid}")
            raise ValueError(f"No conversation found for call_sid: {call_sid}")
        
        try:
            # Check if we've exceeded max turns
            if conversation.turn_count >= self.settings.OPENAI_MAX_TURNS:
                logger.info(f"Max turns reached for {call_sid}, ending conversation")
                return "Thank you for your help. I think I have enough information now. Have a great day!"
            
            # Prepare messages for OpenAI - build the conversation context properly
            messages = []
            
            # Add system prompt
            messages.append({
                "role": "system", 
                "content": conversation.system_prompt
            })
            
            # Add conversation history (keep last 8 messages for context, excluding system)
            recent_messages = conversation.messages[1:]  # Skip the system message
            if len(recent_messages) > 8:
                recent_messages = recent_messages[-8:]
            
            messages.extend(recent_messages)
            
            # Add specific instruction for customer response
            messages.append({
                "role": "user", 
                "content": f"Respond as the customer to the agent's last message. The agent just said: \"{agent_message}\". Keep your response natural, conversational, and realistic. Address what the agent asked and build on the conversation naturally."
            })
            
            logger.info(f"Generating response for {call_sid} with {len(messages)} messages")
            logger.debug(f"Full conversation context: {messages}")
            
            # Initialize OpenAI client if not already done
            if self.openai_client is None:
                import httpx
                # Create custom httpx client to avoid proxy issues
                http_client = httpx.Client()
                self.openai_client = openai.OpenAI(api_key=self.api_key, http_client=http_client)
                logger.info("OpenAI client initialized successfully with custom httpx client")
            
            # Call OpenAI API with better parameters to avoid repetition
            response = self.openai_client.chat.completions.create(
                model=self.settings.OPENAI_MODEL,
                messages=messages,
                max_tokens=100,  # Shorter responses to avoid rambling
                temperature=0.9,  # Higher temperature for more variation
                presence_penalty=0.3,  # Higher penalty to avoid repetition
                frequency_penalty=0.3,  # Higher penalty to avoid repetitive phrases
                top_p=0.9
            )
            
            customer_response = response.choices[0].message.content.strip()
            
            # Check for repetitive responses and regenerate if needed
            if conversation.is_repetitive_response(customer_response):
                logger.warning(f"Detected repetitive response for {call_sid}, regenerating...")
                
                # Try generating again with different parameters
                messages[-1]["content"] = f"Respond as the customer to the agent's last message with a completely different approach. The agent just said: \"{agent_message}\". Don't repeat previous phrases. Be more specific about what's happening."
                
                try:
                    response = self.openai_client.chat.completions.create(
                        model=self.settings.OPENAI_MODEL,
                        messages=messages,
                        max_tokens=100,
                        temperature=1.0,  # Even higher temperature
                        presence_penalty=0.5,  # Higher penalty
                        frequency_penalty=0.5,  # Higher penalty
                        top_p=0.8
                    )
                    
                    customer_response = response.choices[0].message.content.strip()
                    logger.info(f"Regenerated response for {call_sid}: {customer_response}")
                except Exception as regen_error:
                    logger.error(f"Error during regeneration for {call_sid}: {regen_error}")
                    # Keep the original response if regeneration fails
            
            # Add the customer response to the conversation
            conversation.add_customer_message(customer_response)
            
            logger.info(f"Final response for {call_sid}: {customer_response}")
            return customer_response
            
        except Exception as e:
            logger.error(f"Error generating OpenAI response for {call_sid}: {e}")
            logger.exception("Full traceback for OpenAI error:")
            # Re-raise the exception instead of using fallback
            raise e
    
    def get_conversation_log(self, call_sid: str) -> List[str]:
        """Get a formatted conversation log"""
        conversation = self.get_conversation(call_sid)
        if not conversation:
            return []
        
        log = []
        for i, msg in enumerate(conversation.messages[1:], 1):  # Skip system message
            role = "Agent" if msg["role"] == "assistant" else "Customer"
            log.append(f"[{i}] {role}: {msg['content']}")
        
        return log
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up conversations older than specified hours"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        to_remove = []
        
        for call_sid, conversation in self.conversations.items():
            if conversation.last_activity.timestamp() < cutoff_time:
                to_remove.append(call_sid)
        
        for call_sid in to_remove:
            logger.info(f"Cleaning up old conversation: {call_sid}")
            self.end_conversation(call_sid)

# Global conversation manager instance - initialized lazily
conversation_manager = None

def get_conversation_manager():
    """Get the global conversation manager instance, creating it if needed"""
    global conversation_manager
    if conversation_manager is None:
        conversation_manager = ConversationManager()
    return conversation_manager
