"""
Conversation Manager for LexiLLM
Manages the flow of conversation, including history tracking
and state management
"""

from typing import List, Optional, Dict, Any, Tuple

from langchain_community.chat_message_histories import ChatMessageHistory

from ..config import CONVERSATION_CONFIG
from ..logger import get_logger
from ..exceptions import ConversationError
from .conversation_state import ConversationState, ConversationStateEnum, is_confirmation, is_rejection

# Get logger
logger = get_logger()

class ConversationManager:
    """
    Manages conversation flow and history for LexiLLM.
    
    This class handles conversation state, history management,
    and tracks the flow of conversation using a state machine
    for reliable context management.
    """
    
    def __init__(self):
        """Initialize the conversation manager."""
        # Create message history for conversation
        self.chat_history = ChatMessageHistory()
        
        # Initialize conversation state with state machine
        self.state = ConversationState()
        self.last_topic = None  # Track the last specific topic
        
        logger.info("Conversation manager initialized with state machine")
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            message: The user's message
        """
        try:
            logger.debug(f"Adding user message to history: {message[:50]}...")
            self.chat_history.add_user_message(message)
        except Exception as e:
            logger.error(f"Error adding user message to history: {str(e)}")
            raise ConversationError(f"Failed to add user message to history: {str(e)}")
    
    def add_ai_message(self, message: str) -> None:
        """
        Add an AI message to the conversation history.
        
        Args:
            message: The AI's message
        """
        try:
            logger.debug(f"Adding AI message to history: {message[:50]}...")
            self.chat_history.add_ai_message(message)
        except Exception as e:
            logger.error(f"Error adding AI message to history: {str(e)}")
            raise ConversationError(f"Failed to add AI message to history: {str(e)}")
    
    def set_topic(self, topic: str) -> None:
        """
        Set the current conversation topic.
        
        Args:
            topic: The current topic
        """
        logger.debug(f"Setting current topic to: {topic[:50]}...")
        self.last_topic = topic
        self.state.current_topic = topic
    
    def get_topic(self) -> Optional[str]:
        """
        Get the current conversation topic.
        
        Returns:
            The current topic or None if not set
        """
        return self.state.current_topic or self.last_topic
    
    def end_conversation(self) -> None:
        """Mark the conversation as ended."""
        logger.info("Marking conversation as ended")
        self.state.transit_to(ConversationStateEnum.ENDING)
        self.state.is_active = False
    
    def is_conversation_active(self) -> bool:
        """
        Check if the conversation is active.
        
        Returns:
            Boolean indicating if the conversation is active
        """
        return self.state.is_active
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in the conversation history.
        
        Returns:
            List of messages with their metadata
        """
        try:
            messages = []
            for i, msg in enumerate(self.chat_history.messages):
                messages.append({
                    "index": i,
                    "role": "user" if msg.type == "human" else "assistant",
                    "content": msg.content
                })
            return messages
        except Exception as e:
            logger.error(f"Error retrieving conversation messages: {str(e)}")
            return []
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        logger.info("Clearing conversation history")
        self.chat_history = ChatMessageHistory()
    
    def manage_chat_history(self, max_messages: Optional[int] = None) -> None:
        """
        Truncate chat history to prevent memory issues with long conversations.
        
        Args:
            max_messages: Maximum number of messages to keep (pairs of user/assistant messages).
                          If None, uses the default from configuration.
        """
        # Get max history size from configuration if not provided
        if max_messages is None:
            max_messages = CONVERSATION_CONFIG["max_history_pairs"]
        
        preserved_start = CONVERSATION_CONFIG["preserve_initial_messages"]
        
        messages = self.chat_history.messages
        
        # If we have more than max_messages*2 messages (user+assistant pairs), truncate
        if len(messages) > max_messages * 2:
            logger.info(f"Truncating chat history from {len(messages)} messages to {max_messages*2}")
            
            # Keep the first few messages (initial greeting) and the most recent messages
            preserved_start = min(preserved_start, len(messages))
            preserved_recent = max_messages * 2 - preserved_start
            
            try:
                # Create a new history with preserved messages
                new_history = ChatMessageHistory()
                
                # Add the initial messages
                for i in range(preserved_start):
                    if i < len(messages):
                        if messages[i].type == "human":
                            new_history.add_user_message(messages[i].content)
                        else:
                            new_history.add_ai_message(messages[i].content)
                
                # Add the most recent messages
                for i in range(len(messages) - preserved_recent, len(messages)):
                    if i >= 0 and i < len(messages):
                        if messages[i].type == "human":
                            new_history.add_user_message(messages[i].content)
                        else:
                            new_history.add_ai_message(messages[i].content)
                
                # Replace the chat history
                self.chat_history = new_history
                logger.debug("Successfully truncated chat history")
            except Exception as e:
                logger.error(f"Error truncating chat history: {str(e)}")
                # Continue with the original history if truncation fails
                pass
                
    # New methods for state management
    
    def start_onboarding(self) -> None:
        """Start the onboarding process."""
        self.state.transit_to(ConversationStateEnum.ONBOARDING)
        logger.info("Started onboarding process")
    
    def complete_onboarding(self) -> None:
        """Complete the onboarding process."""
        self.state.transit_to(ConversationStateEnum.IDLE)
        logger.info("Completed onboarding process")
    
    def is_in_onboarding(self) -> bool:
        """Check if in onboarding state."""
        return self.state.state == ConversationStateEnum.ONBOARDING
    
    def start_info_collection(self, attribute: str) -> None:
        """
        Start collecting a specific attribute.
        
        Args:
            attribute: The attribute to collect
        """
        self.state.start_info_collection(attribute)
    
    def end_info_collection(self) -> None:
        """End the information collection process."""
        self.state.end_info_collection()
        
        # If we have a pending query, return to PROCESSING state
        if self.state.pending_query:
            self.state.transit_to(ConversationStateEnum.PROCESSING)
        else:
            # Otherwise, go back to IDLE
            self.state.transit_to(ConversationStateEnum.IDLE)
    
    def save_pending_query(self, message: str, intent: str, topic: str) -> None:
        """
        Save a query to be processed after collecting information.
        
        Args:
            message: The user's message
            intent: The detected intent
            topic: The topic of the query
        """
        self.state.save_pending_query(message, intent, topic)
    
    def get_pending_query(self) -> Optional[Dict[str, Any]]:
        """
        Get the pending query if one exists.
        
        Returns:
            The pending query information or None
        """
        if self.state.pending_query:
            return {
                "message": self.state.pending_query.message,
                "intent": self.state.pending_query.intent,
                "topic": self.state.pending_query.topic
            }
        return None
    
    def clear_pending_query(self) -> None:
        """Clear any pending query."""
        self.state.clear_pending_query()
    
    def is_collecting_info(self) -> bool:
        """
        Check if currently collecting information.
        
        Returns:
            True if in information collection state
        """
        return self.state.is_collecting_info()
    
    def get_collecting_attribute(self) -> Optional[str]:
        """
        Get the attribute currently being collected.
        
        Returns:
            The attribute name or None if not collecting
        """
        return self.state.get_info_collection_attribute()
    
    def set_awaiting_confirmation(self, context: Dict[str, Any] = None) -> None:
        """
        Set the state to awaiting confirmation.
        
        Args:
            context: Context information about what needs confirmation
        """
        context = context or {"type": "general"}
        self.state.set_awaiting_confirmation(context)
    
    def is_awaiting_confirmation(self) -> bool:
        """
        Check if awaiting user confirmation.
        
        Returns:
            True if awaiting confirmation
        """
        return self.state.is_awaiting_confirmation()
    
    def get_confirmation_context(self) -> Optional[Dict[str, Any]]:
        """
        Get the context for the current confirmation.
        
        Returns:
            Confirmation context or None if not awaiting confirmation
        """
        if self.is_awaiting_confirmation():
            return self.state.confirmation_context
        return None
    
    def clear_confirmation(self) -> None:
        """Clear the confirmation state."""
        self.state.clear_confirmation()
        # Go back to IDLE state
        self.state.transit_to(ConversationStateEnum.IDLE)
    
    def process_confirmation_response(self, message: str) -> Tuple[bool, bool]:
        """
        Process a response to a confirmation request.
        
        Args:
            message: The user's message
            
        Returns:
            Tuple of (is_processed, is_confirmed)
        """
        if not self.is_awaiting_confirmation():
            return False, False
            
        # Check if the message is a confirmation or rejection
        if is_confirmation(message):
            self.clear_confirmation()
            return True, True
        elif is_rejection(message):
            self.clear_confirmation()
            return True, False
            
        # If it's neither, don't consider it a confirmation response
        # This allows the user to change the subject
        self.clear_confirmation()
        return False, False
    
    def set_intent(self, intent: str) -> None:
        """
        Set the current intent.
        
        Args:
            intent: The intent to set
        """
        self.state.current_intent = intent
        logger.debug(f"Set current intent to: {intent}")
    
    def get_intent(self) -> Optional[str]:
        """
        Get the current intent.
        
        Returns:
            The current intent or None if not set
        """
        return self.state.current_intent
    
    def set_processing(self) -> None:
        """Set state to processing a user query."""
        self.state.transit_to(ConversationStateEnum.PROCESSING)
    
    def set_responding(self) -> None:
        """Set state to responding to a user query."""
        self.state.transit_to(ConversationStateEnum.RESPONDING)
    
    def set_idle(self) -> None:
        """Set state to idle (waiting for user input)."""
        self.state.transit_to(ConversationStateEnum.IDLE)
    
    def get_current_state(self) -> str:
        """
        Get the current conversation state.
        
        Returns:
            The current state as a string
        """
        return self.state.state.value
