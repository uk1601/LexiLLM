"""
Conversation State Management for LexiLLM
Defines the conversation state machine and transitions
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from ..logger import get_logger

# Get logger
logger = get_logger()

class ConversationStateEnum(Enum):
    """Enumeration of possible conversation states."""
    INITIALIZING = "initializing"  # Initial state during bot setup
    ONBOARDING = "onboarding"  # Collecting core user information
    IDLE = "idle"  # Waiting for user input
    COLLECTING_INFO = "collecting_info"  # Collecting information for intent
    PROCESSING = "processing"  # Processing user query
    RESPONDING = "responding"  # Generating response
    AWAITING_CONFIRMATION = "awaiting_confirmation"  # Waiting for user to confirm
    ENDING = "ending"  # Conversation is ending

@dataclass
class PendingQuery:
    """
    Represents a query that's pending while other actions (like info collection) occur.
    """
    message: str  # The original user message
    intent: str  # The detected intent
    topic: str  # The topic/subject of the query
    info_needed: List[str] = field(default_factory=list)  # Information needed to answer
    confirmation_needed: bool = False  # Whether confirmation is needed
    confirm_message: Optional[str] = None  # Message to send for confirmation

@dataclass
class ConversationState:
    """
    Represents the current state of a conversation including context.
    """
    state: ConversationStateEnum = ConversationStateEnum.INITIALIZING
    pending_query: Optional[PendingQuery] = None
    current_topic: Optional[str] = None
    current_intent: Optional[str] = None
    collecting_attribute: Optional[str] = None
    is_active: bool = True
    confirmation_context: Optional[Dict[str, Any]] = None
    
    def transit_to(self, new_state: ConversationStateEnum) -> None:
        """
        Transition to a new conversation state.
        
        Args:
            new_state: The state to transition to
        """
        old_state = self.state
        self.state = new_state
        logger.info(f"Conversation state transition: {old_state.value} -> {new_state.value}")
    
    def save_pending_query(self, message: str, intent: str, topic: str) -> None:
        """
        Save a query to be processed after collecting information.
        
        Args:
            message: The user's message
            intent: The detected intent
            topic: The topic of the query
        """
        self.pending_query = PendingQuery(message=message, intent=intent, topic=topic)
        logger.debug(f"Saved pending query: {message[:50]}... with intent {intent}")
    
    def clear_pending_query(self) -> None:
        """Clear any pending query."""
        was_pending = self.pending_query is not None
        self.pending_query = None
        if was_pending:
            logger.debug("Cleared pending query")
    
    def start_info_collection(self, attribute: str) -> None:
        """
        Start collecting a specific attribute.
        
        Args:
            attribute: The attribute to collect
        """
        self.transit_to(ConversationStateEnum.COLLECTING_INFO)
        self.collecting_attribute = attribute
        logger.info(f"Started collecting attribute: {attribute}")
    
    def end_info_collection(self) -> None:
        """End the information collection process."""
        self.collecting_attribute = None
        logger.info("Ended information collection")
    
    def is_collecting_info(self) -> bool:
        """
        Check if currently collecting information.
        
        Returns:
            True if in information collection state
        """
        return self.state == ConversationStateEnum.COLLECTING_INFO
    
    def set_awaiting_confirmation(self, context: Dict[str, Any]) -> None:
        """
        Set the state to awaiting confirmation.
        
        Args:
            context: Context information about what needs confirmation
        """
        self.transit_to(ConversationStateEnum.AWAITING_CONFIRMATION)
        self.confirmation_context = context
        logger.debug(f"Awaiting confirmation: {context}")
    
    def clear_confirmation(self) -> None:
        """Clear the confirmation state."""
        self.confirmation_context = None
        logger.debug("Cleared confirmation state")
    
    def is_awaiting_confirmation(self) -> bool:
        """
        Check if awaiting user confirmation.
        
        Returns:
            True if awaiting confirmation
        """
        return self.state == ConversationStateEnum.AWAITING_CONFIRMATION
    
    def get_info_collection_attribute(self) -> Optional[str]:
        """
        Get the attribute currently being collected.
        
        Returns:
            The attribute name or None if not collecting
        """
        return self.collecting_attribute if self.is_collecting_info() else None

def is_confirmation(message: str) -> bool:
    """
    Check if a message is a confirmation (yes, continue, etc.)
    
    Args:
        message: The user's message
        
    Returns:
        True if the message appears to be a confirmation
    """
    message = message.lower().strip()
    confirmation_phrases = [
        "yes", "yeah", "yep", "sure", "ok", "okay", "continue", 
        "go ahead", "proceed", "that's right", "correct", "right",
        "do it", "sounds good", "please do", "absolutely"
    ]
    
    # Check for exact matches
    if message in confirmation_phrases:
        return True
    
    # Check for phrases starting with confirmation words
    for phrase in confirmation_phrases:
        if message.startswith(phrase):
            return True
    
    return False

def is_rejection(message: str) -> bool:
    """
    Check if a message is a rejection (no, stop, etc.)
    
    Args:
        message: The user's message
        
    Returns:
        True if the message appears to be a rejection
    """
    message = message.lower().strip()
    rejection_phrases = [
        "no", "nope", "nah", "stop", "don't", "do not", "cancel", 
        "skip", "nevermind", "forget it", "incorrect", "wrong"
    ]
    
    # Check for exact matches
    if message in rejection_phrases:
        return True
    
    # Check for phrases starting with rejection words
    for phrase in rejection_phrases:
        if message.startswith(phrase):
            return True
    
    return False
