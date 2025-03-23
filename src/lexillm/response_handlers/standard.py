"""
Standard Response Handler for LexiLLM
Handles regular (non-streaming) responses
"""

from typing import Any, Dict, Optional, Union

from .base import ResponseHandler
from ..modules.response_generator import ResponseGenerator
from ..logger import get_logger

# Get logger
logger = get_logger()

class StandardResponseHandler(ResponseHandler):
    """
    Handles standard (non-streaming) responses for LexiLLM.
    
    This handler returns string responses for all methods.
    """
    
    def __init__(self, response_generator: ResponseGenerator):
        """
        Initialize the standard response handler.
        
        Args:
            response_generator: The response generator to use
        """
        self.response_generator = response_generator
        logger.debug("StandardResponseHandler initialized")
    
    def handle_response(self, message: str, intent: str, chat_history: Any,
                       user_profile: Any, specific_topic: Optional[str] = None) -> str:
        """
        Handle generating a standard response for a user message.
        
        Args:
            message: The user's message
            intent: The detected intent
            chat_history: Conversation history
            user_profile: User profile information
            specific_topic: Optional specific topic to focus on
            
        Returns:
            A string response to the user
        """
        logger.debug(f"StandardResponseHandler generating response for intent {intent}")
        return self.response_generator.generate_response(
            message,
            intent,
            chat_history,
            user_profile,
            specific_topic
        )
    
    def handle_end_conversation(self, chat_history: Any, user_profile: Any) -> str:
        """
        Handle generating a standard conversation ending message.
        
        Args:
            chat_history: Conversation history
            user_profile: User profile information
            
        Returns:
            A string closing message
        """
        logger.debug("StandardResponseHandler generating end conversation message")
        return self.response_generator.end_conversation(
            chat_history,
            user_profile
        )
    
    def handle_info_collection_message(self, message: str) -> str:
        """
        Handle a standard information collection message.
        
        Args:
            message: The information collection message
            
        Returns:
            The information collection message unchanged
        """
        # Just return the message as is for standard responses
        return message
    
    def handle_response_for_pending_query(self, pending_query: Dict[str, Any], chat_history: Any,
                                         user_profile: Any) -> str:
        """
        Handle generating a standard response for a pending query.
        
        Args:
            pending_query: The pending query dictionary
            chat_history: Conversation history
            user_profile: User profile information
            
        Returns:
            A string response for the pending query
        """
        logger.debug(f"StandardResponseHandler generating response for pending query about {pending_query['topic']}")
        return self.response_generator.generate_response(
            pending_query["message"],
            pending_query["intent"],
            chat_history,
            user_profile,
            pending_query["topic"]
        )
    
    def handle_simple_message(self, message: str) -> str:
        """
        Handle a standard simple message.
        
        Args:
            message: The simple message
            
        Returns:
            The simple message unchanged
        """
        # Just return the message as is for standard responses
        return message
    
    def handle_error_message(self, name: Optional[str], error_message: str) -> str:
        """
        Handle generating a standard error message.
        
        Args:
            name: The user's name, if available
            error_message: Error message
            
        Returns:
            A string error message
        """
        # Format the error message with the user's name if available
        if name:
            return f"I apologize, {name}, but I encountered an error while processing your message. Let's try again with a different question."
        else:
            return "I apologize, but I encountered an error while processing your message. Let's try again with a different question."