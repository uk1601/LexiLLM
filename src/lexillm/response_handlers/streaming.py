"""
Streaming Response Handler for LexiLLM
Handles streaming responses with iterators
"""

from typing import Any, Dict, Optional, Union, Iterator

from .base import ResponseHandler
from ..modules.response_generator import ResponseGenerator
from ..logger import get_logger

# Get logger
logger = get_logger()

class StreamingResponseHandler(ResponseHandler):
    """
    Handles streaming responses for LexiLLM.
    
    This handler returns iterator responses for all methods.
    """
    
    def __init__(self, response_generator: ResponseGenerator):
        """
        Initialize the streaming response handler.
        
        Args:
            response_generator: The response generator to use
        """
        self.response_generator = response_generator
        logger.debug("StreamingResponseHandler initialized")
    
    def handle_response(self, message: str, intent: str, chat_history: Any,
                       user_profile: Any, specific_topic: Optional[str] = None) -> Iterator[str]:
        """
        Handle generating a streaming response for a user message.
        
        Args:
            message: The user's message
            intent: The detected intent
            chat_history: Conversation history
            user_profile: User profile information
            specific_topic: Optional specific topic to focus on
            
        Returns:
            An iterator of response chunks
        """
        logger.debug(f"StreamingResponseHandler generating response for intent {intent}")
        return self.response_generator.generate_response_streaming(
            message,
            intent,
            chat_history,
            user_profile,
            specific_topic
        )
    
    def handle_end_conversation(self, chat_history: Any, user_profile: Any) -> Iterator[str]:
        """
        Handle generating a streaming conversation ending message.
        
        Args:
            chat_history: Conversation history
            user_profile: User profile information
            
        Returns:
            An iterator of response chunks for the closing message
        """
        logger.debug("StreamingResponseHandler generating end conversation message")
        return self.response_generator.end_conversation_streaming(
            chat_history,
            user_profile
        )
    
    def handle_info_collection_message(self, message: str) -> Iterator[str]:
        """
        Handle a streaming information collection message.
        
        Args:
            message: The information collection message
            
        Returns:
            An iterator with a single message chunk
        """
        # For streaming, we need to return an iterator
        # even for simple messages
        yield message
    
    def handle_response_for_pending_query(self, pending_query: Dict[str, Any], chat_history: Any,
                                         user_profile: Any) -> Iterator[str]:
        """
        Handle generating a streaming response for a pending query.
        
        Args:
            pending_query: The pending query dictionary
            chat_history: Conversation history
            user_profile: User profile information
            
        Returns:
            An iterator of response chunks for the pending query
        """
        logger.debug(f"StreamingResponseHandler generating response for pending query about {pending_query['topic']}")
        return self.response_generator.generate_response_streaming(
            pending_query["message"],
            pending_query["intent"],
            chat_history,
            user_profile,
            pending_query["topic"]
        )
    
    def handle_simple_message(self, message: str) -> Iterator[str]:
        """
        Handle a streaming simple message.
        
        Args:
            message: The simple message
            
        Returns:
            An iterator with a single message chunk
        """
        # For streaming, we need to return an iterator
        # even for simple messages
        yield message
    
    def handle_error_message(self, name: Optional[str], error_message: str) -> Iterator[str]:
        """
        Handle generating a streaming error message.
        
        Args:
            name: The user's name, if available
            error_message: Error message
            
        Returns:
            An iterator with a single error message chunk
        """
        # Format the error message with the user's name if available
        if name:
            yield f"I apologize, {name}, but I encountered an error while processing your message. Let's try again with a different question."
        else:
            yield "I apologize, but I encountered an error while processing your message. Let's try again with a different question."