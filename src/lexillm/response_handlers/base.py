"""
Base Response Handler for LexiLLM
Defines the interface for different response strategies
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Iterator

from ..logger import get_logger

# Get logger
logger = get_logger()

class ResponseHandler(ABC):
    """
    Abstract base class for response handlers.
    
    Implements the Strategy pattern for handling different response types
    (like standard responses vs streaming responses).
    """
    
    @abstractmethod
    def handle_response(self, message: str, intent: str, chat_history: Any,
                        user_profile: Any, specific_topic: Optional[str] = None) -> Any:
        """
        Handle generating a response for a user message.
        
        Args:
            message: The user's message
            intent: The detected intent
            chat_history: Conversation history
            user_profile: User profile information
            specific_topic: Optional specific topic to focus on
            
        Returns:
            Response (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def handle_end_conversation(self, chat_history: Any, user_profile: Any) -> Any:
        """
        Handle generating a conversation ending message.
        
        Args:
            chat_history: Conversation history
            user_profile: User profile information
            
        Returns:
            Closing message (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def handle_info_collection_message(self, message: str) -> Any:
        """
        Handle generating a message for information collection.
        
        Args:
            message: The information collection message
            
        Returns:
            Information collection message (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def handle_response_for_pending_query(self, pending_query: Dict[str, Any], chat_history: Any,
                                          user_profile: Any) -> Any:
        """
        Handle generating a response for a pending query.
        
        Args:
            pending_query: The pending query dictionary
            chat_history: Conversation history
            user_profile: User profile information
            
        Returns:
            Response for pending query (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def handle_simple_message(self, message: str) -> Any:
        """
        Handle generating a simple message response.
        
        Args:
            message: The simple message
            
        Returns:
            Simple message (format depends on implementation)
        """
        pass
    
    @abstractmethod
    def handle_error_message(self, name: Optional[str], error_message: str) -> Any:
        """
        Handle generating an error message.
        
        Args:
            name: The user's name, if available
            error_message: Error message
            
        Returns:
            Error message (format depends on implementation)
        """
        pass