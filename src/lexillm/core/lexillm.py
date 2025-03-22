"""
Main LexiLLM class implementation
"""

import os
from typing import Dict, Tuple, Optional, Any, Iterator, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from ..schemas import IntentClassifier, UserInfo
from ..user_profile import UserProfile, UserProfileManager
from ..config import LLM_CONFIG, get_api_key
from ..logger import get_logger
from ..exceptions import (
    LexiLLMError, 
    APIKeyError, 
    ModelError, 
    IntentClassificationError,
    ResponseGenerationError, 
    ConversationError, 
    InfoCollectionError
)
from ..modules.intent_manager import IntentManager
from ..modules.response_generator import ResponseGenerator
from ..modules.conversation_manager import ConversationManager
from ..modules.info_collector import InfoCollector
from ..modules.conversation_state import ConversationStateEnum

from .message_processor import MessageProcessor
from .streaming_processor import StreamingProcessor
from .utils import welcome_message, is_end_request

# Get logger
logger = get_logger()

class LexiLLM:
    """
    LexiLLM - A specialized chatbot for LLM-related topics with streaming support.
    
    This bot can discuss LLM fundamentals, implementation guidance,
    model comparisons, and recent trends in the field using modern
    LangChain practices. It supports both standard and streaming responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, 
                 streaming: bool = True, session_id: str = "default"):
        """
        Initialize the LexiLLM bot with optional streaming support.
        
        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model_name: Name of the LLM model to use (defaults to configuration)
            streaming: Whether to enable streaming responses
            session_id: Unique identifier for this user session
        """
        try:
            # Load environment variables from .env file
            load_dotenv()
            
            # Use provided API key or get from environment
            self.api_key = api_key or get_api_key()
            if not self.api_key:
                error_msg = "OpenAI API key is required. Please set it in the .env file or pass it as a parameter."
                logger.error(error_msg)
                raise APIKeyError(error_msg)
            
            # Use provided model name or get from configuration
            self.model_name = model_name or LLM_CONFIG["default_model"]
            logger.info(f"Initializing LexiLLM with model: {self.model_name}")
            
            # Initialize the LLM with appropriate settings
            self.llm = ChatOpenAI(
                temperature=LLM_CONFIG["main_temperature"],
                model=self.model_name,
                api_key=self.api_key,
                streaming=streaming,
                timeout=LLM_CONFIG["main_timeout"],
                max_retries=LLM_CONFIG["max_retries"]
            )
            
            # Initialize a more robust LLM for classification with lower temperature
            self.classifier_llm = ChatOpenAI(
                temperature=LLM_CONFIG["classifier_temperature"],
                model=self.model_name,
                api_key=self.api_key,
                timeout=LLM_CONFIG["classifier_timeout"],
                max_retries=LLM_CONFIG["max_retries"]
            )
            
            # Initialize user profile management
            self.session_id = session_id
            self.profile_manager = UserProfileManager()
            self.user_profile = self.profile_manager.get_profile(session_id)
            
            # Initialize the sub-modules in the correct order
            # First create conversation manager to track state
            self.conversation_manager = ConversationManager()
            # Next instantiate other components
            self.intent_manager = IntentManager(self.classifier_llm)
            self.response_generator = ResponseGenerator(self.llm)
            # Info collector needs conversation manager for state management
            self.info_collector = InfoCollector(
                self.profile_manager, 
                self.user_profile,
                self.conversation_manager
            )
            
            # Create processor components
            self.message_processor = MessageProcessor(
                self.conversation_manager,
                self.intent_manager,
                self.response_generator,
                self.info_collector,
                self.profile_manager,
                self.user_profile
            )
            
            self.streaming_processor = StreamingProcessor(
                self.conversation_manager,
                self.intent_manager,
                self.response_generator,
                self.info_collector,
                self.profile_manager,
                self.user_profile
            )
            
            # Keep track of the current intent
            self.current_intent = None
            
            # Track if the conversation is active (for main loop control)
            self.conversation_active = True
            
            logger.info(f"LexiLLM initialized successfully with session ID: {session_id}")
        except Exception as e:
            logger.critical(f"Failed to initialize LexiLLM: {str(e)}")
            if isinstance(e, LexiLLMError):
                raise
            raise LexiLLMError(f"Failed to initialize LexiLLM: {str(e)}")
    
    def welcome_message(self) -> str:
        """
        Return the welcome message to start the conversation.
        
        For new users, this begins the onboarding process to collect
        basic user information.
        """
        try:
            # Track the interaction
            self.user_profile.track_interaction()
            
            # If we haven't completed onboarding, start that process
            if not self.user_profile.onboarding_completed:
                logger.info("Starting onboarding process")
                self.info_collector.start_info_collection("name")
                return self.profile_manager.generate_collection_message("name")
            
            # Otherwise return the standard welcome
            logger.info("Returning standard welcome message")
            return welcome_message()
        except Exception as e:
            logger.error(f"Error generating welcome message: {str(e)}")
            return "Welcome to LexiLLM! I'm your specialized assistant for Large Language Models. How can I help you today?"
    
    def process_message(self, message: str) -> str:
        """
        Process a user message and return an appropriate response.
        
        This is the main method for interacting with the bot, with enhanced
        user information collection throughout the conversation.
        
        Args:
            message: The user's message
            
        Returns:
            The bot's response
        """
        # Check for end request first
        if is_end_request(message):
            logger.info("End request detected")
            # Save the profile before ending
            self.profile_manager.save_profile(self.user_profile)
            # End the conversation
            self.conversation_manager.end_conversation()
            # Mark conversation as inactive
            self.conversation_active = False
            return self.response_generator.end_conversation(
                self.conversation_manager.chat_history, 
                self.user_profile
            )
        
        # Delegate to message processor
        self.message_processor.current_intent = self.current_intent
        response, self.current_intent = self.message_processor.process(message)
        return response
    
    def process_message_streaming(self, message: str) -> Iterator[str]:
        """
        Process a user message and return a streaming appropriate response.
        
        This is the main method for interacting with the bot with streaming responses.
        
        Args:
            message: The user's message
            
        Returns:
            An iterator of response chunks
        """
        # Check for end request first
        if is_end_request(message):
            logger.info("End request detected for streaming response")
            # Save the profile before ending
            self.profile_manager.save_profile(self.user_profile)
            # End the conversation
            self.conversation_manager.end_conversation()
            # Mark conversation as inactive
            self.conversation_active = False
            
            # Generate the end conversation message and update history
            logger.debug("Generating end conversation streaming response")
            complete_response = ""
            for chunk in self.response_generator.end_conversation_streaming(
                self.conversation_manager.chat_history,
                self.user_profile
            ):
                complete_response += chunk
                yield chunk
            
            # Add the complete response to chat history
            self.conversation_manager.add_ai_message(complete_response)
            return
        
        # Delegate to streaming processor
        self.streaming_processor.current_intent = self.current_intent
        
        # Process the message and yield chunks
        for chunk in self.streaming_processor.process(message):
            yield chunk
        
        # Update current intent after processing
        self.current_intent = self.streaming_processor.current_intent
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Expose intent classification functionality for external use.
        
        Args:
            query: The user's message
            
        Returns:
            Dictionary with intent and confidence
        """
        try:
            return self.intent_manager.classify_intent(query)
        except Exception as e:
            logger.error(f"Error exposing intent classification: {str(e)}")
            return {"intent": "UNKNOWN", "confidence": 0.0}
            
    def get_user_profile(self) -> UserProfile:
        """
        Get the current user profile.
        
        Returns:
            The current user profile
        """
        return self.user_profile
        
    def reset_conversation(self) -> None:
        """
        Reset the conversation state.
        
        This clears the conversation history and resets the intent tracking.
        """
        try:
            logger.info("Resetting conversation")
            self.conversation_manager.clear_history()
            self.current_intent = None
            # Don't reset the user profile, just the conversation state
        except Exception as e:
            logger.error(f"Error resetting conversation: {str(e)}")
            
    def save_user_profile(self) -> None:
        """
        Explicitly save the user profile.
        
        This is useful when the application is closing or when you want
        to ensure the profile is persisted.
        """
        try:
            logger.info(f"Explicitly saving user profile for session {self.session_id}")
            self.profile_manager.save_profile(self.user_profile)
        except Exception as e:
            logger.error(f"Error saving user profile: {str(e)}")
    
    def is_conversation_active(self) -> bool:
        """
        Check if the conversation is still active.
        
        Returns:
            Boolean indicating if the conversation is active
        """
        return self.conversation_active
    
    def set_conversation_active(self, active: bool) -> None:
        """
        Set the conversation active state.
        
        Args:
            active: Boolean indicating if the conversation is active
        """
        logger.debug(f"Setting conversation active: {active}")
        self.conversation_active = active
