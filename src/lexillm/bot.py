"""
Core LexiLLM bot implementation with streaming capabilities
"""

import os
from typing import Dict, Tuple, Optional, Any, Generator, Iterator, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .schemas import IntentClassifier, UserInfo
from .templates import create_templates, create_intent_prompt, create_extraction_prompt
from .utils import is_end_request, welcome_message
from .user_profile import UserProfile, UserProfileManager
from .modules.intent_manager import IntentManager
from .modules.response_generator import ResponseGenerator
from .modules.conversation_manager import ConversationManager
from .modules.info_collector import InfoCollector
from .modules.conversation_state import ConversationStateEnum, is_confirmation, is_rejection
from .config import LLM_CONFIG, get_api_key
from .logger import get_logger
from .exceptions import (
    LexiLLMError, 
    APIKeyError, 
    ModelError, 
    IntentClassificationError,
    ResponseGenerationError, 
    ConversationError, 
    InfoCollectionError
)

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
            
            # Keep track of the current intent
            self.current_intent = None
            
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
        try:
            # Log the incoming message
            logger.info(f"Processing message: {message[:50]}...")
            
            # Track this interaction
            self.user_profile.track_interaction(self.conversation_manager.get_topic())
            
            # First, check if this is an end request
            if is_end_request(message):
                logger.info("End request detected")
                # Save the profile before ending
                self.profile_manager.save_profile(self.user_profile)
                # If the user wants to end, call end_conversation and return the closing message
                self.conversation_manager.end_conversation()
                return self.response_generator.end_conversation(
                    self.conversation_manager.chat_history, 
                    self.user_profile
                )
            
            # Add the user's message to history
            self.conversation_manager.add_user_message(message)
            
            # Get the current conversation state
            current_state = self.conversation_manager.get_current_state()
            logger.debug(f"Current conversation state: {current_state}")
            
            # Check if we're awaiting confirmation
            if self.conversation_manager.is_awaiting_confirmation():
                # Process the confirmation response
                logger.debug("Processing potential confirmation response")
                is_processed, is_confirmed = self.conversation_manager.process_confirmation_response(message)
                
                # If it was indeed a confirmation response
                if is_processed:
                    if is_confirmed:
                        logger.info("User confirmed, continuing with pending action")
                        # Get pending query if available
                        pending_query = self.conversation_manager.get_pending_query()
                        if pending_query:
                            # Resume processing of the pending query
                            self.conversation_manager.set_processing()
                            
                            # Generate response for the pending query
                            response = self.response_generator.generate_response(
                                pending_query["message"],
                                pending_query["intent"],
                                self.conversation_manager.chat_history,
                                self.user_profile,
                                pending_query["topic"]
                            )
                            
                            # Add response to history and clear pending query
                            self.conversation_manager.add_ai_message(response)
                            self.conversation_manager.clear_pending_query()
                            self.conversation_manager.set_idle()
                            
                            return response
                        else:
                            # No pending query, just acknowledge the confirmation
                            self.conversation_manager.set_idle()
                            return "Great! What would you like to know about LLMs today?"
                    else:
                        # User rejected, cancel the pending action
                        logger.info("User rejected, canceling pending action")
                        self.conversation_manager.clear_pending_query()
                        self.conversation_manager.set_idle()
                        return "No problem. What would you like to know about LLMs instead?"
                
                # If it wasn't a confirmation response, treat it as a new query
                # (continue with normal processing below)
                logger.info("Message not recognized as confirmation, treating as new query")
                self.conversation_manager.clear_confirmation()
                
            # Check if we're collecting information
            if self.info_collector.is_collecting_info():
                logger.debug("Currently collecting information")
                # Process the response to information collection
                attr = self.info_collector.get_current_collection_attribute()
                response = self.info_collector.process_explicit_info_collection(message)
                
                if response is not None:
                    # If there's a response, add it to history and return it
                    self.conversation_manager.add_ai_message(response)
                    return response
                
                # If response is None, collection is complete - check for pending query
                pending_query = self.conversation_manager.get_pending_query()
                if pending_query:
                    logger.info("Information collection complete, resuming pending query")
                    # Resume the pending query
                    self.conversation_manager.set_processing()
                    
                    # Generate a resume message
                    resume_msg = f"Thanks for providing that information. Now, let me answer your question about {pending_query['topic']}."
                    
                    # Generate the actual response
                    response = self.response_generator.generate_response(
                        pending_query["message"],
                        pending_query["intent"],
                        self.conversation_manager.chat_history,
                        self.user_profile,
                        pending_query["topic"]
                    )
                    
                    # Add complete response to history
                    self.conversation_manager.add_ai_message(response)
                    self.conversation_manager.clear_pending_query()
                    self.conversation_manager.set_idle()
                    
                    return response
                else:
                    # No pending query
                    if self.conversation_manager.get_current_state() == ConversationStateEnum.ONBOARDING.value:
                        # If we were in onboarding, it's now complete
                        logger.info("Onboarding complete")
                        self.conversation_manager.complete_onboarding()
                        return "What would you like to know about LLMs today?"
                    else:
                        # Otherwise, we've just collected a single attribute
                        logger.info("Single attribute collection complete")
                        self.conversation_manager.set_idle()
                        return "Thanks for providing that information. What would you like to know about LLMs?"
            
            # If we're here, we're processing a new query or follow-up
            
            # Always try to extract implicit user information from the message
            logger.debug("Extracting implicit user information")
            self.info_collector.extract_user_info_from_message(message)
            
            # Check if this is a follow-up to the previous topic
            is_followup = self.intent_manager.is_followup_question(
                message, 
                self.conversation_manager.chat_history
            )
            
            # Determine intent
            if not is_followup or not self.current_intent:
                logger.debug("Classifying new intent")
                # Classify the intent
                intent_result = self.intent_manager.classify_intent(message)
                intent = intent_result.get("intent", "UNKNOWN")
                self.current_intent = intent
                # Store this as the last topic and intent
                self.conversation_manager.set_topic(message.strip().lower())
                self.conversation_manager.set_intent(intent)
            else:
                logger.debug(f"Using existing intent for follow-up: {self.current_intent}")
                # For follow-up, if the message is very short (like just "embeddings"), use it as the topic
                if len(message.split()) <= 2:
                    self.conversation_manager.set_topic(message.strip().lower())
                # Use the current intent for follow-up questions
                intent = self.current_intent
            
            # Check if this could be a confirmation for the current conversation context
            if is_confirmation(message) and len(message.split()) <= 3:
                # If it looks like a confirmation but we're not specifically awaiting one,
                # we'll still proceed with generating a response as a new query
                logger.debug("Message looks like confirmation but not awaiting one, proceeding as new query")
            
            # Check if we need more information for this intent
            needs_more_info, info_type_needed = self.info_collector.determine_if_more_info_needed(intent)
            
            if needs_more_info:
                logger.info(f"Need more information for intent {intent}: {info_type_needed}")
                # Save the current query to resume after collecting info
                self.conversation_manager.save_pending_query(message, intent, self.conversation_manager.get_topic())
                
                # Start collecting the needed information
                self.info_collector.start_info_collection(info_type_needed)
                info_msg = self.info_collector.get_info_collection_message(info_type_needed)
                
                # Add to history and return
                self.conversation_manager.add_ai_message(info_msg)
                return info_msg
            
            # Check if we have an opportunity to collect more user information
            should_collect, attr_to_collect = self.info_collector.check_for_info_collection_opportunity()
            
            if should_collect and attr_to_collect:
                logger.info(f"Opportunity to collect information: {attr_to_collect}")
                # Save the current query as pending
                self.conversation_manager.save_pending_query(message, intent, self.conversation_manager.get_topic())
                
                # Start collecting the information
                self.info_collector.start_info_collection(attr_to_collect)
                info_msg = self.info_collector.get_info_collection_message(attr_to_collect)
                
                # Add to history and return
                self.conversation_manager.add_ai_message(info_msg)
                return info_msg
            
            # Set state to processing and responding
            self.conversation_manager.set_processing()
            
            # Manage chat history to avoid growing too large
            self.conversation_manager.manage_chat_history()
            
            # Generate the response
            logger.debug(f"Generating response for intent: {intent}")
            response = self.response_generator.generate_response(
                message, 
                intent, 
                self.conversation_manager.chat_history,
                self.user_profile,
                self.conversation_manager.get_topic()
            )
            
            # Set state to idle
            self.conversation_manager.set_idle()
            
            # Add response to history
            self.conversation_manager.add_ai_message(response)
            
            return response
        except Exception as e:
            # If an error occurs, provide a graceful fallback response
            error_msg = f"Error in process_message: {str(e)}"
            logger.error(error_msg)
            
            # Reset state to idle
            if hasattr(self, 'conversation_manager'):
                self.conversation_manager.set_idle()
            
            # Personalize the fallback message if we know the user's name
            name = self.user_profile.get_attribute_value("name")
            if name:
                return f"I apologize, {name}, but I encountered an error while processing your message. Let's try again with a different question."
            else:
                return "I apologize, but I encountered an error while processing your message. Let's try again with a different question."
    
    def process_message_streaming(self, message: str) -> Iterator[str]:
        """
        Process a user message and return a streaming appropriate response.
        
        This is the main method for interacting with the bot with streaming responses.
        
        Args:
            message: The user's message
            
        Returns:
            An iterator of response chunks
        """
        try:
            # Log the incoming message
            logger.info(f"Processing streaming message: {message[:50]}...")
            
            # Track this interaction
            self.user_profile.track_interaction(self.conversation_manager.get_topic())
            
            # First, check if this is an end request
            if is_end_request(message):
                logger.info("End request detected for streaming response")
                # Save the profile before ending
                self.profile_manager.save_profile(self.user_profile)
                # If user wants to end, stream the closing message
                self.conversation_manager.end_conversation()
                
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
                
            # Add the user's message to history
            self.conversation_manager.add_user_message(message)
            
            # Get the current conversation state
            current_state = self.conversation_manager.get_current_state()
            logger.debug(f"Current conversation state for streaming: {current_state}")
            
            # Check if we're awaiting confirmation
            if self.conversation_manager.is_awaiting_confirmation():
                # Process the confirmation response
                logger.debug("Processing potential confirmation response for streaming")
                is_processed, is_confirmed = self.conversation_manager.process_confirmation_response(message)
                
                # If it was indeed a confirmation response
                if is_processed:
                    if is_confirmed:
                        logger.info("User confirmed, continuing with pending action")
                        # Get pending query if available
                        pending_query = self.conversation_manager.get_pending_query()
                        if pending_query:
                            # Resume processing of the pending query
                            self.conversation_manager.set_processing()
                            
                            # Generate streaming response for the pending query
                            complete_response = ""
                            for chunk in self.response_generator.generate_response_streaming(
                                pending_query["message"],
                                pending_query["intent"],
                                self.conversation_manager.chat_history,
                                self.user_profile,
                                pending_query["topic"]
                            ):
                                complete_response += chunk
                                yield chunk
                            
                            # Add response to history and clear pending query
                            self.conversation_manager.add_ai_message(complete_response)
                            self.conversation_manager.clear_pending_query()
                            self.conversation_manager.set_idle()
                            return
                        else:
                            # No pending query, just acknowledge the confirmation
                            self.conversation_manager.set_idle()
                            response = "Great! What would you like to know about LLMs today?"
                            self.conversation_manager.add_ai_message(response)
                            yield response
                            return
                    else:
                        # User rejected, cancel the pending action
                        logger.info("User rejected, canceling pending action")
                        self.conversation_manager.clear_pending_query()
                        self.conversation_manager.set_idle()
                        response = "No problem. What would you like to know about LLMs instead?"
                        self.conversation_manager.add_ai_message(response)
                        yield response
                        return
                
                # If it wasn't a confirmation response, treat it as a new query
                # (continue with normal processing below)
                logger.info("Message not recognized as confirmation, treating as new query for streaming")
                self.conversation_manager.clear_confirmation()
            
            # Check if we're collecting information
            if self.info_collector.is_collecting_info():
                logger.debug("Currently collecting information for streaming response")
                # Process the response to information collection
                attr = self.info_collector.get_current_collection_attribute()
                response = self.info_collector.process_explicit_info_collection(message)
                
                if response is not None:
                    # If there's a response, add it to history and return it
                    self.conversation_manager.add_ai_message(response)
                    yield response
                    return
                
                # If response is None, collection is complete - check for pending query
                pending_query = self.conversation_manager.get_pending_query()
                if pending_query:
                    logger.info("Information collection complete, resuming pending query for streaming")
                    # Resume the pending query
                    self.conversation_manager.set_processing()
                    
                    # Generate the streaming response
                    complete_response = ""
                    for chunk in self.response_generator.generate_response_streaming(
                        pending_query["message"],
                        pending_query["intent"],
                        self.conversation_manager.chat_history,
                        self.user_profile,
                        pending_query["topic"]
                    ):
                        complete_response += chunk
                        yield chunk
                    
                    # Add complete response to history
                    self.conversation_manager.add_ai_message(complete_response)
                    self.conversation_manager.clear_pending_query()
                    self.conversation_manager.set_idle()
                    return
                else:
                    # No pending query
                    if self.conversation_manager.get_current_state() == ConversationStateEnum.ONBOARDING.value:
                        # If we were in onboarding, it's now complete
                        logger.info("Onboarding complete for streaming")
                        self.conversation_manager.complete_onboarding()
                        response = "What would you like to know about LLMs today?"
                        self.conversation_manager.add_ai_message(response)
                        yield response
                        return
                    else:
                        # Otherwise, we've just collected a single attribute
                        logger.info("Single attribute collection complete for streaming")
                        self.conversation_manager.set_idle()
                        response = "Thanks for providing that information. What would you like to know about LLMs?"
                        self.conversation_manager.add_ai_message(response)
                        yield response
                        return
            
            # If we're here, we're processing a new query or follow-up
            
            # Always try to extract implicit user information from the message
            logger.debug("Extracting implicit user information for streaming response")
            self.info_collector.extract_user_info_from_message(message)
            
            # Check if this is a follow-up to the previous topic
            is_followup = self.intent_manager.is_followup_question(
                message, 
                self.conversation_manager.chat_history
            )
            
            # Determine intent
            if not is_followup or not self.current_intent:
                logger.debug("Classifying new intent for streaming response")
                # Classify the intent
                intent_result = self.intent_manager.classify_intent(message)
                intent = intent_result.get("intent", "UNKNOWN")
                self.current_intent = intent
                # Store this as the last topic and intent
                self.conversation_manager.set_topic(message.strip().lower())
                self.conversation_manager.set_intent(intent)
            else:
                logger.debug(f"Using existing intent for streaming follow-up: {self.current_intent}")
                # For follow-up, if the message is very short (like just "embeddings"), use it as the topic
                if len(message.split()) <= 2:
                    self.conversation_manager.set_topic(message.strip().lower())
                # Use the current intent for follow-up questions
                intent = self.current_intent
            
            # Check if this could be a confirmation for the current conversation context
            if is_confirmation(message) and len(message.split()) <= 3:
                # If it looks like a confirmation but we're not specifically awaiting one,
                # we'll still proceed with generating a response as a new query
                logger.debug("Message looks like confirmation but not awaiting one, proceeding as new query for streaming")
            
            # Check if we need more information for this intent
            needs_more_info, info_type_needed = self.info_collector.determine_if_more_info_needed(intent)
            
            if needs_more_info:
                logger.info(f"Need more information for streaming intent {intent}: {info_type_needed}")
                # Save the current query to resume after collecting info
                self.conversation_manager.save_pending_query(message, intent, self.conversation_manager.get_topic())
                
                # Start collecting the needed information
                self.info_collector.start_info_collection(info_type_needed)
                info_msg = self.info_collector.get_info_collection_message(info_type_needed)
                
                # Add to history and yield
                self.conversation_manager.add_ai_message(info_msg)
                yield info_msg
                return
            
            # Check if we have an opportunity to collect more user information
            should_collect, attr_to_collect = self.info_collector.check_for_info_collection_opportunity()
            
            if should_collect and attr_to_collect:
                logger.info(f"Opportunity to collect information for streaming: {attr_to_collect}")
                # Save the current query as pending
                self.conversation_manager.save_pending_query(message, intent, self.conversation_manager.get_topic())
                
                # Start collecting the information
                self.info_collector.start_info_collection(attr_to_collect)
                info_msg = self.info_collector.get_info_collection_message(attr_to_collect)
                
                # Add to history and yield
                self.conversation_manager.add_ai_message(info_msg)
                yield info_msg
                return
            
            # Set state to processing and responding
            self.conversation_manager.set_processing()
            
            # Manage chat history to avoid growing too large
            self.conversation_manager.manage_chat_history()
            
            # Generate the streaming response
            logger.debug(f"Generating streaming response for intent: {intent}")
            complete_response = ""
            for chunk in self.response_generator.generate_response_streaming(
                message,
                intent,
                self.conversation_manager.chat_history,
                self.user_profile,
                self.conversation_manager.get_topic()
            ):
                complete_response += chunk
                yield chunk
            
            # Add complete response to history
            if complete_response:
                self.conversation_manager.add_ai_message(complete_response)
            
            # Set state to idle
            self.conversation_manager.set_idle()
            
        except Exception as e:
            # If an error occurs, yield a fallback message
            error_msg = f"Error in process_message_streaming: {str(e)}"
            logger.error(error_msg)
            
            # Reset state to idle
            if hasattr(self, 'conversation_manager'):
                self.conversation_manager.set_idle()
            
            # Personalize the fallback message if we know the user's name
            name = self.user_profile.get_attribute_value("name")
            if name:
                fallback_msg = f"I apologize, {name}, but I encountered an error while processing your message. Let's try again with a different question."
            else:
                fallback_msg = "I apologize, but I encountered an error while processing your message. Let's try again with a different question."
                
            yield fallback_msg
            # Also add to chat history
            self.conversation_manager.add_ai_message(fallback_msg)
    
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
