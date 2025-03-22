"""
Streaming processor for LexiLLM
Handles streaming message processing
"""

from typing import Dict, Tuple, Optional, Any, Iterator, List

from ..logger import get_logger
from ..modules.conversation_state import ConversationStateEnum, is_confirmation, is_rejection

# Get logger
logger = get_logger()

class StreamingProcessor:
    """
    Process user messages and generate streaming responses.
    
    This class handles the core message processing logic for streaming responses.
    """
    
    def __init__(self, conversation_manager, intent_manager, response_generator, 
                info_collector, profile_manager, user_profile):
        """
        Initialize the streaming processor.
        
        Args:
            conversation_manager: Manages conversation state and history
            intent_manager: Handles intent classification
            response_generator: Generates responses based on intent
            info_collector: Collects user information
            profile_manager: Manages user profiles
            user_profile: The current user's profile
        """
        self.conversation_manager = conversation_manager
        self.intent_manager = intent_manager
        self.response_generator = response_generator
        self.info_collector = info_collector
        self.profile_manager = profile_manager
        self.user_profile = user_profile
        self.current_intent = None
    
    def process(self, message: str) -> Iterator[str]:
        """
        Process a user message and yield streaming response chunks.
        
        Args:
            message: The user's message
            
        Returns:
            Iterator of response chunks
        """
        try:
            # Log the incoming message
            logger.info(f"Processing streaming message: {message[:50]}...")
            
            # Track this interaction
            self.user_profile.track_interaction(self.conversation_manager.get_topic())
            
            # Add the user's message to history
            self.conversation_manager.add_user_message(message)
            
            # Get the current conversation state
            current_state = self.conversation_manager.get_current_state()
            logger.debug(f"Current conversation state for streaming: {current_state}")
            
            # Process based on conversation state
            if self.conversation_manager.is_awaiting_confirmation():
                # Handle confirmation state
                for chunk in self._handle_confirmation_state(message):
                    yield chunk
                return
                
            if self.info_collector.is_collecting_info():
                # Handle info collection state
                for chunk in self._handle_info_collection_state(message):
                    yield chunk
                return
            
            # Process new query or follow-up
            for chunk in self._process_query(message):
                yield chunk
                
        except Exception as e:
            # If an error occurs, yield a fallback message
            error_msg = f"Error in process_message_streaming: {str(e)}"
            logger.error(error_msg)
            
            # Reset state to idle
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
    
    def _handle_confirmation_state(self, message: str) -> Iterator[str]:
        """
        Handle messages when in the confirmation state.
        
        Args:
            message: The user's message
            
        Yields:
            Response chunks
        """
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
        logger.info("Message not recognized as confirmation, treating as new query for streaming")
        self.conversation_manager.clear_confirmation()
        
        # Continue with normal processing (handled by the caller)
        return
    
    def _handle_info_collection_state(self, message: str) -> Iterator[str]:
        """
        Handle messages when in the information collection state.
        
        Args:
            message: The user's message
            
        Yields:
            Response chunks
        """
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
            
            # Check if the user's message is a simple confirmation
            is_simple_confirmation = is_confirmation(message) and len(message.split()) <= 3
            
            # Use original topic from pending query, not the confirmation message
            topic = pending_query["topic"]
            logger.debug(f"Streaming response for original topic: {topic}")
            
            # Generate the streaming response using the original message
            complete_response = ""
            for chunk in self.response_generator.generate_response_streaming(
                pending_query["message"],  # Use the original query, not the confirmation
                pending_query["intent"],
                self.conversation_manager.chat_history,
                self.user_profile,
                pending_query["topic"]  # Use the original topic
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
    
    def _process_query(self, message: str) -> Iterator[str]:
        """
        Process a new query or follow-up question with streaming responses.
        
        Args:
            message: The user's message
            
        Yields:
            Response chunks
        """
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
