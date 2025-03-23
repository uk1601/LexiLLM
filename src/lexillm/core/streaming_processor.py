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
        
        # Define patterns for direct questions and follow-ups
        self.direct_question_indicators = [
            "what is", "how do", "how does", "can you explain", "tell me about",
            "what are", "how can", "why is", "when should", "where can",
            "which", "who", "whose", "whom", "what about", "how about"
        ]
        
        self.followup_patterns = [
            "what about", "how about", "tell me more about", "elaborate on",
            "can you expand", "more details", "latest", "recent", "newest",
            "advancements", "developments", "progress", "updates", "improvements",
            "new", "current", "modern", "trending", "popular", "state of the art"
        ]
    
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
            
            # Check if this is a follow-up question about advancements or latest developments
            is_followup, enriched_message = self._handle_followup_question(message)
            if is_followup:
                logger.info(f"Detected follow-up question about advancements/latest: {message}")
                # Process the enriched message directly
                message = enriched_message
                # Generate streaming response with priority
                for chunk in self._process_query(message, prioritize_response=True):
                    yield chunk
                return
            
            # Check if this is a direct question that should be prioritized
            if self._is_direct_question(message):
                logger.info(f"Detected direct question for streaming: {message}")
                # Process the query with priority over information collection
                for chunk in self._process_query(message, prioritize_response=True):
                    yield chunk
                return
            
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
            
            # Generate a fallback message
            try:
                # Try to use the streaming fallback generator first
                for chunk in self.response_generator.generate_fallback_message_streaming(
                    message, 
                    self.conversation_manager.chat_history
                ):
                    yield chunk
                return
            except Exception as fallback_error:
                logger.error(f"Error generating streaming fallback: {str(fallback_error)}")
                
                # Personalize the fallback message if we know the user's name
                name = self.user_profile.get_attribute_value("name")
                if name:
                    fallback_msg = (f"I apologize, {name}, but I encountered an error while processing your message. "
                                 "As a specialized LLM assistant, I can help with questions about language models, "
                                 "embeddings, transformers, and other LLM topics. Would you like to ask about one of these areas?")
                else:
                    fallback_msg = ("I apologize, but I encountered an error while processing your message. "
                                "As a specialized LLM assistant, I can help with questions about language models, "
                                "embeddings, transformers, and other LLM topics. Would you like to ask about one of these areas?")
                    
                yield fallback_msg
                # Also add to chat history
                self.conversation_manager.add_ai_message(fallback_msg)
            
    def _is_direct_question(self, message: str) -> bool:
        """
        Check if the message is a direct question that should be prioritized.
        
        Args:
            message: The user's message
            
        Returns:
            True if this is a direct question
        """
        message_lower = message.lower().strip()
        
        # Check for question marks - strong indicator of a question
        if message_lower.endswith("?"):
            return True
            
        # Check for direct question indicators
        for indicator in self.direct_question_indicators:
            if message_lower.startswith(indicator):
                return True
                
        return False
        
    def _handle_followup_question(self, message: str) -> Tuple[bool, str]:
        """
        Check if the message is a follow-up question to the previous topic,
        particularly about advancements or latest developments.
        
        Args:
            message: The user's message
            
        Returns:
            Tuple of (is_followup, enriched_message)
        """
        message_lower = message.lower().strip()
        
        # Get the most recent topic discussed
        previous_topic = self.conversation_manager.get_topic()
        
        # If no previous topic, it's not a follow-up
        if not previous_topic:
            return False, message
            
        # Check for follow-up patterns
        is_followup = False
        for pattern in self.followup_patterns:
            if pattern in message_lower:
                is_followup = True
                break
                
        # If this looks like a follow-up, enrich the message with the previous topic
        if is_followup:
            # Determine if this is likely about "latest" or "advancements"
            about_latest = any(word in message_lower for word in ["latest", "recent", "new", "current", "modern", "trending", "state of the art"])
            about_advancements = any(word in message_lower for word in ["advancements", "developments", "progress", "updates", "improvements"])
            
            # Construct an enriched message that includes the previous context
            if about_latest:
                enriched_message = f"What are the latest developments in {previous_topic}?"
            elif about_advancements:
                enriched_message = f"What are the recent advancements and developments in {previous_topic}?"
            else:
                enriched_message = f"{message} regarding {previous_topic}"
                
            logger.debug(f"Enriched follow-up message: {enriched_message}")
            return True, enriched_message
            
        return False, message
    
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
        yield from self._process_query(message)
    
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
            # Resume the pending query immediately without confirmation
            self.conversation_manager.set_processing()
            
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
    
    def _process_query(self, message: str, prioritize_response: bool = False) -> Iterator[str]:
        """
        Process a new query or follow-up question with streaming responses.
        
        Args:
            message: The user's message
            prioritize_response: Whether to prioritize responding over info collection
            
        Yields:
            Response chunks
        """
        try:
            # Always try to extract implicit user information from the message
            logger.debug("Extracting implicit user information for streaming response")
            self.info_collector.extract_user_info_from_message(message)
            
            # Check if this is a follow-up to the previous topic
            is_followup = self.intent_manager.is_followup_question(
                message, 
                self.conversation_manager.chat_history
            )
            
            # Determine intent
            matched_keywords = []
            if not is_followup or not self.current_intent:
                logger.debug("Classifying new intent for streaming response")
                # Classify the intent
                intent_result = self.intent_manager.classify_intent(message)
                intent = intent_result.get("intent", "UNKNOWN")
                confidence = intent_result.get("confidence", 0.0)
                matched_keywords = intent_result.get("matched_keywords", [])
                
                # For non-LLM topics, ensure we use the UNKNOWN intent to trigger fallback
                if intent == "UNKNOWN":
                    logger.info(f"Message classified as UNKNOWN with confidence {confidence}")
                    # Generate fallback message for non-LLM topics
                    complete_response = ""
                    for chunk in self.response_generator.generate_fallback_message_streaming(
                        message, 
                        self.conversation_manager.chat_history,
                        matched_keywords
                    ):
                        complete_response += chunk
                        yield chunk
                    
                    # Add to history
                    if complete_response:
                        self.conversation_manager.add_ai_message(complete_response)
                    self.conversation_manager.set_idle()
                    return
                
                # Store as current intent for follow-up questions
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
            
            # Only check for additional info needs if we're not prioritizing the response
            if not prioritize_response:
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
            
            # If we're here, we're proceeding with direct query response
            logger.info(f"Prioritizing streaming response to query: {message[:50]}...")
            
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
                self.conversation_manager.get_topic(),
                matched_keywords
            ):
                complete_response += chunk
                yield chunk
            
            # Add complete response to history
            if complete_response:
                self.conversation_manager.add_ai_message(complete_response)
            
            # Set state to idle
            self.conversation_manager.set_idle()
            
        except Exception as e:
            # If an error occurs during response generation, use the fallback
            error_msg = f"Error in streaming query processing: {str(e)}"
            logger.error(error_msg)
            
            # Set state to idle
            self.conversation_manager.set_idle()
            
            # Try to generate a fallback message
            try:
                # Try to use the streaming fallback generator
                for chunk in self.response_generator.generate_fallback_message_streaming(
                    message, 
                    self.conversation_manager.chat_history
                ):
                    yield chunk
            except Exception as fallback_error:
                # If even the fallback generation fails, provide a hardcoded message
                logger.error(f"Error generating fallback: {str(fallback_error)}")
                name = self.user_profile.get_attribute_value("name")
                
                if name:
                    fallback_msg = (f"I apologize, {name}, but I couldn't process your request. "
                                  "As a specialized LLM assistant, I can help with questions about language models, "
                                  "embeddings, transformers, and other LLM topics. "
                                  "Would you like to ask about one of these areas?")
                else:
                    fallback_msg = ("I apologize, but I couldn't process your request. "
                                  "As a specialized LLM assistant, I can help with questions about language models, "
                                  "embeddings, transformers, and other LLM topics. "
                                  "Would you like to ask about one of these areas?")
                
                yield fallback_msg
                # Add to chat history
                self.conversation_manager.add_ai_message(fallback_msg)
