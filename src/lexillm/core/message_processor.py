"""
Message processor for LexiLLM
Handles non-streaming message processing
"""

from typing import Dict, Tuple, Optional, Any, List

from ..logger import get_logger
from ..modules.conversation_state import ConversationStateEnum, is_confirmation, is_rejection

# Get logger
logger = get_logger()

class MessageProcessor:
    """
    Process user messages and generate responses.
    
    This class handles the core message processing logic for non-streaming responses.
    """
    
    def __init__(self, conversation_manager, intent_manager, response_generator, 
                info_collector, profile_manager, user_profile):
        """
        Initialize the message processor.
        
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
    
    def process(self, message: str) -> Tuple[str, Optional[str]]:
        """
        Process a user message and return a response.
        
        Args:
            message: The user's message
            
        Returns:
            Tuple of (response, current_intent)
        """
        try:
            # Log the incoming message
            logger.info(f"Processing message: {message[:50]}...")
            
            # Track this interaction
            self.user_profile.track_interaction(self.conversation_manager.get_topic())
            
            # Add the user's message to history
            self.conversation_manager.add_user_message(message)
            
            # Check if this is a follow-up question about advancements or latest developments
            is_followup, enriched_message = self._handle_followup_question(message)
            if is_followup:
                logger.info(f"Detected follow-up question about advancements/latest developments: {message}")
                # Process the enriched message directly
                message = enriched_message
                return self._process_query(message, prioritize_response=True), self.current_intent
            
            # Check if this is a direct question that should be prioritized over information collection
            if self._is_direct_question(message):
                logger.info(f"Detected direct question: {message}")
                # Process the query with priority over information collection
                return self._process_query(message, prioritize_response=True), self.current_intent
                
            # Get the current conversation state
            current_state = self.conversation_manager.get_current_state()
            logger.debug(f"Current conversation state: {current_state}")
            
            # Check if we're awaiting confirmation
            if self.conversation_manager.is_awaiting_confirmation():
                response = self._handle_confirmation_state(message)
                if response:
                    return response, self.current_intent
            
            # Check if we're collecting information
            if self.info_collector.is_collecting_info():
                response = self._handle_info_collection_state(message)
                if response:
                    return response, self.current_intent
            
            # If we're here, we're processing a new query or follow-up
            return self._process_query(message), self.current_intent
            
        except Exception as e:
            # If an error occurs, provide a graceful fallback response
            error_msg = f"Error in process_message: {str(e)}"
            logger.error(error_msg)
            
            # Reset state to idle
            self.conversation_manager.set_idle()
            
            # Personalize the fallback message if we know the user's name
            name = self.user_profile.get_attribute_value("name")
            if name:
                fallback = self._generate_error_fallback(name)
            else:
                fallback = self._generate_error_fallback()
            
            return fallback, self.current_intent
    
    def _generate_error_fallback(self, name: Optional[str] = None) -> str:
        """
        Generate a fallback message for error cases.
        
        Args:
            name: Optional user name for personalization
            
        Returns:
            A fallback message
        """
        try:
            # Try to use the dynamic fallback template
            logger.debug("Generating error fallback message with dynamic fallback template")
            technical_level = self.user_profile.get_attribute_value("technical_level") or ""
            
            response = self.response_generator.generate_dynamic_fallback(
                "I encountered an error processing your message", 
                self.conversation_manager.chat_history,
                related_topics=[],
                technical_level=technical_level,
                user_name=name
            )
            return response
        except Exception as e:
            # If even the fallback generation fails, use a hardcoded message
            logger.error(f"Error generating fallback message: {str(e)}")
            
            if name:
                return (f"I apologize, {name}, but I encountered an error while processing your message. "
                       "As a specialized LLM assistant, I can help with questions about language models, "
                       "their architecture, implementation, or recent developments. Would you like to "
                       "try asking an LLM-related question?")
            else:
                return ("I apologize, but I encountered an error while processing your message. "
                       "As a specialized LLM assistant, I can help with questions about language models, "
                       "their architecture, implementation, or recent developments. Would you like to "
                       "try asking an LLM-related question?")
    
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
    
    def _handle_confirmation_state(self, message: str) -> Optional[str]:
        """
        Handle messages when in the confirmation state.
        
        Args:
            message: The user's message
            
        Returns:
            Response or None if no confirmation handling needed
        """
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
        logger.info("Message not recognized as confirmation, treating as new query")
        self.conversation_manager.clear_confirmation()
        return None  # No confirmation handling needed, continue with normal processing
    
    def _handle_info_collection_state(self, message: str) -> Optional[str]:
        """
        Handle messages when in the information collection state.
        
        Args:
            message: The user's message
            
        Returns:
            Response or None if no info collection handling needed
        """
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
            # Resume the pending query immediately without waiting for confirmation
            self.conversation_manager.set_processing()
            
            # Generate the actual response using the pending query, not asking for confirmation
            response = self.response_generator.generate_response(
                pending_query["message"],
                pending_query["intent"],
                self.conversation_manager.chat_history,
                self.user_profile,
                pending_query["topic"]  # Use the original topic
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
        
        return None  # Should never reach here, but for type safety
    
    def _process_query(self, message: str, prioritize_response: bool = False) -> str:
        """
        Process a new query or follow-up question.
        
        Args:
            message: The user's message
            prioritize_response: Whether to prioritize responding over info collection
            
        Returns:
            Response to the query
        """
        try:
            # Always try to extract implicit user information from the message
            logger.debug("Extracting implicit user information")
            self.info_collector.extract_user_info_from_message(message)
            
            # Check if this is a follow-up to the previous topic
            is_followup = self.intent_manager.is_followup_question(
                message, 
                self.conversation_manager.chat_history
            )
            
            # Determine intent
            matched_keywords = []
            if not is_followup or not self.current_intent:
                logger.debug("Classifying new intent")
                # Classify the intent
                intent_result = self.intent_manager.classify_intent(message, self.conversation_manager.chat_history)
                intent = intent_result.get("intent", "UNKNOWN")
                confidence = intent_result.get("confidence", 0.0)
                related_topics = intent_result.get("related_topics", [])
                reasoning = intent_result.get("reasoning", "")
                
                # For non-LLM topics, ensure we use the UNKNOWN intent to trigger fallback
                if intent == "UNKNOWN":
                    logger.info(f"Message classified as UNKNOWN with confidence {confidence}")
                    
                    # Get user's technical level if known
                    technical_level = self.user_profile.get_attribute_value("technical_level") or ""
                    
                    # Get user's name if known
                    user_name = self.user_profile.get_attribute_value("name")
                    
                    # Generate a personalized fallback message
                    fallback_response = self.response_generator.generate_dynamic_fallback(
                        message, 
                        self.conversation_manager.chat_history,
                        related_topics=related_topics,
                        relevance_confidence=confidence,
                        technical_level=technical_level,
                        user_name=user_name,
                        reasoning=reasoning
                    )
                    self.conversation_manager.add_ai_message(fallback_response)
                    self.conversation_manager.set_idle()
                    return fallback_response
                
                # Store as current intent for follow-up questions
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
            
            # Only check for additional info needs if we're not prioritizing the response
            if not prioritize_response:
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
            
            # If we're here, we're responding to the query directly
            logger.info(f"Prioritizing response to query: {message[:50]}...")
            
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
                self.conversation_manager.get_topic(),
                matched_keywords
            )
            
            # Set state to idle
            self.conversation_manager.set_idle()
            
            # Add response to history
            self.conversation_manager.add_ai_message(response)
            
            return response
            
        except Exception as e:
            # If we encounter an error, use the fallback template to provide a helpful response
            error_msg = f"Error in process_query: {str(e)}"
            logger.error(error_msg)
            
            # Set state to idle
            self.conversation_manager.set_idle()
            
            # Generate a fallback message
            try:
                # Get user name and technical level if available
                user_name = self.user_profile.get_attribute_value("name")
                technical_level = self.user_profile.get_attribute_value("technical_level") or ""
                
                fallback_response = self.response_generator.generate_dynamic_fallback(
                    message, 
                    self.conversation_manager.chat_history,
                    related_topics=[],
                    technical_level=technical_level,
                    user_name=user_name
                )
                self.conversation_manager.add_ai_message(fallback_response)
                return fallback_response
            except Exception as fallback_error:
                # If even the fallback generation fails, provide a hardcoded message
                logger.error(f"Error generating fallback: {str(fallback_error)}")
                name = self.user_profile.get_attribute_value("name")
                
                if name:
                    return (f"I apologize, {name}, but I couldn't process your request. "
                           "As a specialized LLM assistant, I can help with questions about language models, "
                           "embeddings, transformers, prompt engineering, and other LLM topics. "
                           "Would you like to ask about one of these areas?")
                else:
                    return ("I apologize, but I couldn't process your request. "
                           "As a specialized LLM assistant, I can help with questions about language models, "
                           "embeddings, transformers, prompt engineering, and other LLM topics. "
                           "Would you like to ask about one of these areas?")
