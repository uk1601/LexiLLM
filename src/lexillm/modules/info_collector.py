"""
Information Collector for LexiLLM
Manages user information collection during conversations
"""

from typing import Dict, Any, Optional, Tuple, List

from ..utils import is_end_request
from ..user_profile import UserProfile, UserProfileManager
from ..config import PROFILE_CONFIG
from ..logger import get_logger
from ..exceptions import InfoCollectionError, UserProfileError
from .conversation_state import ConversationStateEnum, is_confirmation

# Get logger
logger = get_logger()

class InfoCollector:
    """
    Manages collection of user information during conversations.
    
    This class handles all aspects of collecting user information,
    including determining when to collect, what to collect, and
    processing responses.
    """
    
    def __init__(self, profile_manager: UserProfileManager, user_profile: UserProfile, 
                 conversation_manager=None):
        """
        Initialize the information collector.
        
        Args:
            profile_manager: Manager for user profiles
            user_profile: Current user profile
            conversation_manager: Conversation state manager
        """
        self.profile_manager = profile_manager
        self.user_profile = user_profile
        self.conversation_manager = conversation_manager
        
        # Initialize onboarding state
        self.in_onboarding = not self.user_profile.onboarding_completed
        
        # Flag to track if we're actively collecting information
        self.collecting_info = False
        self.current_collection_attribute = None
        
        logger.info(f"Info collector initialized, onboarding: {self.in_onboarding}")
        
        # If we're in onboarding and have a conversation manager, set the state
        if self.in_onboarding and self.conversation_manager:
            self.conversation_manager.start_onboarding()
    
    def check_for_info_collection_opportunity(self) -> Tuple[bool, Optional[str]]:
        """
        Check if we should collect user information now.
        
        Returns:
            Tuple of (should_collect, attribute_to_collect)
        """
        # Don't interrupt if we're already collecting info or in onboarding
        if self.is_collecting_info() or self.is_in_onboarding():
            logger.debug("No collection opportunity: already collecting or in onboarding")
            return False, None
        
        # Don't interrupt if we're awaiting confirmation (if we have a conversation manager)
        if self.conversation_manager and self.conversation_manager.is_awaiting_confirmation():
            logger.debug("No collection opportunity: awaiting confirmation")
            return False, None
        
        # Don't interrupt active topic discussions (check conversation state)
        if (self.conversation_manager and 
            self.conversation_manager.get_current_state() in 
            [ConversationStateEnum.PROCESSING.value, ConversationStateEnum.RESPONDING.value]):
            logger.debug("No collection opportunity: user is in active topic discussion")
            return False, None
        
        # Check if recent messages are part of a natural flow - don't interrupt follow-ups
        if self.conversation_manager and len(self.conversation_manager.chat_history.messages) >= 4:
            # Get most recent user message
            recent_messages = self.conversation_manager.chat_history.messages[-4:]
            recent_user_msgs = [msg.content for msg in recent_messages if msg.type == "human"]
            
            # If recent message is very short (1-2 words), it's likely a follow-up or response
            # to a question - not a good time to interrupt for information collection
            if recent_user_msgs and len(recent_user_msgs[-1].split()) <= 2:
                logger.debug("No collection opportunity: user is in the middle of a follow-up exchange")
                return False, None
            
        # Get threshold values from configuration - increase thresholds
        interaction_threshold = PROFILE_CONFIG.get("interaction_threshold", 4) + 2  # Increased threshold
        collection_interval = PROFILE_CONFIG.get("collection_interval", 5) + 3  # Increased interval
        
        # Don't collect information too frequently
        if self.user_profile.interaction_count < interaction_threshold:
            logger.debug(f"No collection opportunity: not enough interactions ({self.user_profile.interaction_count})")
            return False, None
            
        # Every few interactions, check if there's an attribute we should collect
        if self.user_profile.interaction_count % collection_interval != 0:
            logger.debug(f"No collection opportunity: not at collection interval ({self.user_profile.interaction_count})")
            return False, None
        
        # Only collect information during natural breaks in conversation
        # Check if the conversation is at a good breaking point
        if self.conversation_manager:
            # If we've asked a question in the last response, wait for the answer first
            last_bot_msg = ""
            for msg in reversed(self.conversation_manager.chat_history.messages):
                if msg.type == "ai":
                    last_bot_msg = msg.content
                    break
            
            # If our last message ended with a question, skip collection for now
            if last_bot_msg and ("?" in last_bot_msg[-30:] or 
                               any(phrase in last_bot_msg[-50:].lower() for phrase in 
                                  ["would you like", "could you", "can you", "what do you", "how about"])):
                logger.debug("No collection opportunity: bot's last message ended with a question")
                return False, None
        
        # Check if there's an attribute we should collect
        next_attr = self.user_profile.get_next_attribute_to_collect()
        if next_attr:
            logger.info(f"Opportunity to collect user information: {next_attr}")
            return True, next_attr
            
        logger.debug("No collection opportunity: no attributes to collect")
        return False, None
    
    def determine_if_more_info_needed(self, intent: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if more information is needed for a specific intent.
        
        Args:
            intent: The detected intent
            
        Returns:
            Tuple of (needs_more_info, info_type_needed)
        """
        try:
            # Check which attribute is needed for this intent based on value existence
            intent_attribute_map = {
                "LLM_FUNDAMENTALS": "technical_level",
                "LLM_IMPLEMENTATION": "project_stage",
                "LLM_COMPARISON": "comparison_criterion",
                "LLM_NEWS": "interest_area"
            }
            
            # Confidence threshold for requiring explicit collection
            confidence_threshold = 0.6
            
            # If the intent requires specific information
            if intent in intent_attribute_map:
                attr_name = intent_attribute_map[intent]
                attr_value = self.user_profile.get_attribute_value(attr_name)
                attr_confidence = self.user_profile.get_attribute_confidence(attr_name)
                
                # Need to collect if value is None or confidence is too low
                if attr_value is None:
                    logger.info(f"Intent {intent} requires attribute {attr_name} which is missing")
                    return True, attr_name
                elif attr_confidence < confidence_threshold:
                    logger.info(f"Intent {intent} requires attribute {attr_name} which has low confidence ({attr_confidence})")
                    return True, attr_name
            
            logger.debug(f"No additional information needed for intent {intent}")
            return False, None
            
        except Exception as e:
            logger.error(f"Error determining if more info needed: {str(e)}")
            return False, None
    
    def get_info_collection_message(self, info_type: str) -> str:
        """
        Generate a message to collect information of a specific type.
        
        Args:
            info_type: The type of information to collect
            
        Returns:
            Message asking for the information
        """
        try:
            message = self.profile_manager.generate_collection_message(info_type, self.user_profile)
            logger.debug(f"Generated collection message for {info_type}")
            return message
        except Exception as e:
            logger.error(f"Error generating info collection message: {str(e)}")
            # Fallback to a generic message
            name_prefix = ""
            if self.user_profile.name and self.user_profile.name.value:
                name_prefix = f"{self.user_profile.name.value}, "
            return f"{name_prefix}could you tell me about your {info_type.replace('_', ' ')}? This helps me provide more relevant information."
    
    def process_explicit_info_collection(self, message: str) -> Optional[str]:
        """
        Process a response to an explicit information collection question.
        
        Args:
            message: The user's response to the information collection question
            
        Returns:
            Next message to send to the user, or None if collection is complete
        """
        try:
            logger.debug(f"Processing explicit info collection for {self.current_collection_attribute}")
            
            # Validate state
            if not self.current_collection_attribute:
                logger.warning("Attempted to process info collection with no current attribute")
                return None
            
            # Store the current attribute being collected
            current_attr = self.current_collection_attribute
            
            # Process the response for the current attribute
            value, confidence = self.profile_manager.process_explicit_response(
                current_attr, message
            )
            
            # Update the profile with this information
            self.user_profile.update_attribute(
                current_attr, value, confidence, "explicit"
            )
            
            # Log the update
            logger.info(f"Updated user profile attribute: {current_attr} = {value} (confidence: {confidence})")
            
            # Save the updated profile immediately
            self.profile_manager.save_profile(self.user_profile)
            
            # Reset current collection state to avoid loops
            self.current_collection_attribute = None
            self.collecting_info = False
            
            # If using the conversation manager, update its state
            if self.conversation_manager:
                self.conversation_manager.end_info_collection()
            
            # If we're in onboarding, check if we need to ask for more information
            if self.in_onboarding:
                # Check for missing core attributes
                missing_attrs = self.user_profile.get_missing_core_attributes()
                logger.debug(f"Onboarding - Missing attributes: {missing_attrs}")
                
                if missing_attrs:
                    # Find next attribute to collect (other than the one we just collected)
                    next_attr = None
                    for attr in self.user_profile.CORE_ATTRIBUTES:
                        if attr in missing_attrs and attr != current_attr:
                            next_attr = attr
                            break
                    
                    if next_attr:
                        logger.info(f"Moving to next onboarding attribute: {next_attr}")
                        self.start_info_collection(next_attr)
                        return self.get_info_collection_message(next_attr)
                
                # If no more core attributes are missing, complete onboarding
                logger.info("All core attributes collected, completing onboarding")
                self.user_profile.complete_onboarding()
                self.in_onboarding = False
                
                # Update conversation manager if available
                if self.conversation_manager:
                    self.conversation_manager.complete_onboarding()
                
                # Generate personalized welcome
                name = self.user_profile.get_attribute_value("name") or "there"
                tech_level = self.user_profile.get_attribute_value("technical_level") or "intermediate"
                interest = self.user_profile.get_attribute_value("interest_area") or "research"
                
                welcome = f"Thanks for sharing that information, {name}! "
                welcome += f"I'll tailor my responses to your {tech_level} level and focus on {interest}. "
                welcome += "What would you like to know about LLMs today?"
                
                logger.debug(f"Generated personalized welcome for {name}")
                return welcome
            else:
                # Single attribute collection completed
                logger.debug(f"Single attribute collection for {current_attr} completed")
                
                # If we have a conversation manager and there's a pending query, resume it
                if self.conversation_manager and self.conversation_manager.get_pending_query():
                    pending = self.conversation_manager.get_pending_query()
                    
                    # Generate a resume message - Don't mention the original query directly
                    # to avoid interpreting "yes"/"no" as topics
                    name_prefix = ""
                    if self.user_profile.get_attribute_value("name"):
                        name_prefix = f"{self.user_profile.get_attribute_value('name')}, "
                        
                    # Get the technical level and other relevant attributes
                    technical_level = self.user_profile.get_attribute_value("technical_level") or "beginner"
                    depth_preference = self.user_profile.get_attribute_value("depth_preference") or "standard"
                    
                    # Log what we're doing
                    logger.info(f"Resuming pending query about '{pending['topic']}' with technical_level={technical_level}, depth_preference={depth_preference}")
                    
                    # Construct a message acknowledging the completion of information collection
                    # and indicating we're now addressing their original query
                    resume_msg = f"Thanks for providing that information. "
                    resume_msg += f"Now, let me answer your question about {pending['topic']} "
                    
                    # Use both technical_level and depth_preference if depth is specified
                    if depth_preference != "standard":
                        resume_msg += f"with {depth_preference} technical details at a {technical_level} level."
                    else:
                        resume_msg += f"at a {technical_level} level of detail."
                    
                    return resume_msg
                
                # Otherwise, just return None to indicate completion
                return None
                
        except Exception as e:
            logger.error(f"Error in process_explicit_info_collection: {str(e)}")
            self.collecting_info = False
            self.current_collection_attribute = None
            
            # Reset conversation manager state if available
            if self.conversation_manager:
                self.conversation_manager.end_info_collection()
                
            raise InfoCollectionError(f"Failed to process information collection: {str(e)}", 
                                     attribute=self.current_collection_attribute)
    
    def extract_user_info_from_message(self, message: str) -> List[Tuple[str, Any]]:
        """
        Extract user information from a message.
        
        Args:
            message: The user's message
            
        Returns:
            List of tuples (attribute_name, value) of information that was extracted
        """
        try:
            # Skip extraction for very short messages like "yes" or "no"
            # These are likely confirmation responses and not useful for extraction
            if len(message.strip()) <= 5:
                logger.debug(f"Skipping extraction for short message: '{message}'")
                return []
                
            logger.debug("Attempting to extract implicit user information from message")
            updated_attrs = self.profile_manager.update_profile_from_message(self.user_profile, message)
            
            # If we extracted anything, save the profile
            if updated_attrs:
                self.profile_manager.save_profile(self.user_profile)
                logger.info(f"Extracted {len(updated_attrs)} user attributes implicitly: {updated_attrs}")
            else:
                logger.debug("No user information extracted from message")
            
            return updated_attrs
        except Exception as e:
            logger.error(f"Error extracting user info from message: {str(e)}")
            return []
    
    def is_collecting_info(self) -> bool:
        """
        Check if we're currently collecting user information.
        
        Returns:
            Boolean indicating if we're collecting information
        """
        # If we have a conversation manager, use its state
        if self.conversation_manager:
            return self.conversation_manager.is_collecting_info()
        # Otherwise use our internal state
        return self.collecting_info
    
    def start_info_collection(self, attribute: str) -> None:
        """
        Start collecting a specific attribute.
        
        Args:
            attribute: The attribute to collect
        """
        logger.info(f"Starting collection for attribute: {attribute}")
        self.collecting_info = True
        self.current_collection_attribute = attribute
        
        # Update conversation manager if available
        if self.conversation_manager:
            self.conversation_manager.start_info_collection(attribute)
    
    def stop_info_collection(self) -> None:
        """Stop collecting information."""
        logger.info("Stopping information collection")
        self.collecting_info = False
        self.current_collection_attribute = None
        
        # Update conversation manager if available
        if self.conversation_manager:
            self.conversation_manager.end_info_collection()
    
    def get_current_collection_attribute(self) -> Optional[str]:
        """
        Get the attribute currently being collected.
        
        Returns:
            The attribute being collected, or None if not collecting
        """
        # If we have a conversation manager, use its state
        if self.conversation_manager:
            return self.conversation_manager.get_collecting_attribute()
        # Otherwise use our internal state
        return self.current_collection_attribute
    
    def is_in_onboarding(self) -> bool:
        """
        Check if we're currently in the onboarding process.
        
        Returns:
            Boolean indicating if we're in onboarding
        """
        # If we have a conversation manager, use its state
        if self.conversation_manager:
            return self.conversation_manager.is_in_onboarding()
        # Otherwise use our internal state
        return self.in_onboarding
    
    def reset_collection_state(self) -> None:
        """Reset the collection state to default values."""
        logger.info("Resetting collection state")
        self.collecting_info = False
        self.current_collection_attribute = None
        self.in_onboarding = not self.user_profile.onboarding_completed
        
        # Reset conversation manager if available
        if self.conversation_manager:
            if self.in_onboarding:
                self.conversation_manager.start_onboarding()
            else:
                self.conversation_manager.set_idle()
    
    def check_for_confirmation(self, message: str) -> bool:
        """
        Check if a message is a confirmation.
        
        Args:
            message: The user's message
            
        Returns:
            True if message appears to be a confirmation
        """
        return is_confirmation(message)
    
    def set_conversation_manager(self, conversation_manager) -> None:
        """
        Set the conversation manager after initialization.
        
        Args:
            conversation_manager: The conversation manager to use
        """
        self.conversation_manager = conversation_manager
        
        # Update conversation state if in onboarding
        if self.in_onboarding:
            self.conversation_manager.start_onboarding()
        elif self.collecting_info:
            self.conversation_manager.start_info_collection(self.current_collection_attribute)
