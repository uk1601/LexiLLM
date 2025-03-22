"""
Intent Manager for LexiLLM
Handle intent classification and related operations
"""

import time
from typing import Dict, Any, Optional, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from ..schemas import IntentClassifier
from ..templates import create_intent_prompt
from ..config import INTENT_CONFIG, CONVERSATION_CONFIG
from ..logger import get_logger
from ..exceptions import IntentClassificationError, TimeoutError

# Get logger
logger = get_logger()

class IntentManager:
    """
    Manages intent classification for LexiLLM.
    
    This class handles all aspects of determining the user's intent,
    including classification and follow-up detection.
    """
    
    def __init__(self, classifier_llm: ChatOpenAI):
        """
        Initialize the intent manager.
        
        Args:
            classifier_llm: LLM instance to use for classification
        """
        self.classifier_llm = classifier_llm
        self.intent_prompt = create_intent_prompt()
        self.setup_chains()
        logger.info("Intent manager initialized")
    
    def setup_chains(self):
        """Set up the classification chains."""
        # Intent classification using structured output
        try:
            self.intent_chain = self.intent_prompt | self.classifier_llm.with_structured_output(
                IntentClassifier,
                method="function_calling"
            )
            logger.debug("Intent classification chain created")
        except Exception as e:
            logger.error(f"Failed to set up intent classification chain: {str(e)}")
            raise IntentClassificationError(f"Failed to set up intent classification chain: {str(e)}")
    
    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Determine the user's intent from their input.
        
        Args:
            query: The user's message
            
        Returns:
            Dictionary with intent and confidence
        """
        try:
            logger.debug(f"Classifying intent for query: {query[:50]}...")
            result = self.intent_chain.invoke({"query": query})
            
            # Check confidence threshold
            if result.confidence < INTENT_CONFIG["unknown_confidence_threshold"]:
                logger.info(f"Intent confidence too low ({result.confidence}), defaulting to UNKNOWN")
                return {"intent": "UNKNOWN", "confidence": result.confidence}
            
            logger.info(f"Classified intent as {result.intent} with confidence {result.confidence}")
            return {"intent": result.intent, "confidence": result.confidence}
        except Exception as e:
            logger.error(f"Intent classification error: {str(e)}")
            return {"intent": "UNKNOWN", "confidence": 0.0}
    
    def is_followup_question(self, message: str, chat_history) -> bool:
        """
        Determine if a message is a follow-up question to the previous topic.
        
        Args:
            message: The user's message
            chat_history: The conversation history
            
        Returns:
            Boolean indicating if this is a follow-up question
        """
        # Very short messages are often follow-ups
        short_message_threshold = CONVERSATION_CONFIG["short_message_threshold"]
        if len(message.split()) < short_message_threshold:
            logger.debug(f"Message is a follow-up (short message): {message}")
            return True
            
        # Check if this is a direct question without context
        if message.strip().endswith('?'):
            logger.debug(f"Message is a follow-up (ends with ?): {message}")
            return True
            
        # Check for follow-up phrases
        message_lower = message.lower()
        for phrase in INTENT_CONFIG["followup_phrases"]:
            if phrase in message_lower:
                logger.debug(f"Message is a follow-up (contains phrase '{phrase}'): {message}")
                return True
                
        # If we don't have much context yet, assume it's not a follow-up
        if len(chat_history.messages) < 4:
            logger.debug(f"Message is not a follow-up (not enough context): {message}")
            return False
        
        # For simple cases, avoid using the LLM to reduce latency
        # Only use LLM for ambiguous cases
        if len(chat_history.messages) >= 4 and len(message.split()) < 10:
            logger.debug(f"Message is a follow-up (ongoing conversation with short message): {message}")
            return True
            
        # Use a more sophisticated check using our LLM only for ambiguous cases
        try:
            # Create a simple prompt to check if it's a follow-up
            last_bot_message = chat_history.messages[-1].content if chat_history.messages else ""
            last_user_message = chat_history.messages[-2].content if len(chat_history.messages) > 1 else ""
            
            prompt = f"""Given the following conversation, determine if the last message is a follow-up question related to the previous topic:
            
            Previous bot message: {last_bot_message}
            Previous user message: {last_user_message}
            Current user message: {message}
            
            Is the current user message a follow-up question? Answer yes or no."""
            
            # Use a low temperature for this classification with a timeout
            start_time = time.time()
            # Set a maximum time for this operation
            max_time = CONVERSATION_CONFIG["followup_detection_timeout"]
            
            # Only run if we have enough time
            if time.time() - start_time < max_time:
                logger.debug("Using LLM to determine if message is a follow-up")
                response = self.classifier_llm.invoke(prompt).content.lower()
                result = "yes" in response
                logger.debug(f"LLM determined message is{'' if result else ' not'} a follow-up: {message}")
                return result
            else:
                logger.warning("LLM followup check timed out, using heuristic")
                result = len(message.split()) < short_message_threshold
                logger.debug(f"Using heuristic, message is{'' if result else ' not'} a follow-up: {message}")
                return result
        except Exception as e:
            logger.error(f"Error in follow-up detection: {str(e)}")
            # Fall back to simple rules-based approach
            result = len(message.split()) < short_message_threshold
            logger.debug(f"Error in follow-up detection, using heuristic. Message is{'' if result else ' not'} a follow-up: {message}")
            return result
