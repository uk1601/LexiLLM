"""
Utility functions for LexiLLM
"""

from typing import Dict, Tuple, Optional, Any

from .config import END_REQUEST_CONFIG, get_welcome_message, get_exit_reminder
from .logger import get_logger

# Get logger
logger = get_logger()

def is_end_request(query: str) -> bool:
    """
    Check if the user wants to end the conversation.
    
    Args:
        query: The user's message
        
    Returns:
        Boolean indicating if this is an end request
    """
    query = query.lower().strip()
    
    logger.debug(f"Checking if '{query}' is an end request")
    
    # Direct exact matches - these should always be considered end requests
    direct_matches = END_REQUEST_CONFIG["direct_matches"] 
    if query in direct_matches:
        logger.info(f"End request detected (exact match): '{query}'")
        return True
        
    # More specific end phrases to reduce false positives
    end_phrases = END_REQUEST_CONFIG["end_phrases"]
    
    # Check for phrases within the query
    for phrase in end_phrases:
        if phrase in query:
            logger.info(f"End request detected (phrase match): '{query}' contains '{phrase}'")
            return True
    
    # Check for positive responses to the "do you want to end" question
    if ("yes" in query or "yeah" in query or "sure" in query) and ("end" in query or "exit" in query or "stop" in query):
        logger.info(f"End request detected (affirmative response): '{query}'")
        return True
    
    logger.debug(f"'{query}' is not an end request")
    return False


def welcome_message() -> str:
    """Return the welcome message to start the conversation."""
    return get_welcome_message()


def get_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    
    This is a simplified estimation method. For more accurate results,
    use a tokenizer specific to the model being used.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated number of tokens
    """
    # Simple estimation: ~4 characters per token for English text
    return max(1, len(text) // 4)
