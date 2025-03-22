"""
Utility functions for LexiLLM core
"""

def welcome_message() -> str:
    """
    Generate a welcome message for returning users.
    
    Returns:
        Welcome message text
    """
    return (
        "Welcome to LexiLLM! I'm your specialized assistant for Large Language Models. "
        "I can help with technical questions, implementation advice, model comparisons, "
        "and the latest developments in the field. How can I assist you today?"
    )

def is_end_request(message: str) -> bool:
    """
    Check if a message is requesting to end the conversation.
    
    Args:
        message: The user's message
        
    Returns:
        True if this is an end request
    """
    message = message.lower().strip()
    end_phrases = [
        "exit", "end", "quit", "stop", "bye", "goodbye", 
        "see you", "farewell", "terminate", "close"
    ]
    
    # Check for exact matches first
    if message in end_phrases:
        return True
    
    # Check for phrases starting with end words
    for phrase in end_phrases:
        if message.startswith(phrase):
            return True
        
    return False
