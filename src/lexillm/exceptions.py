"""
Custom exceptions for LexiLLM

This module defines custom exception classes for different error scenarios
in the LexiLLM package, providing more specific error handling and reporting.
"""

class LexiLLMError(Exception):
    """Base exception class for all LexiLLM errors."""
    pass


class APIKeyError(LexiLLMError):
    """Exception raised when there are issues with the API key."""
    pass


class ModelError(LexiLLMError):
    """Exception raised when there are issues with the LLM model."""
    
    def __init__(self, message, model_name=None, status_code=None):
        self.model_name = model_name
        self.status_code = status_code
        super().__init__(message)


class IntentClassificationError(LexiLLMError):
    """Exception raised when intent classification fails."""
    
    def __init__(self, message, query=None):
        self.query = query
        super().__init__(message)


class ResponseGenerationError(LexiLLMError):
    """Exception raised when response generation fails."""
    
    def __init__(self, message, intent=None):
        self.intent = intent
        super().__init__(message)


class UserProfileError(LexiLLMError):
    """Exception raised when there are issues with user profiles."""
    
    def __init__(self, message, user_id=None, attribute=None):
        self.user_id = user_id
        self.attribute = attribute
        super().__init__(message)


class ConversationError(LexiLLMError):
    """Exception raised when there are issues with conversation management."""
    pass


class InfoCollectionError(LexiLLMError):
    """Exception raised when information collection fails."""
    
    def __init__(self, message, attribute=None):
        self.attribute = attribute
        super().__init__(message)


class ConfigurationError(LexiLLMError):
    """Exception raised when there are issues with configuration."""
    
    def __init__(self, message, config_key=None):
        self.config_key = config_key
        super().__init__(message)


class StreamingError(LexiLLMError):
    """Exception raised when streaming responses fail."""
    pass


class TimeoutError(LexiLLMError):
    """Exception raised when an operation times out."""
    
    def __init__(self, message, operation=None, timeout_seconds=None):
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        super().__init__(message)


class ValidationError(LexiLLMError):
    """Exception raised when data validation fails."""
    
    def __init__(self, message, field=None, value=None):
        self.field = field
        self.value = value
        super().__init__(message)
