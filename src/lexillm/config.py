"""
Configuration module for LexiLLM

This module centralizes all configuration parameters for the LexiLLM package,
making it easier to modify settings without changing code throughout the application.
"""

import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
LLM_CONFIG = {
    # Model settings
    "default_model": os.environ.get("LEXILLM_MODEL", "gpt-4o"),
    "fallback_model": os.environ.get("LEXILLM_FALLBACK_MODEL", "gpt-3.5-turbo"),
    
    # API settings
    "api_key_env_var": "OPENAI_API_KEY",
    
    # Performance settings
    "main_temperature": float(os.environ.get("LEXILLM_TEMPERATURE", "0.7")),
    "classifier_temperature": float(os.environ.get("LEXILLM_CLASSIFIER_TEMPERATURE", "0.0")),
    "main_timeout": int(os.environ.get("LEXILLM_TIMEOUT", "30")),
    "classifier_timeout": int(os.environ.get("LEXILLM_CLASSIFIER_TIMEOUT", "15")),
    "max_retries": int(os.environ.get("LEXILLM_MAX_RETRIES", "2")),
}

# Conversation Management Configuration
CONVERSATION_CONFIG = {
    # History management
    "max_history_pairs": int(os.environ.get("LEXILLM_MAX_HISTORY_PAIRS", "10")),
    "preserve_initial_messages": int(os.environ.get("LEXILLM_PRESERVE_INITIAL_MESSAGES", "2")),
    
    # Followup detection
    "short_message_threshold": int(os.environ.get("LEXILLM_SHORT_MESSAGE_THRESHOLD", "5")),
    "followup_detection_timeout": float(os.environ.get("LEXILLM_FOLLOWUP_DETECTION_TIMEOUT", "2.0")),
}

# User Profile Configuration
PROFILE_CONFIG = {
    # Storage settings
    "storage_dir": os.environ.get("LEXILLM_PROFILE_STORAGE_DIR", "user_profiles"),
    
    # Attribute collection
    "interaction_threshold": int(os.environ.get("LEXILLM_INTERACTION_THRESHOLD", "2")),
    "collection_interval": int(os.environ.get("LEXILLM_COLLECTION_INTERVAL", "3")),
    "collection_confidence_threshold": float(os.environ.get("LEXILLM_COLLECTION_CONFIDENCE_THRESHOLD", "0.5")),
    "update_interval_days": int(os.environ.get("LEXILLM_UPDATE_INTERVAL_DAYS", "7")),
    
    # Default values
    "default_technical_level": "intermediate",
    "default_project_stage": "development",
    "default_comparison_criterion": "accuracy",
    "default_interest_area": "research",
    "default_depth_preference": "standard",
    
    # Core and advanced attributes
    "core_attributes": ["name", "technical_level", "interest_area"],
    "advanced_attributes": ["project_stage", "comparison_criterion", "depth_preference"],
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": os.environ.get("LEXILLM_LOG_LEVEL", "INFO"),
    "log_file": os.environ.get("LEXILLM_LOG_FILE", "lexillm.log"),
    "console_logging": os.environ.get("LEXILLM_CONSOLE_LOGGING", "True") == "True",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# Intent Classification Configuration
INTENT_CONFIG = {
    # Intent classification
    "unknown_confidence_threshold": float(os.environ.get("LEXILLM_UNKNOWN_CONFIDENCE_THRESHOLD", "0.6")),  # Increased threshold,
    
    # Followup detection
    "followup_phrases": [
        "tell me more", "how about", "what about", 
        "can you explain", "such as", "for example", 
        "like what", "in what way", "how are", "how is", 
        "how do", "how does", "how can", "how would", 
        "what is", "what are", "why is", "why are",
        "where", "when", "who", "which"
    ],
    
    # LLM Domain Filter - Enhanced with more political terms
    "non_llm_topics": [
        "president", "trump", "biden", "politics", "election", "weather", 
        "sports", "game", "movie", "actor", "celebrity", "singer", 
        "song", "music", "history", "war", "country", "city", "travel",
        "recipe", "food", "health", "disease", "medicine", "doctor",
        "news", "economy", "market", "price", "buy", "sell",
        "white house", "congress", "senate", "government", "politician",
        "campaign", "vote", "democracy", "republican", "democrat",
        "political party", "governor", "mayor", "administration", "policy",
        "foreign policy", "domestic policy", "supreme court", "justice"
    ],
    
    "llm_keywords": [
        "llm", "language model", "gpt", "bert", "transformer", "embedding", 
        "token", "prompt", "fine-tune", "vector", "attention", "nlp", 
        "natural language", "ai model", "neural network", "machine learning",
        "generative", "generation", "text generation", "chatgpt", "claude",
        "llama", "mistral", "gemini", "bard", "rag", "retrieval"
    ],
    
    # Higher threshold for stricter UNKNOWN detection
    "force_fallback_confidence": float(os.environ.get("LEXILLM_FORCE_FALLBACK_CONFIDENCE", "0.95")),  # Increased from 0.8
}

# End Request Detection Configuration
END_REQUEST_CONFIG = {
    # Direct matches
    "direct_matches": ["exit", "end", "bye", "goodbye", "quit", "stop"],
    
    # Phrases
    "end_phrases": [
        "exit conversation", "end conversation", 
        "end the chat", "quit the chat", "stop the conversation", "stop talking",
        "that's all", "i'm done", "we're done"
    ],
}

def get_exit_reminder() -> str:
    """Get the standard exit reminder message."""
    return " You can also say 'exit' or 'end' at any time to end our conversation."

def get_api_key() -> Optional[str]:
    """Get the API key from environment variables."""
    return os.environ.get(LLM_CONFIG["api_key_env_var"])

def get_info_collection_messages() -> Dict[str, str]:
    """Get the standard information collection messages."""
    exit_reminder = get_exit_reminder()
    
    return {
        "name": "Before we dive in, I'd love to know your name so I can address you properly." + exit_reminder,
        
        "technical_level": "To tailor my responses to your background, could you tell me your level of experience with Large Language Models? (Beginner/Intermediate/Advanced)" + exit_reminder,
        
        "interest_area": "What aspects of LLMs are you most interested in learning about? Research advances, practical applications, or something else?" + exit_reminder,
        
        "project_stage": "Are you currently working on an LLM project? If so, what stage are you in? (Planning/Development/Optimization)" + exit_reminder,
        
        "comparison_criterion": "When evaluating different LLM options, what's most important to you? (Accuracy/Speed/Cost)" + exit_reminder,
        
        "depth_preference": "How detailed would you like my explanations to be? Brief overviews, standard explanations, or in-depth technical details?" + exit_reminder,
        
        "background": "What's your background or field of expertise? This helps me provide more relevant examples." + exit_reminder,
        
        "experience_with_llms": "Have you worked with any specific LLM models or frameworks before?" + exit_reminder,
        
        "project_goal": "What's the main goal or use case for your LLM project?" + exit_reminder,
        
        "industry": "Which industry or domain are you applying LLMs to?" + exit_reminder,
    }

def get_welcome_message() -> str:
    """Get the standard welcome message."""
    exit_reminder = get_exit_reminder()
    
    return ("Welcome to LexiLLM! I'm your specialized assistant for navigating the world of "
            "Large Language Models. I can help you understand LLM fundamentals, "
            "provide implementation guidance, compare different models, or discuss "
            "the latest trends in the field. What would you like to explore today?" + 
            exit_reminder)
