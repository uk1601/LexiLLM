"""
Session state management for LexiLLM Streamlit Interface
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import streamlit as st
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import logging

from lexillm import LexiLLM

def initialize_state():
    """Initialize or reset the session state."""
    # Initialize basic state variables if they don't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processing" not in st.session_state:
        st.session_state.processing = False
        
    # Generate a session ID if one doesn't exist
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

def get_api_key():
    """
    Get the OpenAI API key from environment variables or Streamlit secrets.
    
    Returns:
        str: The API key if found, None otherwise
    """
    # First try to load from .env file for local development
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # If not found in environment, try Streamlit secrets
    if not api_key and hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        
    return api_key

def create_bot():
    """Create a new bot instance."""
    try:
        # Get API key
        api_key = get_api_key()
        
        if not api_key:
            st.error("No OpenAI API key found. Please set it in the .env file or Streamlit secrets.")
            logging.error("API key missing")
            return None
            
        # Use session_id for profile persistence
        bot = LexiLLM(api_key=api_key, streaming=True, session_id=st.session_state.session_id)
        logging.info(f"Created new bot instance with session ID: {st.session_state.session_id}")
        return bot
    except Exception as e:
        logging.error(f"Error creating bot: {str(e)}")
        st.error(f"Error initializing LexiLLM: {str(e)}")
        return None

def initialize_bot():
    """Initialize the bot and add welcome message if needed."""
    if "bot" not in st.session_state or st.session_state.bot is None:
        st.session_state.bot = create_bot()
        
        # Add welcome message if bot was created and no messages exist
        if st.session_state.bot and not st.session_state.messages:
            welcome = st.session_state.bot.welcome_message()
            st.session_state.messages.append({
                "role": "assistant",
                "content": welcome,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "category": "general"
            })
            logging.info("Added welcome message")

def reset_conversation():
    """Reset the conversation state."""
    # Generate a new session ID for a fresh start
    st.session_state.session_id = str(uuid.uuid4())
    logging.info(f"Generated new session ID: {st.session_state.session_id}")
    
    # Create a new bot instance with the new session ID
    st.session_state.bot = create_bot()
    
    # Reset messages but keep welcome message
    if st.session_state.bot:
        welcome = st.session_state.bot.welcome_message()
        st.session_state.messages = [{
            "role": "assistant",
            "content": welcome,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "category": "general"
        }]
    else:
        st.session_state.messages = []
    
    # Reset processing flag
    st.session_state.processing = False