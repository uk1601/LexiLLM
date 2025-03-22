"""
Enhanced Streamlit UI for LexiLLM - Modular, user-centric implementation
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import os
import sys
import time
import logging
from datetime import datetime
import streamlit as st

# Add path to src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import UI components from our modular package
from ui import (
    # Page configuration and styling
    configure_page,
    apply_custom_css,
    
    # State management
    initialize_state,
    initialize_bot,
    reset_conversation,
    
    # UI components
    render_header,
    render_category_cards,
    render_chat_container,
    render_sidebar,
    get_intent_category,
    get_response_timing
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lexillm_ui.log"),
        logging.StreamHandler()
    ]
)

def main():
    """Main application."""
    # Configure page and apply styling
    configure_page()
    apply_custom_css()
    
    # Initialize session state and bot
    initialize_state()
    initialize_bot()
    
    # Render UI components
    render_header()
    render_category_cards()
    chat_container = render_chat_container(st.session_state.messages)
    reset_button = render_sidebar(st.session_state.processing)
    
    # Handle reset button
    if reset_button:
        reset_conversation()
        st.rerun()
    
    # Chat input
    user_input = st.chat_input(
        "Ask about LLMs...",
        disabled=st.session_state.processing,
        key="chat_input"
    )
    
    # Process user input with streaming
    if user_input and not st.session_state.processing:
        # Verify bot exists
        if st.session_state.bot is None:
            initialize_bot()
            if st.session_state.bot is None:
                st.error("Could not initialize bot. Please check your API key and try again.")
                st.stop()
        
        # Set processing flag
        st.session_state.processing = True
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Pre-classify intent for styling
        try:
            intent_result = st.session_state.bot.classify_intent(user_input)
            category = get_intent_category(intent_result.get("intent", "UNKNOWN"))
            logging.info(f"Classified intent: {intent_result.get('intent', 'UNKNOWN')}")
        except Exception as e:
            logging.error(f"Error classifying intent: {str(e)}")
            category = "general"
        
        # Stream the response
        with st.chat_message("assistant", avatar="🤖"):
            # Display category pill if applicable
            if category != "general":
                # Create category pill directly instead of importing
                category_pill = f'<span class="category-pill {category}-pill">{category.capitalize()}</span>'
                st.markdown(category_pill, unsafe_allow_html=True)
            
            # Placeholder for streaming text
            message_placeholder = st.empty()
            
            # Start timing
            start_time = datetime.now()
            
            try:
                # Create a wrapper for streaming
                def safe_stream(input_text):
                    try:
                        chunks = []
                        for chunk in st.session_state.bot.process_message_streaming(input_text):
                            chunks.append(chunk)
                            yield chunk
                    except Exception as e:
                        logging.error(f"Streaming error: {str(e)}")
                        error_msg = f"Error generating response: {str(e)}"
                        yield error_msg
                
                # Stream the response and display
                response_stream = safe_stream(user_input)
                full_response = ""
                
                for chunk in response_stream:
                    if chunk:  # Only append and display non-empty chunks
                        full_response += chunk
                        message_placeholder.markdown(
                            f"{full_response}<span class='typing-indicator'>▌</span>",
                            unsafe_allow_html=True
                        )
                
                # Final display without cursor
                timestamp, response_time = get_response_timing(start_time)
                
                if full_response.strip():  # Only update if there's a non-empty response
                    message_placeholder.markdown(
                        f"{full_response}\n\n<div class='timestamp'>{timestamp} · {response_time}s</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": f"{timestamp} · {response_time}s",
                        "category": category
                    })
                    
                    # If this was an end message, show a notification
                    from lexillm.utils import is_end_request
                    if is_end_request(user_input):
                        st.success("Conversation ended. You can start a new one by clicking 'Reset Conversation' or asking a new question.")
                
            except Exception as e:
                # Handle errors
                logging.error(f"Error in streaming response: {str(e)}")
                error_message = f"Sorry, an error occurred: {str(e)}"
                message_placeholder.markdown(error_message)
                
                # Add error to chat history
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "timestamp": timestamp,
                    "category": "general"
                })
        
        # Reset processing flag and refresh
        st.session_state.processing = False
        st.rerun()

if __name__ == "__main__":
    main()