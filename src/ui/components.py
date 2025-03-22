"""
UI Components for LexiLLM Streamlit Interface
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import streamlit as st
from datetime import datetime
import os
import sys

# Add direct import for image utils functions
def create_placeholder_logo():
    """Create a simple text-based logo if the image is not found."""
    return """
    <div style="text-align: center; 
                background-color: ##6298f0; 
                color: black; 
                font-weight: bold; 
                padding: 10px; 
                border-radius: 50%; 
                width: 80px; 
                height: 80px; 
                display: flex; 
                align-items: center; 
                justify-content: center;">
        <div>LexiLLM</div>
    </div>
    """

def render_header():
    """Render the app header with logo and title."""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # Check if SVG exists, otherwise use placeholder
        logo_path = os.path.join(os.getcwd(), "assets", "lexillm_logo.jpeg")
        if os.path.exists(logo_path):
            st.image(logo_path, width=80)
        else:
            st.markdown(create_placeholder_logo(), unsafe_allow_html=True)
    
    with col2:
        st.title("LexiLLM: Your LLM Expert Assistant")
        st.markdown("*Navigating Large Language Models with expertise and clarity*")

def render_category_cards():
    """Render the category cards at the top of the interface."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="category-card fundamentals">
            <strong>🧠 Fundamentals</strong><br>
            How transformers work, embeddings, attention mechanisms
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="category-card implementation">
            <strong>⚙️ Implementation</strong><br>
            Fine-tuning, RAG, reducing hallucinations, deployment
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="category-card comparison">
            <strong>⚖️ Comparison</strong><br>
            Compare models like GPT-4, Claude, Llama
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="category-card news">
            <strong>📰 News & Trends</strong><br>
            Latest research, applications, and developments
        </div>
        """, unsafe_allow_html=True)

def render_chat_container(messages):
    """Render the chat container with all messages."""
    # Only create and display the chat container if there are messages
    if not messages:
        return None
    
    chat_container = st.container()
    
    with chat_container:
        # Display all messages
        for message in messages:
            role = message["role"]
            content = message["content"]
            timestamp = message.get("timestamp", "")
            category = message.get("category", "general")
            
            with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
                if role == "assistant" and category != "general":
                    st.markdown(get_category_pill(category), unsafe_allow_html=True)
                
                st.markdown(content)
                if timestamp:
                    st.markdown(f'<div class="timestamp">{timestamp}</div>', unsafe_allow_html=True)
    
    return chat_container

def render_sidebar(processing):
    """Render the sidebar with about info and status."""
    with st.sidebar:
        st.title("About LexiLLM")
        
        st.markdown("""
        <div style="padding: 1rem; background-color: #9DC1FF; border-radius: 10px; border: 1px solid #e9ecef; margin-bottom: 1rem;">
            <h3 style="margin-top: 0;">LLM Expert Assistant</h3>
            <p>A specialized chatbot designed to assist with understanding and implementing Large Language Models.</p>
            <hr style="margin: 0.8rem 0; opacity: 0.3;">
            <p><strong>Features:</strong></p>
            <ul style="padding-left: 1.2rem; margin-bottom: 0;">
                <li>Intent recognition</li>
                <li>User information collection</li>
                <li>Personalized responses</li>
                <li>LLM expertise across domains</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Status indicator
        if processing:
            st.markdown("""
            <div class="status-container status-processing">
                <div>⏳ Processing your request...</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-container status-ready">
                <div>✅ Ready for your questions!</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Help section
        with st.expander("💡 How to use LexiLLM"):
            st.markdown("""
            - **Ask about LLM fundamentals**: "How do transformers work?"
            - **Implementation guidance**: "How to fine-tune a model for sentiment analysis?"
            - **Compare models**: "What's better for RAG - GPT-4 or Claude 3?"
            - **Get latest news**: "What are recent breakthroughs in LLMs?"
            - **End the conversation**: Say "exit" or "end" at any time
            """)
        
        # Reset button
        reset_button = st.button("🔄 Reset Conversation", use_container_width=True, disabled=processing)
        
        # Project info
        st.markdown("""
        <div class="footer" style="font-weight: bold; color: white;">
            <p>Developed by: Uday Kiran Dasari</p>
        </div>
        """, unsafe_allow_html=True)
        
        return reset_button

def get_category_pill(category):
    """Generate HTML for category pill."""
    pills = {
        "fundamentals": '<span class="category-pill fundamentals-pill">Fundamentals</span>',
        "implementation": '<span class="category-pill implementation-pill">Implementation</span>',
        "comparison": '<span class="category-pill comparison-pill">Comparison</span>',
        "news": '<span class="category-pill news-pill">News & Trends</span>'
    }
    return pills.get(category, '')

def get_intent_category(intent):
    """Map intent to category for styling."""
    categories = {
        "LLM_FUNDAMENTALS": "fundamentals",
        "LLM_IMPLEMENTATION": "implementation", 
        "LLM_COMPARISON": "comparison",
        "LLM_NEWS": "news"
    }
    return categories.get(intent, "general")

def get_response_timing(start_time):
    """Calculate response time and format it with timestamp."""
    end_time = datetime.now()
    response_time = round((end_time - start_time).total_seconds(), 2)
    timestamp = end_time.strftime("%H:%M:%S")
    return timestamp, response_time
