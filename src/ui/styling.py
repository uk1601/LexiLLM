"""
Styling for LexiLLM Streamlit Interface
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import streamlit as st

def apply_custom_css():
    """Apply custom CSS styling to the Streamlit interface."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="st-"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main .block-container {
            padding-top: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .status-container {
            padding: 0.75rem;
            border-radius: 8px;
            margin-top: 1rem;
            border-left: 4px solid;
        }
        
        .status-ready {
            background-color: #BBC6FF;
            border-left-color: #00cc8e;
        }
        
        .status-processing {
            background-color: #fff8e6;
            border-left-color: #ffcc00;
        }
        
        .status-error {
            background-color: #ffe6e6;
            border-left-color: #cc0000;
        }
        
        .category-card {
            padding: 1rem;
            border-radius: 10px;
            border-top: 4px solid;
            margin-bottom: 0.7rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .category-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .fundamentals {
            background-color: #BBC6FF;
            border-top-color: #0072b1;
        }
        
        .implementation {
            background-color: #AEAFAA;
            border-top-color: #00802b;
        }
        
        .comparison {
            background-color: #85D8EF;
            border-top-color: #8000ff;
        }
        
        .news {
            background-color: #EAAAF7;
            border-top-color: #cc0000;
        }
        
        /* Modified to remove fixed height */
        .chat-container {
            overflow-y: auto;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            margin-bottom: 1rem;
            background-color: #FAFBFB;
        }
        
        .timestamp {
            font-size: 0.8rem;
            color: #6c757d;
            text-align: right;
            margin-top: 0.2rem;
        }
        
        .typing-indicator {
            display: inline-block;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0% { opacity: 0.4; }
            50% { opacity: 1; }
            100% { opacity: 0.4; }
        }
        
        .category-pill {
            display: inline-block;
            padding: 0.3rem 0.6rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
        }
        
        .fundamentals-pill {
            background-color: #BBC6FF;
            color: #0072b1;
        }
        
        .implementation-pill {
            background-color: #AEAFAA;
            color: #00802b;
        }
        
        .comparison-pill {
            background-color: #85D8EF;
            color: #8000ff;
        }
        
        .news-pill {
            background-color: #EAAAF7;
            color: #cc0000;
        }
        
        .stChatInput {
            margin-bottom: 80px;
        }
        
        .footer {
            margin-top: 2rem;
            font-size: 0.8rem;
            color: #6c757d;
            text-align: center;
        }
        
        /* Make user profile pic container a bit nicer */
        .stChatMessageContent:has(img[src*='🧑‍💻']) div:first-child {
            background-color: #f0f7ff !important;
        }
        
        /* Make AI profile pic container a bit nicer */
        .stChatMessageContent:has(img[src*='🤖']) div:first-child {
            background-color: #f0fff5 !important;
        }
        
        /* Mobile-friendly adjustments */
        @media (max-width: 768px) {
            .category-card {
                padding: 0.7rem;
                font-size: 0.9rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

def configure_page():
    """Configure page settings."""
    st.set_page_config(
        page_title="LexiLLM: Your LLM Expert Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )