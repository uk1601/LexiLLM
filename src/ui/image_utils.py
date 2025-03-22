"""
Image utilities for LexiLLM
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import os
import base64

def get_image_base64(image_path):
    """Get base64 encoding of an image file."""
    if not os.path.exists(image_path):
        return None
    
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def create_placeholder_logo():
    """Create a simple text-based logo if the image is not found."""
    return """
    <div style="text-align: center; 
                background-color: #4a89dc; 
                color: white; 
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