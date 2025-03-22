"""
Unit tests for LexiLLM templates
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import unittest
from langchain_core.prompts import ChatPromptTemplate

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from lexillm.templates import create_templates, create_intent_prompt, create_extraction_prompt

class TestTemplates(unittest.TestCase):
    """Test cases for prompt templates."""
    
    def test_create_templates(self):
        """Test that templates are created correctly."""
        templates = create_templates()
        
        # Check that all expected templates exist
        expected_templates = [
            "LLM_FUNDAMENTALS", 
            "LLM_IMPLEMENTATION", 
            "LLM_COMPARISON", 
            "LLM_NEWS", 
            "FALLBACK", 
            "END_CONVERSATION"
        ]
        
        for template_name in expected_templates:
            self.assertIn(template_name, templates)
            self.assertIsInstance(templates[template_name], ChatPromptTemplate)
    
    def test_intent_prompt(self):
        """Test that the intent classification prompt is created correctly."""
        prompt = create_intent_prompt()
        self.assertIsInstance(prompt, ChatPromptTemplate)
        
        # Check that the prompt contains expected content
        prompt_str = str(prompt)
        self.assertIn("LLM_FUNDAMENTALS", prompt_str)
        self.assertIn("LLM_IMPLEMENTATION", prompt_str)
        self.assertIn("LLM_COMPARISON", prompt_str)
        self.assertIn("LLM_NEWS", prompt_str)
        self.assertIn("UNKNOWN", prompt_str)
    
    def test_extraction_prompt(self):
        """Test that the user info extraction prompt is created correctly."""
        prompt = create_extraction_prompt()
        self.assertIsInstance(prompt, ChatPromptTemplate)
        
        # Check that the prompt contains expected content
        prompt_str = str(prompt)
        self.assertIn("Extract", prompt_str)
        self.assertIn("user information", prompt_str.lower())


if __name__ == '__main__':
    unittest.main()
