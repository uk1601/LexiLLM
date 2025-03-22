"""
Unit tests for LexiLLM schemas
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import unittest
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from lexillm.schemas import IntentClassifier, UserInfo

class TestSchemas(unittest.TestCase):
    """Test cases for schema validation."""
    
    def test_intent_classifier_schema(self):
        """Test that the IntentClassifier schema validates correctly."""
        # Test valid data
        valid_data = IntentClassifier(intent="LLM_FUNDAMENTALS", confidence=0.95)
        self.assertEqual(valid_data.intent, "LLM_FUNDAMENTALS")
        self.assertEqual(valid_data.confidence, 0.95)
        
        # Test invalid intent
        with self.assertRaises(ValidationError):
            IntentClassifier(intent="INVALID_INTENT", confidence=0.9)
        
        # Test confidence bounds
        with self.assertRaises(ValidationError):
            IntentClassifier(intent="LLM_FUNDAMENTALS", confidence=1.5)
        
        with self.assertRaises(ValidationError):
            IntentClassifier(intent="LLM_FUNDAMENTALS", confidence=-0.1)
    
    def test_user_info_schema(self):
        """Test that the UserInfo schema validates correctly."""
        # Test with all fields
        full_data = UserInfo(
            name="John Doe",
            technical_level="intermediate",
            project_stage="development",
            comparison_criterion="accuracy",
            interest_area="research"
        )
        self.assertEqual(full_data.name, "John Doe")
        self.assertEqual(full_data.technical_level, "intermediate")
        self.assertEqual(full_data.project_stage, "development")
        self.assertEqual(full_data.comparison_criterion, "accuracy")
        self.assertEqual(full_data.interest_area, "research")
        
        # Test with partial fields
        partial_data = UserInfo(name="Jane Doe")
        self.assertEqual(partial_data.name, "Jane Doe")
        self.assertIsNone(partial_data.technical_level)
        self.assertIsNone(partial_data.project_stage)
        self.assertIsNone(partial_data.comparison_criterion)
        self.assertIsNone(partial_data.interest_area)
        
        # Test with empty constructor
        empty_data = UserInfo()
        self.assertIsNone(empty_data.name)
        self.assertIsNone(empty_data.technical_level)
        

if __name__ == '__main__':
    unittest.main()
