"""
Comprehensive tests for LexiLLM bot
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import unittest
import os
import sys
import time
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import our module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lexillm import LexiLLM
from src.lexillm.schemas import IntentClassifier
from src.lexillm.utils import is_end_request

class TestLexiLLMUnit(unittest.TestCase):
    """Unit tests for the LexiLLM class."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Mock the OpenAI API key
        self.api_key = "mock_api_key"
        
        # Create mock for the LLM
        self.llm_patcher = patch('lexillm.bot.ChatOpenAI')
        self.mock_llm = self.llm_patcher.start()
        
        # Set up a mock model response for the LLM
        mock_model = MagicMock()
        self.mock_llm.return_value = mock_model
        mock_model.with_structured_output.return_value = mock_model
        
        # Initialize bot with mocked components
        with patch('lexillm.bot.load_dotenv'):
            self.bot = LexiLLM(api_key=self.api_key)
    
    def tearDown(self):
        """Clean up after each test."""
        self.llm_patcher.stop()
    
    def test_welcome_message(self):
        """Test that the welcome message is correctly returned."""
        welcome = self.bot.welcome_message()
        self.assertIsInstance(welcome, str)
        self.assertGreater(len(welcome), 0)
    
    def test_intent_classification(self):
        """Test that intent classification works correctly."""
        # Create a mock result
        mock_result = IntentClassifier(intent="LLM_FUNDAMENTALS", confidence=0.95)
        
        # Create a new mock for the intent manager
        self.bot.intent_manager.classify_intent = MagicMock(
            return_value={"intent": "LLM_FUNDAMENTALS", "confidence": 0.95}
        )
        
        # Test classification
        result = self.bot.classify_intent("How do transformers work?")
        self.assertEqual(result["intent"], "LLM_FUNDAMENTALS")
        self.assertAlmostEqual(result["confidence"], 0.95)
    
    def test_process_message(self):
        """Test the main message processing pipeline."""
        # Mock the internal methods
        with patch('lexillm.bot.is_end_request', return_value=False):
            self.bot.intent_manager.classify_intent = MagicMock(
                return_value={"intent": "LLM_FUNDAMENTALS", "confidence": 0.9}
            )
            self.bot.info_collector.extract_user_info_from_message = MagicMock()
            self.bot.info_collector.determine_if_more_info_needed = MagicMock(
                return_value=(True, "technical_level")
            )
            self.bot.info_collector.get_info_collection_message = MagicMock(
                return_value="What's your technical level?"
            )
            
            # Test initial message that needs more info
            response = self.bot.process_message("How do transformers work?")
            self.assertEqual(response, "What's your technical level?")
    
    def test_end_conversation(self):
        """Test conversation ending functionality."""
        # Mock end_conversation to avoid LLM calls
        self.bot.response_generator.end_conversation = MagicMock(
            return_value="Goodbye, thanks for chatting!"
        )
        
        # Mock is_end_request to return True
        with patch('lexillm.bot.is_end_request', return_value=True):
            response = self.bot.process_message("goodbye")
            
            # Check the response is what we expect
            self.assertEqual(response, "Goodbye, thanks for chatting!")
            
            # Verify end_conversation was called
            self.bot.response_generator.end_conversation.assert_called_once()


def run_integration_tests(api_key=None):
    """Run manual integration tests for LexiLLM."""
    try:
        print("=" * 70)
        print("LEXILLM INTEGRATION TESTS")
        print("=" * 70)
        
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print("Warning: No OpenAI API key found. Tests will be limited.")
        
        session_id = f"test_session_{int(time.time())}"
        print(f"Using session ID: {session_id}")
        
        # Initialize the bot
        print("\nInitializing LexiLLM...")
        bot = LexiLLM(api_key=api_key, session_id=session_id)
        
        # Welcome message
        print("\n1. WELCOME MESSAGE TEST")
        print("-" * 50)
        welcome = bot.welcome_message()
        print(f"Bot: {welcome}")
        
        # Basic conversation flow
        print("\n2. BASIC CONVERSATION FLOW")
        print("-" * 50)
        
        # First response to collect user information
        name_response = "My name is Test User"
        print(f"User: {name_response}")
        bot_reply = bot.process_message(name_response)
        print(f"Bot: {bot_reply}")
        
        # Provide technical level
        tech_response = "I'm an intermediate user"
        print(f"User: {tech_response}")
        bot_reply = bot.process_message(tech_response)
        print(f"Bot: {bot_reply}")
        
        # Ask about LLMs
        llm_question = "What are large language models?"
        print(f"User: {llm_question}")
        bot_reply = bot.process_message(llm_question)
        print(f"Bot: {bot_reply}")
        
        # Test a follow-up question
        followup = "How do they work?"
        print(f"User: {followup}")
        bot_reply = bot.process_message(followup)
        print(f"Bot: {bot_reply}")
        
        # Test streaming functionality
        print("\n3. STREAMING RESPONSE TEST")
        print("-" * 50)
        streaming_question = "Explain transformer architecture"
        print(f"User: {streaming_question}")
        print("Bot: ", end="", flush=True)
        
        # Just collect the first chunk to avoid long output
        chunks = []
        for i, chunk in enumerate(bot.process_message_streaming(streaming_question)):
            if i < 3:  # Just show the first 3 chunks
                print(chunk, end="", flush=True)
                chunks.append(chunk)
        print("\n...stream continues...")
        
        # Test error handling
        print("\n4. ERROR HANDLING TEST")
        print("-" * 50)
        unusual_input = "!@#$%^&*()_+"
        print(f"User: {unusual_input}")
        bot_reply = bot.process_message(unusual_input)
        print(f"Bot: {bot_reply}")
        
        # Test conversation reset
        print("\n5. CONVERSATION RESET TEST")
        print("-" * 50)
        bot.reset_conversation()
        print("Conversation reset")
        reset_query = "Tell me about LLMs"
        print(f"User: {reset_query}")
        bot_reply = bot.process_message(reset_query)
        print(f"Bot: {bot_reply}")
        
        # End conversation
        print("\n6. END CONVERSATION TEST")
        print("-" * 50)
        end_message = "goodbye"
        print(f"User: {end_message}")
        bot_reply = bot.process_message(end_message)
        print(f"Bot: {bot_reply}")
        
        print("\nAll integration tests completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Integration test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)
    
    # Run integration tests if requested
    if "--integration" in sys.argv:
        print("\nRunning integration tests...")
        run_integration_tests()
