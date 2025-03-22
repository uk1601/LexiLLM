"""
Main entry point for LexiLLM Bot
Author: Uday Kiran Dasari
Northeastern University - Prompt Engineering
Spring 2025
"""

import os
from dotenv import load_dotenv
from lexillm import LexiLLM

def interactive_demo():
    """Run an interactive demo of the LexiLLM bot."""
    # Load environment variables
    load_dotenv()
    
    # Initialize the bot with OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Print a helpful message if API key is missing
    if not api_key:
        print("\nError: No OpenAI API key found!")
        print("Please add your API key to the .env file in the following format:")
        print("OPENAI_API_KEY=your_api_key_here\n")
        return
        
    bot = LexiLLM(api_key=api_key)
    
    # Start conversation
    print("LexiLLM:", bot.welcome_message())
    
    while bot.is_conversation_active():
        # Get user input
        user_input = input("You: ")
        
        # Process the message
        response = bot.process_message(user_input)
        
        # Display the response
        print("LexiLLM:", response)
    
    # Save the user profile before exiting
    bot.save_user_profile()
    print("\nThanks for using LexiLLM! Goodbye!")


def streaming_demo():
    """Run a demo of the LexiLLM bot with streaming responses."""
    # Load environment variables
    load_dotenv()
    
    # Initialize the bot with OpenAI API key and streaming enabled
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Print a helpful message if API key is missing
    if not api_key:
        print("\nError: No OpenAI API key found!")
        print("Please add your API key to the .env file in the following format:")
        print("OPENAI_API_KEY=your_api_key_here\n")
        return
        
    bot = LexiLLM(api_key=api_key, streaming=True)
    
    # Start conversation
    print("LexiLLM:", bot.welcome_message())
    
    while bot.is_conversation_active():
        # Get user input
        user_input = input("You: ")
        
        # Process the message with streaming
        print("LexiLLM: ", end="", flush=True)
        for chunk in bot.process_message_streaming(user_input):
            print(chunk, end="", flush=True)
        print()  # Add a newline at the end
    
    # Save the user profile before exiting
    bot.save_user_profile()
    print("\nThanks for using LexiLLM! Goodbye!")


if __name__ == "__main__":
    # Choose which demo to run
    use_streaming = os.environ.get("USE_STREAMING", "false").lower() in ("true", "1", "t", "yes")
    
    if use_streaming:
        streaming_demo()
    else:
        interactive_demo()
