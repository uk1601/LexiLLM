# LexiLLM: Large Language Model Assistant

## Project Overview
LexiLLM is a specialized chatbot designed to assist users with understanding and implementing Large Language Models (LLMs). This project was developed as part of the Prompt Engineering graduate course at Northeastern University.

## Features
- **LLM Fundamentals**: Explanations of core concepts and architectures
- **Implementation Guidance**: Practical advice for building LLM applications
- **Model Comparison**: Objective comparisons between different LLM options
- **News & Trends**: Updates on recent developments in the field
- **Streaming UI**: Web interface with real-time response generation
- **Multi-platform Support**: Command-line and Streamlit web interface
- **Modular Architecture**: Clearly separated components for maintainability
- **Comprehensive User Profiles**: Advanced user information management with memory
- **Natural Onboarding**: Progressive collection of user preferences
- **Personalized Responses**: Tailored content based on user profile

## Technical Implementation
LexiLLM implements all required components from the assignment (see [complete technical documentation](documentation/complete_documentation.md) for details):
- **Welcome Message**: Clear introduction with purpose statement
- **Intent Recognition**: Classification for four distinct LLM-related topics
- **User Input Collection**: Comprehensive profile management with technical level assessment
- **Conditional Logic**: Conversation branching based on user expertise and needs
- **Personalized Text Responses**: Content tailored to user profiles
- **Professional Conversation Closure**: Polite ending with summary
- **Robust Error Handling**: Graceful recovery from errors with helpful messages
- **Helpful Fallback Messages**: Guidance when intent can't be determined

## Modular Architecture
LexiLLM features a modular architecture with clear separation of concerns:

1. **LexiLLM Class (bot.py)**: Central orchestrator that coordinates modules
2. **Intent Manager**: Handles intent classification and follow-up detection
3. **Info Collector**: Manages user information collection and extraction
4. **Response Generator**: Creates personalized responses
5. **Conversation Manager**: Maintains conversation state and history
6. **User Profile System**: Handles user data with confidence scoring

This architecture provides several key benefits:
- **Improved Maintainability**: Each module has a clear responsibility
- **Enhanced Testability**: Modules can be tested independently
- **Better Code Organization**: Logical separation of concerns
- **Reduced Complexity**: Main class focuses on orchestration
- **Easier Extension**: New features can be added to specific modules

## User Profile System

LexiLLM's user profile system:

- **Collects information naturally** throughout the conversation
- **Maintains persistent memory** across sessions
- **Provides personalized responses** based on user's technical level and interests
- **Uses intelligent onboarding** to collect essential information
- **Adapts to expertise levels** for more relevant explanations
- **Learns from implicit information** shared during conversations
- **Periodically updates user attributes** without disrupting conversation flow

## Directory Structure
```
LexiLLM/
├── README.md                      # Project overview
├── documentation/                 # Detailed documentation
│   └── complete_documentation.md  # Comprehensive technical documentation
├── assets/                        # Static assets
├── src/                           # Source code
│   ├── lexillm/                   # Main package
│   │   ├── __init__.py            # Package initialization
│   │   ├── bot.py                 # Core bot implementation (orchestrator)
│   │   ├── config.py              # Configuration settings
│   │   ├── exceptions.py          # Custom exception classes
│   │   ├── logger.py              # Logging configuration
│   │   ├── schemas.py             # Data schemas using Pydantic
│   │   ├── templates.py           # Prompt templates
│   │   ├── user_profile.py        # User profile management
│   │   ├── utils.py               # Utility functions
│   │   └── modules/               # Modular components
│   │       ├── __init__.py        # Module initialization
│   │       ├── conversation_manager.py # Conversation state and history
│   │       ├── conversation_state.py   # State enumeration and utilities
│   │       ├── info_collector.py  # User information collection
│   │       ├── intent_manager.py  # Intent classification
│   │       └── response_generator.py # Response generation
│   ├── ui/                        # UI Components
│   │   ├── __init__.py            # UI package initialization
│   │   ├── components.py          # UI component definitions
│   │   ├── image_utils.py         # Image handling utilities
│   │   ├── state.py               # Streamlit state management
│   │   └── styling.py             # CSS styling for UI
│   ├── main.py                    # CLI entry point
│   └── ui_streamlit_enhanced.py   # Streamlit UI implementation
├── user_profiles/                 # Storage for user profiles
└── tests/                         # Unit and integration tests
```

## Installation and Usage

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Bot

### Web Interface (Streamlit)

The Streamlit interface provides an intuitive, web-based experience with streaming response generation:

```bash
# Run the UI
./run_ui.sh
```

### Command Line Interface

You can also run the bot in a terminal:

```bash
./run.sh
```

### Developer Mode

For development work, you can install the package in editable mode:

```bash
pip install -e .
```

## Testing

Run the comprehensive test suite using the provided script:

```bash
./run_tests.sh
```

The test suite includes:
- Unit tests for individual modules
- Integration tests for module interactions
- End-to-end conversation flow tests

## Profile Management System

The core of LexiLLM's personalization capabilities is its user profile system:

- **ProfileAttribute**: Single piece of user information with confidence score
- **UserProfile**: Comprehensive collection of user attributes
- **UserProfileManager**: Handles persistence and updates

Profiles are automatically saved between sessions using unique session IDs, enabling continuous improvement of the user experience as the bot learns more about each user.

## Author
Uday Kiran Dasari  
Northeastern University  
Prompt Engineering - Spring 2025