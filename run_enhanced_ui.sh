#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Change to the project root directory
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Warning: No virtual environment found at .venv"
fi

# Check if OpenAI API key is in .env file
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Creating template."
    echo "OPENAI_API_KEY=your_api_key_here" > .env
    echo "Please edit the .env file to add your OpenAI API key."
fi

# Export the PYTHONPATH to include the src directory
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# Clear the log file
echo "" > lexillm_ui.log

# Create the assets directory if it doesn't exist
mkdir -p assets

# Check if the logo file exists, if not, print a message
if [ ! -f "assets/lexillm_logo.svg" ]; then
    echo "Warning: Logo file not found at assets/lexillm_logo.svg"
    echo "A placeholder logo will be used instead."
fi

# Make script executable
chmod +x "$0"

# Run the Streamlit app
echo "Starting LexiLLM Enhanced UI..."
streamlit run src/ui_streamlit_enhanced.py --server.enableCORS=false --server.enableXsrfProtection=false --browser.serverAddress=localhost