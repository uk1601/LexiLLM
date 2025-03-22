#!/bin/bash
# Run script for LexiLLM

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run the bot
python src/main.py
