#!/bin/bash
# Test script for LexiLLM

# Set up color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================${NC}"
echo -e "${BLUE}      LexiLLM Test Suite Runner       ${NC}"
echo -e "${BLUE}=======================================${NC}"

# Ensure we have the right environment
if [ -d ".venv" ]; then
    echo -e "${GREEN}Activating virtual environment...${NC}"
    source .venv/bin/activate
else
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python -m venv .venv
    source .venv/bin/activate
    echo -e "${GREEN}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Run all unit tests
echo -e "\n${BLUE}Running unit tests...${NC}"
python -m pytest tests/test_lexillm.py tests/test_modules.py tests/test_schemas.py tests/test_templates.py -v

# Run integration tests if requested
if [ "$1" == "--integration" ]; then
    echo -e "\n${BLUE}Running integration tests...${NC}"
    python tests/test_lexillm.py --integration
fi

echo -e "\n${GREEN}All tests completed!${NC}"
