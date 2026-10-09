#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Interactive Receipt Pretty Printer...${NC}"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# Run the python script
# We set PYTHONPATH to PROJECT_ROOT to ensure imports work correctly
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if virtualenv exists and use it if it does
if [ -d "$PROJECT_ROOT/.venv" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

python3 "$SCRIPT_DIR/pretty_print_jsonl.py"
