#!/bin/bash

# Dedicated script to run Teacher Label Generation pipeline using uv

echo "===================================================="
echo "    Teacher Label Generation Pipeline (UV)"
echo "===================================================="
echo ""

# Function to select environment
select_env() {
    echo "--- Select Environment ---"
    echo "1) Development (Local/8B) [Default]"
    echo "2) Production (Cloud/30B)"
    echo ""
    read -p "Environment Choice [1-2, Enter for default]: " env_choice
    
    case $env_choice in
        2) export ENV="prod" ;;
        *) export ENV="dev" ;;
    esac
    echo "Active Environment: $ENV"
    echo ""
}

# Select Env first as it drives configuration
select_env

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The project module root is one level up from the bin directory
MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MODULE_ROOT" || exit

PIPELINE_DIR="pipelines/data"

echo "Running: Generate Teacher Labels..."
# Model is auto-selected based on ENV in config.py
# Using explicit paths to silver base images and shared gold output
uv run python "$PIPELINE_DIR/generate_teacher_labels.py" \
    --image-dir "../artifacts/receipts/silver/base/images" \
    --output "../artifacts/receipts/platinum/teacher/teacher_labels.jsonl" \
    "$@"

echo ""
echo "===================================================="
echo "    Label Generation Complete"
echo "===================================================="
