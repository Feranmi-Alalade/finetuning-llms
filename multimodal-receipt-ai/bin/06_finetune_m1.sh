#!/bin/bash

# Dedicated script to run Fine-tuning pipeline using uv

echo "===================================================="
echo "    Qwen3-VL Fine-Tuning Pipeline (TRL)"
echo "===================================================="
echo ""

# Function to select environment
select_env() {
    echo "--- Select Environment ---"
    echo "1) Development (Local/Quick) [Default]"
    echo "2) Production (Full Training)"
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

PIPELINE_DIR="pipelines/training"

echo "Running: Fine-tune Qwen3-VL 8B..."
# Config is loaded via fine_tuning.yaml inside the python script
uv run python "$PIPELINE_DIR/fine_tune_qwen.py" --env "$ENV" "$@"

echo ""
echo "===================================================="
echo "    Fine-Tuning Complete"
echo "===================================================="
