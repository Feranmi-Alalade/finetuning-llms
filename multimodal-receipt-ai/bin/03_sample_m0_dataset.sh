#!/bin/bash
set -e

# 03_sample_m0_dataset.sh
# Trigger the Python sampling pipeline for M0 OCR dataset

echo "===================================================="
echo "    M0 Dataset Sampling Pipeline"
echo "===================================================="

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The project module root is one level up from the bin directory
MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PIPELINE_SCRIPT="pipelines/data/sample_m0_dataset.py"

# Function to show usage
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --dry-run    Run without copying files"
    echo "  --clean      Clean output directory before running"
    echo "  --seed N     Set random seed (default: 42)"
    echo "  -h, --help   Show this help message"
}

# Parse arguments
PYTHON_ARGS=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) PYTHON_ARGS="$PYTHON_ARGS --dry-run" ;;
        --clean) PYTHON_ARGS="$PYTHON_ARGS --clean" ;;
        --seed) PYTHON_ARGS="$PYTHON_ARGS --seed $2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown parameter: $1"; usage; exit 1 ;;
    esac
    shift
done

cd "$MODULE_ROOT" || exit
export PYTHONPATH=$PYTHONPATH:.

echo "Running sampling script: $PIPELINE_SCRIPT"
echo "Args: $PYTHON_ARGS"
echo "----------------------------------------------------"

# Run with uv
uv run python "$PIPELINE_SCRIPT" $PYTHON_ARGS

echo "===================================================="
echo "    Sampling Complete"
echo "===================================================="
