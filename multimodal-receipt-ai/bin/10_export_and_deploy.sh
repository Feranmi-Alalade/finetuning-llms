#!/bin/bash

# ==============================================================================
# 10_export_and_deploy.sh
# ==============================================================================
# Wrapper for Master Export Pipeline (Python)
# ==============================================================================

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Default arguments
MODEL_NAME=""
ADAPTER_PATH=""
ENV="dev"
QUANT="q4_k_m"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --adapter) ADAPTER_PATH="$2"; shift ;;
        --env) ENV="$2"; shift ;;
        --quant) QUANT="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# 1. Selection: Find latest trained adapters if not provided
if [ -z "$ADAPTER_PATH" ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Model Export & Deployment Pipeline"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Get the directory where the script is located
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # The project module root is one level up from the bin directory
    PROJECT_MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$PROJECT_MODULE_ROOT" || exit 1
    
    OUTPUTS_DIR="outputs"

    if [ ! -d "$OUTPUTS_DIR" ]; then
        print_error "Error: outputs/ directory not found in $(pwd)"
        exit 1
    fi

    ADAPTERS=($(find "$OUTPUTS_DIR" -type d -name "adapter" 2>/dev/null))

    if [ ${#ADAPTERS[@]} -eq 0 ]; then
        print_error "No trained adapters found in $OUTPUTS_DIR"
        exit 1
    fi

    echo "Select an adapter to export:"
    for i in "${!ADAPTERS[@]}"; do
        echo "  $((i+1))) ${ADAPTERS[$i]}"
    done

    read -p "Selection [1-${#ADAPTERS[@]}]: " choice
    ADAPTER_PATH=${ADAPTERS[$((choice-1))]}
fi

if [ -z "$ADAPTER_PATH" ]; then
    print_error "Invalid selection or no adapter provided."
    exit 1
fi

if [ -z "$MODEL_NAME" ]; then
    read -p "Enter Target Model Name (m0, m1, unified): " MODEL_NAME
fi

# 2. Dependencies (Ensures cross-platform llama.cpp is ready)
print_info "Ensuring dependencies are ready..."
task deps:install:llama || { print_error "Failed to build llama.cpp"; exit 1; }

# 3. Execute Master Export Pipeline
print_info "Starting Export Pipeline (Merge -> HF -> GGUF -> Ollama)..."
uv run python pipelines/deploy/export_model.py \
    --adapter "$ADAPTER_PATH" \
    --model-name "$MODEL_NAME" \
    --env "$ENV" \
    --quantization "$QUANT" || { print_error "Pipeline failed"; exit 1; }

print_success "Export and Deployment process finished!"
