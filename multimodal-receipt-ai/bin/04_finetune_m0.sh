#!/bin/bash

# ==============================================================================
# M0 OCR Fine-Tuning Pipeline - Interactive Launcher
# ==============================================================================
# This script provides an interactive menu to configure and run fine-tuning
# with appropriate settings for different training scenarios.
#
# Usage:
#   ./04_finetune_m0.sh [additional args passed to Python script]
#
# Modes:
#   - Dev (Quick Test): Fast sanity check on local machine
#   - Dev (Full): Full training on local machine with custom losses
#   - Prod (LoRA): Production training with LoRA
#   - Prod (QLoRA): Production training with 4-bit quantization
# ==============================================================================

echo "===================================================="
echo "    Qwen3-VL M0 OCR Fine-Tuning Pipeline"
echo "===================================================="
echo ""

# Color codes for better UX
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Mac Memory Optimization (Disable private pool limit)
if [[ "$(uname)" == "Darwin" ]]; then
    export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
fi

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ==============================================================================
# Training Configuration Selection
# ==============================================================================

select_training_config() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Select Training Configuration"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  ${GREEN}Local Development${NC}"
    echo "  1) Quick Sanity Check (OCR only, ~2 min)"
    echo "     • Environment: dev"
    echo "     • Mode: lora"
    echo "     • Custom Losses: OFF"
    echo "     • Purpose: Verify pipeline works correctly"
    echo ""
    echo "  2) Full Dev Training (All objectives, ~10-30 min)"
    echo "     • Environment: dev"
    echo "     • Mode: lora"
    echo "     • Custom Losses: ON"
    echo "     • Purpose: Test full training loop locally"
    echo ""
    echo "  ${YELLOW}Production Training (GPU Cluster)${NC}"
    echo "  3) Production LoRA (Standard, ~2-4 hours)"
    echo "     • Environment: prod"
    echo "     • Mode: lora"
    echo "     • Custom Losses: ON"
    echo "     • Purpose: Full training with standard precision"
    echo ""
    echo "  4) Production QLoRA (Quantized, ~3-5 hours)"
    echo "     • Environment: prod"
    echo "     • Mode: qlora"
    echo "     • Custom Losses: ON"
    echo "     • Purpose: Memory-efficient training with 4-bit quantization"
    echo ""
    echo "  5) Custom Configuration"
    echo "     • Manually specify all options"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    read -p "Select configuration [1-5, default=1]: " config_choice
    
    case $config_choice in
        2)
            print_info "Selected: Full Dev Training"
            export ENV="dev"
            export MODE="lora"
            export CUSTOM_LOSSES="enabled"
            export CONFIG_NAME="dev_full"
            ;;
        3)
            print_info "Selected: Production LoRA"
            export ENV="prod"
            export MODE="lora"
            export CUSTOM_LOSSES="enabled"
            export CONFIG_NAME="prod_lora"
            ;;
        4)
            print_info "Selected: Production QLoRA"
            export ENV="prod"
            export MODE="qlora"
            export CUSTOM_LOSSES="enabled"
            export CONFIG_NAME="prod_qlora"
            ;;
        5)
            print_info "Selected: Custom Configuration"
            custom_config
            export CONFIG_NAME="custom"
            ;;
        *)
            print_info "Selected: Quick Sanity Check (Default)"
            export ENV="dev"
            export MODE="lora"
            export CUSTOM_LOSSES="disabled"
            export CONFIG_NAME="dev_sanity"
            ;;
    esac
    
    echo ""
}

# Custom configuration wizard
custom_config() {
    echo ""
    echo "━━━ Custom Configuration ━━━"
    
    # Environment
    echo ""
    echo "Select Environment:"
    echo "  1) dev  - Local CPU/MPS (Mac)"
    echo "  2) prod - GPU Cluster"
    read -p "Environment [1-2]: " env_choice
    case $env_choice in
        2) export ENV="prod" ;;
        *) export ENV="dev" ;;
    esac
    print_success "Environment: $ENV"
    
    # Mode
    echo ""
    echo "Select Training Mode:"
    echo "  1) lora  - Standard LoRA (recommended)"
    echo "  2) qlora - 4-bit Quantized LoRA (memory efficient)"
    read -p "Mode [1-2]: " mode_choice
    case $mode_choice in
        2) export MODE="qlora" ;;
        *) export MODE="lora" ;;
    esac
    print_success "Mode: $MODE"
    
    # Custom Losses
    echo ""
    echo "Enable Custom Loss Functions?"
    echo "  1) Yes - Full M0 objectives (OCR + Field + Rejection)"
    echo "  2) No  - OCR only (for sanity checking)"
    read -p "Custom Losses [1-2]: " losses_choice
    case $losses_choice in
        2) export CUSTOM_LOSSES="disabled" ;;
        *) export CUSTOM_LOSSES="enabled" ;;
    esac
    print_success "Custom Losses: $CUSTOM_LOSSES"
}

# ==============================================================================
# Display Configuration Summary
# ==============================================================================

display_config_summary() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Training Configuration Summary"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "  Configuration: ${YELLOW}${CONFIG_NAME}${NC}"
    echo "  Environment:   ${BLUE}${ENV}${NC}"
    echo "  Mode:          ${BLUE}${MODE}${NC}"
    echo "  Custom Losses: ${BLUE}${CUSTOM_LOSSES}${NC}"
    echo ""
    
    # Loss weights display
    if [ "$CUSTOM_LOSSES" = "enabled" ]; then
        echo "  Loss Function Weights:"
        echo "    • OCR Token Loss:      0.75"
        echo "    • Field Presence Loss: 0.15"
        echo "    • Rejection Loss:      0.10"
        echo ""
        echo "  Auxiliary Tasks:"
        echo "    • Field Detection: 5 fields (merchant, date, total, tax, items)"
        echo "    • Image Rejection: Empty, non-receipt, severely degraded"
    else
        echo "  Loss Function Weights:"
        echo "    • OCR Token Loss:      1.00"
        echo ""
        echo "  ${YELLOW}Note: Running in sanity check mode (OCR only)${NC}"
    fi
    
    echo ""
    
    # Expected duration
    case $CONFIG_NAME in
        dev_sanity)
            echo "  Expected Duration: ~2 minutes"
            echo "  Training Steps:    20 steps (quick validation)"
            ;;
        dev_full)
            echo "  Expected Duration: ~10-30 minutes"
            echo "  Training Steps:    Per config (full dev run)"
            ;;
        prod_lora)
            echo "  Expected Duration: ~2-4 hours"
            echo "  Training Steps:    Per config (full training)"
            ;;
        prod_qlora)
            echo "  Expected Duration: ~3-5 hours"
            echo "  Training Steps:    Per config (quantized training)"
            ;;
    esac
    
    echo ""
    echo "  W&B Experiment: m0_ocr_${MODE}_$(date +%Y%m%d_%H%M%S)_${ENV}"
    if [ "$CUSTOM_LOSSES" = "disabled" ]; then
        echo "                  (suffixed with: _no_custom_loss)"
    fi
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# ==============================================================================
# Validation
# ==============================================================================

validate_environment() {
    print_info "Validating environment..."
    
    # Determine module root based on script location
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    cd "$PROJECT_MODULE_ROOT" || exit 1

    # Check if Python script exists
    if [ ! -f "pipelines/training/finetune_m0.py" ]; then
        print_error "Error: finetune_m0.py not found in $(pwd)/pipelines/training/"
        exit 1
    fi
    
    # Warn if prod mode on local
    if [ "$ENV" = "prod" ] && [ ! -command -v nvidia-smi &> /dev/null ]; then
        print_warning "Warning: prod mode selected but no GPU detected!"
        print_warning "This may fail or run very slowly."
        echo ""
        read -p "Continue anyway? [y/N]: " continue_choice
        if [[ ! $continue_choice =~ ^[Yy]$ ]]; then
            print_info "Aborted."
            exit 0
        fi
    fi
    
    print_success "Environment validation passed"
    echo ""
}

# ==============================================================================
# Confirmation
# ==============================================================================

confirm_start() {
    echo ""
    read -p "Start training with this configuration? [Y/n]: " confirm
    echo ""
    
    if [[ $confirm =~ ^[Nn]$ ]]; then
        print_info "Training cancelled."
        exit 0
    fi
}

# ==============================================================================
# Main Execution
# ==============================================================================
main() {
    # Parse arguments early to check for non-interactive mode
    INTERACTIVE=true
    for arg in "$@"; do
        if [[ $arg == "--env" ]] || [[ $arg == "--mode" ]]; then
            INTERACTIVE=false
            break
        fi
    done

    if [ "$INTERACTIVE" = true ]; then
        # Select configuration
        select_training_config
        
        # Display summary
        display_config_summary
        
        # Validate
        validate_environment
        
        # Confirm
        confirm_start
    else
        # In non-interactive mode, we assume the user knows what they are doing
        print_info "Running in non-interactive mode based on arguments"
        # Extract ENV and MODE from args if possible, or set defaults
        # This is a bit simplified; the python script will handle the actual parsing
        # but the script needs some variables for the CMD below.
        # We'll let the python script defaults take over or use simple detection
        
        # Prepare paths - use script location to find project root
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PROJECT_MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
        PIPELINE_DIR="pipelines/training"
        
        # Move to project subdirectory
        cd "$PROJECT_MODULE_ROOT" || exit 1
        
        CMD="uv run python $PIPELINE_DIR/finetune_m0.py $@"
        
        print_info "Executing: $CMD"
        eval $CMD
        EXIT_CODE=$?
        
        if [ $EXIT_CODE -eq 0 ]; then
            print_success "Training completed successfully!"
        else
            print_error "Training failed with exit code: $EXIT_CODE"
        fi
        exit $EXIT_CODE
    fi
    
    # Prepare paths - use script location to find project root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    PIPELINE_DIR="pipelines/training"
    
    # Move to project subdirectory
    cd "$PROJECT_MODULE_ROOT" || exit 1
    
    # Build command
    CMD="uv run python $PIPELINE_DIR/finetune_m0.py --env $ENV --mode $MODE"
    
    # Add custom loss flag if disabled
    if [ "$CUSTOM_LOSSES" = "disabled" ]; then
        CMD="$CMD --no-custom-losses"
    fi
    
    # Add any additional arguments passed to script
    CMD="$CMD $@"
    
    # Print command for transparency
    print_info "Executing: $CMD"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    # Execute
    eval $CMD
    EXIT_CODE=$?
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check result
    if [ $EXIT_CODE -eq 0 ]; then
        print_success "Training completed successfully!"
        echo ""
        print_info "Next steps:"
        echo "  • Check W&B dashboard for metrics"
        echo "  • Review adapter in outputs directory"
        echo "  • Run evaluation pipeline if configured"
    else
        print_error "Training failed with exit code: $EXIT_CODE"
        echo ""
        print_info "Troubleshooting:"
        echo "  • Check logs for error messages"
        echo "  • Verify data directory exists and is populated"
        echo "  • Try running with --no-custom-losses for sanity check"
    fi
    
    echo ""
    echo "===================================================="
}

# Run main function
main "$@"
