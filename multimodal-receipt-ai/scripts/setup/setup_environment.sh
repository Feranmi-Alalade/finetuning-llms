#!/bin/bash
# Installation script for multimodal-receipt-ai
# GTX 1080 Ti with CUDA 12.4 driver

set -e

echo "==================================="
echo "Multimodal Receipt AI Installation"
echo "==================================="
echo ""

# Detect system
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
    INSTALL_TYPE="cuda"
else
    echo "⚠ No NVIDIA GPU detected, installing CPU version"
    INSTALL_TYPE="cpu"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Uninstall old PyTorch versions
echo "Removing existing PyTorch installations..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
echo ""

# Install PyTorch based on system
if [ "$INSTALL_TYPE" = "cuda" ]; then
    echo "Installing PyTorch with CUDA 12.1 support..."
    echo "(Compatible with your CUDA 12.4 driver)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo ""

# Install project dependencies
echo "Installing project dependencies..."
pip install -e .
echo ""

# Install development dependencies (optional)
read -p "Install development dependencies? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -e ".[dev]"
fi
echo ""

# Verify installation
echo "Verifying PyTorch installation..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("Running on CPU")
EOF
echo ""

echo "==================================="
echo "Installation complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Create a .env file with your API keys (if using Gemini)"
echo "2. Run: python pipelines/data/generate_teacher_labels.py --teacher qwen3-vl-8b"
echo ""