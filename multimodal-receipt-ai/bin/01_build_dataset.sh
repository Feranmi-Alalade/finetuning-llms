#!/bin/bash
# 01_build_dataset.sh
# Pipeline A: Ingest & Augment
# Bronze -> Silver (Base + Stability)

set -e

echo "==========================================="
echo "Pipeline A: Data Ingestion & Augmentation"
echo "==========================================="

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The project module root is one level up from the bin directory
MODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$MODULE_ROOT"

# Ensure PYTHONPATH includes project root
export PYTHONPATH=$PYTHONPATH:.

# Run the ingestion pipeline
uv run python pipelines/data/build_datasets.py \
    --input "../artifacts/receipts/bronze" \
    --silver-root "../artifacts/receipts/silver" \
    --gold-root "../artifacts/receipts/gold" \
    "$@"

echo "==========================================="
echo "Ingestion Complete."
echo "Silver (Images): artifacts/receipts/silver"
echo "Gold (Synthetic Labels): artifacts/receipts/gold/stability"
echo "==========================================="
