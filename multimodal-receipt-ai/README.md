# Multimodal Receipt AI

A high-performance teacher-student distillation pipeline for FMCG receipt understanding.

This project produces high-precision structured training data by distilling knowledge from state-of-the-art Vision-Language Models (Gemini 2.0/Qwen3) into production-ready students.

## 🎯 Project Goals

*   **High-Precision OCR**: Extract exact text and amounts from diverse receipt formats.
*   **Semantic Grounding**: Map raw OCR text to standardized Fast-Moving Consumer Goods (FMCG) categories.
*   **Production Robustness**: Handle blurs, low contrast, rotation, and partial receipts via stability training.
*   **Efficient Distillation**: Generate thousands of high-quality labels using robust batch processing and validation.

## 🚀 Architecture

The pipeline follows a tiered data architecture:

1.  **Bronze (Raw)**: Original PDFs and images.
2.  **Silver (Normalized)**: OCR-optimized images (enhanced via OpenCV) and synthetic stability variants.
3.  **Gold (Labeled)**: Validated teacher-generated JSON labels and per-image metadata.

## 🛠 Setup

### Prerequisites
*   [UV](https://github.com/astral-sh/uv) (Extremely fast Python package manager)
*   AWS CLI (configured for S3 synchronization)
*   Gemini API Key (for cloud teacher)

### Installation
```bash
# Clone the repository
cd multimodal-receipt-ai

# Setup environment and install dependencies
bash scripts/setup_environment.sh

# Configure environment variables
cat <<EOF > .env
GOOGLE_API_KEY=your_key_here
S3_BUCKET=your-bucket-name
AWS_REGION=eu-central-1
EOF
```

## 📈 Data Pipeline

### 1. Ingestion & Enhancement
Converts Bronze sources into OCR-ready Silver images.
```bash
# Process raw data with automatic OCR enhancement
bash 01_build_dataset.sh
```
*   **Features**: OpenCV-based CLAHE, denoising, and adaptive thresholding for maximum extraction accuracy.
*   **Stability**: Generates blurry, low-contrast, and empty variants to improve student robustness.

### 2. Teacher Label Generation
Uses Gemini or Qwen to generate structured JSON labels.
```bash
# Generate labels using the optimized teacher pipeline
bash 02_run_teacher_gen.sh
```
*   **Smart Batching**: Intelligent token-aware batch packing with automatic individual retries on failure.
*   **Content-Aware Caching**: Image hashing prevents redundant API calls for previously processed receipts.
*   **Built-in Validation**: Every label is validated using Pydantic against the production schema.

### 3. Validation & Cleanup
Ensures 100% data integrity before training.
```bash
# Identify and move malformed labels to marked_for_deletion/
uv run scripts/validate_teacher_labels.py
```
*   **Field-Level Debugging**: Detailed MongoDB-style field paths in `logs/validation_errors.log`.

## ☁️ Cloud Persistence

The entire `artifacts` tree is automatically mirrored to AWS S3 using the high-performance `aws s3 sync` CLI.
*   **Perfect Fidelity**: Deletions (e.g., from cleanup scripts) are instantly reflected in the cloud.
*   **Efficiency**: Multi-threaded sync ensures fast uploads of thousands of images.

