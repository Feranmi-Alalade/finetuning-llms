# Getting Started - Complete Walkthrough

## What This Project Does

Train a small, efficient receipt OCR model by:
1. Using a strong teacher model (Gemini or Qwen-7B) to label your images
2. Training a smaller student model (Qwen-2B/7B) with LoRA
3. Enforcing strict evaluation contracts before deployment

**Result**: High-quality OCR model trained on ~100-1000 images in hours, not days.

---

## Prerequisites

- Python 3.10+
- (Optional) GPU for faster training
- (Optional) Google Gemini API key for teacher labels
- (Optional) AWS account for SageMaker training

---

## Installation (5 minutes)

```bash
# Clone repo
git clone <your-repo>
cd multimodal-receipt-ai

# Run setup script
bash scripts/setup_environment.sh

# Set API keys
export WANDB_API_KEY="your_wandb_key"           # Get from wandb.ai
export GOOGLE_API_KEY="your_gemini_key"         # Get from ai.google.dev (optional)
```

---

## Quick Start (30 minutes on CPU)

### 1. Add Your Images (2 minutes)

```bash
# Add 10-20 receipt images for testing
mkdir -p data/sources/receipts_raw
cp /path/to/your/receipts/*.jpg data/sources/receipts_raw/
```

### 2. Generate Training Labels (10 minutes)

**Option A: Use Gemini (best quality, costs ~$0.01)**

```bash
export GOOGLE_API_KEY="your_key"

python pipelines/data/generate_teacher_labels.py \
  --teacher gemini \
  --prompt-type ocr \
  --max-images 20 \
  --output data/artifacts/shared/ocr_base_labels.jsonl
```

**Option B: Use local Qwen-2B (free, slower on CPU)**

```bash
python pipelines/data/generate_teacher_labels.py \
  --teacher qwen-2b \
  --prompt-type ocr \
  --max-images 10 \
  --output data/artifacts/shared/ocr_base_labels.jsonl
```

### 3. Generate Stability Data (2 minutes)

```bash
python pipelines/data/generate_stability_data.py \
  --num-empty 5 \
  --num-blurry 5 \
  --num-non-receipt 5
```

### 4. Build Dataset (1 minute)

```bash
python pipelines/data/build_datasets.py --lineage m0
```

This creates:
- `data/artifacts/model_scoped/m0_ocr/train.jsonl`
- `data/artifacts/model_scoped/m0_ocr/val.jsonl`

### 5. Train Model (15 minutes on CPU)

```bash
# Edit config for CPU testing
# lineages/m0_ocr/training.yaml:
#   base_model: "Qwen/Qwen2-VL-2B-Instruct"
#   batch_size: 1
#   max_epochs: 1

python pipelines/training/train_lineage.py --lineage m0_ocr
```

### 6. Evaluate (2 minutes)

```bash
python pipelines/evaluation/eval_lineage.py --lineage m0_ocr
```

---

## Production Workflow (GPU/SageMaker)

### Phase 1: Local GPU Development

```bash
# 1. Generate labels with better teacher
export GOOGLE_API_KEY="your_key"
python pipelines/data/generate_teacher_labels.py \
  --teacher gemini \
  --max-images 1000

# 2. Build full dataset
python pipelines/data/build_datasets.py --lineage m0

# 3. Update config for GPU
# lineages/m0_ocr/training.yaml:
#   base_model: "Qwen/Qwen2-VL-7B-Instruct"
#   batch_size: 4
#   max_epochs: 3
#   load_in_8bit: true  # If GPU < 24GB

# 4. Train
USE_GPU=true python pipelines/training/train_lineage.py --lineage m0_ocr

# 5. Evaluate
python pipelines/evaluation/eval_lineage.py --lineage m0_ocr
```

### Phase 2: SageMaker Multi-GPU

```python
# Edit notebooks/sagemaker_training.py
INSTANCE_TYPE = "ml.g5.12xlarge"  # 4x A10G
WANDB_API_KEY = "your_key"
LINEAGE = "m0_ocr"

# Upload data to S3
# sagemaker_session.upload_data(...)

# Launch
python notebooks/sagemaker_training.py
```

---

## Understanding the Output

### After Training

```
lineages/m0_ocr/checkpoints/
├── epoch00-step100-val_loss0.2345.ckpt
├── epoch01-step200-val_loss0.1234.ckpt  ← Best model
├── epoch02-step300-val_loss0.1456.ckpt
└── last.ckpt

lineages/m0_ocr/checkpoints/lora_final/
├── adapter_config.json
└── adapter_model.safetensors  ← LoRA weights (~100MB)
```

### After Evaluation

```
lineages/m0_ocr/eval_results.json:
{
  "lineage": "m0_ocr",
  "metrics": {
    "cer": 0.08,         # 8% character error
    "wer": 0.15,         # 15% word error
    "exact_match": 0.35  # 35% perfect matches
  },
  "contract_passed": true
}
```

### After Promotion

```
lineages/m0_ocr/PROMOTED  ← Model approved for deployment
```

or

```
lineages/m0_ocr/BLOCKED  ← Model failed contract
```

---

## Common Issues & Solutions

### Issue: "GOOGLE_API_KEY not set"

```bash
# Get free API key from https://aistudio.google.com/apikey
export GOOGLE_API_KEY="your_key_here"
```

### Issue: Out of memory during training

```yaml
# Solution 1: Reduce batch size
batch_size: 1
gradient_accumulation_steps: 8

# Solution 2: Enable quantization
load_in_8bit: true

# Solution 3: Use smaller model
base_model: "Qwen/Qwen2-VL-2B-Instruct"
```

### Issue: Training too slow on CPU

```bash
# Use Gemini for teacher (fast API)
python pipelines/data/generate_teacher_labels.py --teacher gemini

# Train on GPU or SageMaker instead
USE_GPU=true python pipelines/training/train_lineage.py --lineage m0_ocr
```

### Issue: Evaluation contract fails

```bash
# Check which metrics failed
cat lineages/m0_ocr/BLOCKED

# Common fixes:
# - More training data (aim for 500+ images)
# - Better teacher labels (use Gemini instead of Qwen-2B)
# - More training epochs (3-5 instead of 1)
# - Lower learning rate (1e-4 instead of 2e-4)
```

---

## Next Steps After M0

Once M0 is promoted:

### 1. Freeze M0
```bash
# Don't modify anything in lineages/m0_ocr/
# Model is now production baseline
```

### 2. Create M1 (Structured Output)

```bash
# Create new lineage
mkdir -p lineages/m1_structured
cp -r lineages/m0_ocr/*.yaml lineages/m1_structured/

# Update manifest to use structured prompts
# data/manifests/m1.yaml:
#   sources:
#     - name: structured_labels
#       file: structured_labels.jsonl
#       task_type: structured

# Generate new labels with JSON schema
python pipelines/data/generate_teacher_labels.py \
  --teacher gemini \
  --prompt-type structured \
  --output data/artifacts/shared/structured_labels.jsonl

# Build M1 dataset
python pipelines/data/build_datasets.py --lineage m1

# Train M1 starting from M0 checkpoint
python pipelines/training/train_lineage.py \
  --lineage m1_structured \
  --resume-from lineages/m0_ocr/checkpoints/best.ckpt
```

---

## File Checklist

Before starting, verify you have:

**Essential Code:**
- [ ] `models/assemblies/ocr_only.py` - Model
- [ ] `pipelines/data/generate_teacher_labels.py` - Label generation
- [ ] `pipelines/data/build_datasets.py` - Dataset building
- [ ] `pipelines/training/train_lineage.py` - Training
- [ ] `pipelines/evaluation/eval_lineage.py` - Evaluation
- [ ] `teachers/base.py` - Teacher models
- [ ] `teachers/registry.py` - Teacher factory
- [ ] `data/receipt_dataset.py` - PyTorch dataset

**Essential Config:**
- [ ] `lineages/m0_ocr/training.yaml` - Hyperparameters
- [ ] `lineages/m0_ocr/eval_contract.yaml` - Success thresholds
- [ ] `data/manifests/m0.yaml` - Dataset recipe

**Your Data:**
- [ ] Receipt images in `data/sources/receipts_raw/`

**API Keys:**
- [ ] `WANDB_API_KEY` set (required)
- [ ] `GOOGLE_API_KEY` set (optional, for Gemini teacher)

---

## Quick Commands Reference

```bash
# Setup
bash scripts/setup_environment.sh

# Data pipeline (all steps)
make data

# Data pipeline (individual steps)
make data-teacher TEACHER=gemini
make data-stability
make data-build LINEAGE=m0

# Training
make train LINEAGE=m0_ocr

# Evaluation
make eval LINEAGE=m0_ocr

# Cleanup
make clean
```

---

## Expected Timeline

| Phase | Time | Hardware |
|-------|------|----------|
| Setup | 5 min | Any |
| Label 100 images (Gemini) | 10 min | API |
| Label 100 images (Qwen-2B) | 30 min | CPU |
| Build dataset | 1 min | Any |
| Train (CPU, 1 epoch) | 30 min | CPU |
| Train (GPU, 3 epochs) | 20 min | A100 |
| Train (SageMaker FSDP) | 15 min | ml.g5.12xlarge |
| Evaluate | 5 min | Any |

**Total (CPU)**: ~1 hour  
**Total (GPU)**: ~30 minutes  
**Total (SageMaker)**: ~30 minutes + setup

---

## Cost Estimates

| Resource | Cost |
|----------|------|
| Gemini API (1000 images) | ~$0.50 |
| W&B (free tier) | $0 |
| Local GPU | $0 |
| SageMaker ml.g5.12xlarge (1 hour) | ~$5 |
| SageMaker Spot (70% off) | ~$1.50 |

**Cheapest path**: Use Gemini for labels + local GPU = **~$0.50 total**

---

## Support

- **Documentation**: See `docs/` folder
- **Issues**: Open GitHub issue
- **Questions**: Check `docs/pipeline_guide.md` and `docs/deployment_guide.md`

---

## Success Criteria

You're ready to move to production when:
- [ ] M0 passes eval contract (CER < 10%)
- [ ] Checkpoints saved to `lineages/m0_ocr/checkpoints/`
- [ ] `lineages/m0_ocr/eval_results.json` shows PASSED
- [ ] W&B run shows stable loss curves
- [ ] Sample predictions look good

Then freeze M0 and start M1! 🚀