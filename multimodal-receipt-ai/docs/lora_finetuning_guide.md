# LoRA Fine-Tuning Guide for Receipt OCR

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Pretrained VLM Model                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐      ┌──────────────┐      ┌────────┐ │
│  │ Vision Encoder   │      │  Projector   │      │  LLM   │ │
│  │   (CLIP ViT)     │ ───> │    (MLP)     │ ───> │ +LoRA  │ │
│  │    ❄️ FROZEN     │      │  ❄️ FROZEN   │      │   🔥   │ │
│  └──────────────────┘      └──────────────┘      └────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┐
```

## Why This Approach

### ✅ What We DO
- Start from pretrained VLM (Qwen2-VL, Phi-3-Vision, LLaVA)
- Freeze vision encoder (preserve spatial grounding)
- Freeze projector (preserve multimodal alignment)
- Apply LoRA only to LLM (teach what to say, not how to see)

### ❌ What We DON'T DO
- ~~Pretrain projector from scratch~~ (destroys alignment)
- ~~Unfreeze vision encoder~~ (wastes compute, risks instability)
- ~~Full LLM fine-tuning~~ (expensive, prone to overfitting)

## Supported Base Models

| Role | Model | Size | Best For |
|-------|-------|------|----------|
| **Teacher** | Qwen3-VL | 32B | High-fidelity OCR & reasoning (Teacher) |
| **Teacher** | Qwen3-VL | 8B | Local teacher sanity checks |
| **Student** | Qwen2.5-VL | 7B | Best OCR student (Recommended) |
| **Student** | Qwen2.5-VL | 3B | Small student for local/CPU |

**Recommendation**: Start with **Qwen3-VL-32B** as your teacher and **Qwen2.5-VL-7B** as your student for the best distillation results.

## LoRA Configuration

### Target Modules

**For Qwen2-VL:**
```yaml
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

**For LLaMA/Vicuna:**
```yaml
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
```

### Hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `lora_r` | 64-128 | Higher rank for complex OCR task |
| `lora_alpha` | 2x rank | Standard scaling |
| `lora_dropout` | 0.05 | Light regularization |
| `learning_rate` | 2e-4 | Standard for LoRA |
| `weight_decay` | 0.0 | LoRA doesn't need it |
| `warmup_ratio` | 0.03 | 3% warmup |

## Training Process

### 1. Data Format

The system expects JSONL files with images relative to `data/sources/receipts_raw/`.

```json
{
  "image_path": "receipts/001.jpg",
  "prompt": "Transcribe this receipt exactly as shown.",
  "response": "WALMART\nDate: 01/15/2024\nTime: 14:23\n..."
}
```

### 2. Training Command

```bash
# Basic training
make train LINEAGE=m0_ocr

# Custom experiment
python pipelines/training/train_lineage.py \
  --lineage m0_ocr \
  --experiment-name my-receipt-experiment
```

### 3. What Gets Trained

**Trainable parameters (~50M-150M):**
- LoRA adapters in LLM attention and MLP layers.
- Total: ~1-3% of full model parameters.

**Frozen parameters:**
- Vision encoder and Projector (MANDATORY).

## What Breaks OCR (Avoid These)

### ❌ Training Mistakes

1. **Unfreezing vision encoder**
   - Destroys spatial grounding.
   - Causes position drift in OCR.

2. **Unfreezing projector**
   - Breaks existing multimodal alignment.
   - Requires massive data to re-align.

3. **Too high learning rate (>5e-4)**
   - LoRA adapters overfit quickly.
   - Loss of generalization.

### ✅ OCR-Preserving Settings

```yaml
learning_rate: 1e-4        # Conservative LR
max_epochs: 3              # Avoid over-fitting
lora_r: 64                 # Balanced capacity
gradient_clip_val: 1.0     # Standard clipping
```

## Evaluation Checklist

After training, verify the model against your `eval_contract.yaml`:

- [ ] CER (Character Error Rate) < 10%
- [ ] WER (Word Error Rate) < 20%
- [ ] Exact match > 30%
- [ ] No hallucination on empty images
- [ ] Preserves layout (newlines, spacing)

## Merging LoRA Weights

After training, you can merge LoRA weights into the base model for faster inference:

```python
from peft import PeftModel
from transformers import Qwen2VLForConditionalGeneration

# 1. Load base model
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# 2. Load LoRA weights
model = PeftModel.from_pretrained(base_model, "lineages/m0_ocr/checkpoints/lora_final")

# 3. Merge and unload
merged_model = model.merge_and_unload()

# 4. Save merged model
merged_model.save_pretrained("models/m0_ocr_merged")
```

## Next Steps

1. **M1 (Structured)**: Add JSON schema enforcement using `system/schemas/receipt_ocr.json`.
2. **M2 (Semantic)**: Item canonicalization and categories using `teachers/prompts/semantics/`.
3. **M3 (Tools)**: External knowledge integration via tool-calling prompts.

For each new lineage:
- Start from previous frozen model checkpoint.
- Add ONE new capability via prompting and data mixture.
- Apply LoRA only to the LLM component.
- **Never** retrain or unfreeze the vision encoder or projector.