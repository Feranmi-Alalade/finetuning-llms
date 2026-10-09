# Deployment Guide

A structured naming convention ensures that as you scale from testing to production and from OCR to full semantic extraction, your models remain findable and their purpose is clear.

## 1. Core Naming Convention

**Pattern**: `{org}/{base-model}-{stage}-{task}-{type}[-suffix]`

| Segment | Value | Description |
| :--- | :--- | :--- |
| **org** | `bildcraft` | Organization name |
| **base-model** | `qwen3-vl-8b` or `qwen3-vl-2b` | Base model architecture |
| **stage** | `m0`, `m1`, `unified` | Pipeline stage (OCR, Semantic, or Combined) |
| **task** | `ocr`, `semantic`, `receipt-understanding` | Specific task focus |
| **type** | `adapter`, `merged` | LoRA Adapter vs Full Merged Weights |
| **suffix** | `dev`, `prod`, `v1`... | Environment or version identifier |

## 2. Suggested Repository Names

### Stage M0: OCR & Stability
Focus: Transcription, field detection (merchant, date, total), and image rejection.

*   **Production (Merged)**: `bildcraft/qwen3-vl-8b-m0-ocr-merged`
*   **Production (Adapter)**: `bildcraft/qwen3-vl-8b-m0-ocr-adapter`
*   **Development**: `bildcraft/qwen3-vl-2b-m0-ocr-dev`

### Stage M1: Semantic Understanding
Focus: Item-level parsing, categorization, and complex reasoning.

*   **Production (Merged)**: `bildcraft/qwen3-vl-8b-m1-semantic-merged`
*   **Production (Adapter)**: `bildcraft/qwen3-vl-8b-m1-semantic-adapter`

### Unified / End-to-End
*   **Unified Model**: `bildcraft/qwen3-vl-8b-receipt-master`

## 3. GGUF & Specialized Formats (Ollama)

For edge deployment/local use.

*   **GGUF Master Repo**: `bildcraft/qwen3-vl-8b-m0-ocr-gguf`
    *   Filenames: `qwen3-vl-8b-m0-q4_k_m.gguf`
*   **Ollama Names**:
    *   `ollama run bildcraft/receipt-ai-m0:8b-q4`

## 4. Tagging Strategy (HF Metadata)

Include in `README.md` frontmatter:

```yaml
tags:
- multimodal
- vision
- ocr
- receipt-understanding
- qwen3-vl
- bildcraft
pipeline_tag: visual-question-answering
license: apache-2.0
```

## 5. Configuration Integration

Update `fine_tuning.yaml` to match these conventions:

```yaml
export:
  huggingface:
    enabled: true
    repo_id: "bildcraft/qwen3-vl-8b-m0-ocr-adapter"
    private: true
  
  ollama:
    enabled: true
    model_name: "bildcraft/receipt-m0-ocr"
```