# Task Roadmap — Multimodal Receipt Intelligence

Each task introduces a new model lineage. Old models are frozen, evaluated, and never silently modified. Models grow vertically in capability, not horizontally in hacks.

> [!IMPORTANT]
> **Core Principle**: Each task introduces a new model lineage. Old models are frozen, evaluated, and never silently modified.

---

## Model Lineages Overview

| Lineage | Scope | Purpose |
| :--- | :--- | :--- |
| **M0** | Structured OCR | Exact transcription + JSON |
| **M2** | OCR + semantics | Meaningful line items |
| **M3** | OCR + tools | External knowledge integration |

---

## Task 1 — Structured OCR (M0)

**Goal**: Teach a vision-language model to faithfully transcribe receipts and emit strict structured outputs. This task focuses on **perception + discipline**, not reasoning.

### Explicit Capabilities
- Exact text transcription (no paraphrasing)
- Line-level layout awareness
- **Structured fields**:
  - `merchant_name`
  - `date`
  - `line_items` (raw text)
  - `subtotal`, `tax`, `total`
- Detect unreadable / missing regions

### Inputs & Outputs
- **Inputs**:
  - Image: receipt photo (phone quality, skewed, noisy)
  - Text Prompt: Structured OCR instruction
- **Outputs**:
  - JSON matching schema
  - No hallucinated fields
  - Explicit `null` for missing values

### Why This Task Exists
- Establishes visual grounding
- Forces schema discipline early
- Creates baseline for all future models

### Model Lineage
- `M0_OCR` → `Qwen2.5-VL-7B` (Structured OCR)

---

## Task 2 — Semantic Line-Item Understanding (M2)

**Goal**: Move from **what is written** → **what it means**. This task introduces controlled abstraction without external tools.

### Explicit Capabilities
- **Canonicalize item names**: "ORG BAN 2LB" → "Bananas"
- **Category classification**: Produce / Dairy / Household
- **Quantity inference**: Count vs Weight
- **Price normalization**: Unit vs Total

### Inputs & Outputs
- **Inputs**:
  - Image: receipt photo
  - Text Context: raw OCR output from M1
- **Outputs**:
  - Structured JSON with semantic fields: `canonical_name`, `category`, `quantity`, `unit_type`, `price_per_unit`

### Constraints
- Must not change OCR text
- Semantics must reference original line items
- Uncertainty must be explicit

### Model Lineage
- `M2_SEMANTIC_RECEIPTS`
- Initialized from M1 weights
- M1 frozen as regression baseline

---

## Task 3 — Tool / Function Calling (M3)

**Goal**: Enable the model to know when it does not know and request external information. This task introduces **agentic behavior**, not just prediction.

### Explicit Capabilities
- Detect items needing enrichment
- Emit structured tool calls (not prose)
- Merge tool results into final output

### Example Tool Calls
- Nutrition lookup
- Product metadata (EAN/UPC)
- Allergen detection

### Inputs & Outputs
- **Inputs**: Image, OCR + semantic context, Tool schemas
- **Outputs**:
  - JSON with `tool_calls[]`
  - Enriched line items
  - Provenance tracking (what came from tools)

### Constraints
- No hallucinated external facts
- Tools must be optional, not mandatory
- Tool failure must not break output schema

### Model Lineage
- `M3_RECEIPT_AGENT`
- Initialized from M2
- M2 remains deployable independently

---

## Canonical Repository Structure

```text
multimodal-receipt-ai/
├── README.md                # Philosophy, quick start, and model lineages
├── Makefile                 # Entry points (make data, make train, make eval)
├── pyproject.toml           # Dependencies (uv-managed)
├── uv.lock
│
├── docs/
│   ├── vision.md            # Incremental growth philosophy
│   ├── task_roadmap.md      # M0 → M3 roadmap (this file)
│   ├── walkthrough.md       # Getting started guide
│   └── lora_finetuning_guide.md # Technical details for Qwen-VL LoRA
│
├── data/
│   ├── sources/             # Raw receipt images and stability tests
│   ├── artifacts/           # Built datasets (shared & model-scoped)
│   ├── manifests/           # Dataset recipes (m0.yaml, etc.)
│   ├── utils/               # image_io.py, jsonl.py, validation.py
│   └── receipt_dataset.py   # PyTorch Dataset for VLMs
│
├── lineages/
│   ├── m0_ocr/              # Baseline OCR configuration
│   │   ├── training.yaml    # Hyperparams & LoRA config
│   │   ├── eval_contract.yaml # Thresholds (CER/WER)
│   │   └── checkpoints/     # Locally saved weights
│   ├── m1_structured/       # Planned lineages
│   └── m2_semantic/
│
├── teachers/
│   ├── registry.py          # Factory (Gemini, Qwen-2B/7B)
│   ├── base.py              # Teacher classes (Gemini & Qwen)
│   ├── inference/
│   │   └── qwen_vl.py       # Helper for local Qwen inference
│   └── prompts/             # Task-specific prompt templates
│
├── pipelines/
│   ├── data/
│   │   ├── generate_teacher_labels.py
│   │   ├── generate_stability_data.py
│   │   └── build_datasets.py
│   ├── training/
│   │   ├── lightning_module.py  # Model wrapper logic
│   │   └── train_lineage.py     # Unified training entry point
│   └── evaluation/
│       └── eval_lineage.py      # Quantitative evaluation
│
├── models/
│   ├── registry.py          # Lineage-based model selection
│   ├── components/          # Abstract base classes
│   └── assemblies/
│       └── ocr_only.py      # M0 model (Qwen2-VL wrapper)
│
├── evaluation/
│   ├── metrics/             # CER/WER metrics
│   ├── frozen_sets/         # Immutable test data
│   ├── regression/          # check_thresholds.py
│   ├── freeze_test_set.py   # Utility to lock test data
│   └── run_frozen_eval.py   # Evaluation runner
│
├── system/
│   ├── schemas/             # JSON schemas (receipt_ocr, etc.)
│   ├── logging.py           # W&B and console logging
│   └── config.py            # Global environment & SageMaker config
│
└── experiments/
    └── notes.md             # Insights and dead ends
        └── Human thinking, dead ends, insights
```

---

## Final Design Rules (Non-Negotiable)

- [ ] No task modifies an existing model.
- [ ] Every lineage has a frozen eval set and explicit "does not support".
- [ ] Teachers are replaceable.
- [ ] Stability tasks are always mixed in.

---

> [!TIP]
> **One-Line North Star**: Build small multimodal systems that grow in capability without forgetting how to see.