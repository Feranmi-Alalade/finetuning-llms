# Fine-Tuning LLMs & Multimodal AI Research

This repository contains a suite of projects and research materials focused on the fine-tuning, optimization, and production deployment of Large Language Models (LLMs) and Multimodal models.

## 📂 Repository Structure

The repository is organized into several key areas:

### 1. [Multimodal Receipt AI]
A production-grade pipeline for Fast-Moving Consumer Goods (FMCG) receipt understanding.
*   **Core Logic**: Teacher-student distillation using Gemini 2.0 and Qwen-VL.
*   **Data Tiering**: Bronze (Raw) → Silver (Normalized/Stability) → Gold (Labeled).
*   **Stability Training**: Automated generation of blurry, partial, and low-contrast receipt variants.
*   **Cloud Integration**: High-fidelity S3 synchronization for large-scale dataset management.

### 2. [LLM Inference Research]
A comprehensive study guide and documentation for engineering high-performance LLM inference systems.
*   **[1-Month Curriculum]**: A structured guide from single-node basics to multi-node production clusters.
*   **Technical Deep Dives**:
    *   **Engines**: vLLM, TensorRT-LLM, Text Generation Inference (TGI).
    *   **Optimization**: Quantization (AWQ, GPTQ, GGUF), PagedAttention, and Speculative Decoding.
    *   **Custom Silicon**: Research on AWS Trainium and Inferentia economics.

## 🛠 Tech Stack

*   **Languages**: Python (3.11+), SQL, C++ (`llama.cpp`).
*   **Package Management**: [UV](https://github.com/astral-sh/uv) (Extremely fast).
*   **Core Libraries**: PyTorch, Transformers, Accelerate, DeepSpeed, Ray.
*   **Inference Engines**: vLLM, TensorRT-LLM, Hugging Face TGI.
*   **Cloud Infrastructure**: AWS SageMaker, EC2 (G5, P4d, Inf2, Trn1).

## 🚀 Getting Started

To get started with the Multimodal Receipt AI project:
```bash
cd multimodal-receipt-ai
bash scripts/setup_environment.sh
```

To explore the LLM Inference study materials:
```bash
cd llm-inference
# Start with the Goals.md file for the 1-month plan
```
