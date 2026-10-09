# 🚀 1-Month LLM Inference Engineering Curriculum (Cloud-First)

## Philosophy
Touch production-grade systems immediately. Learn theory only when it explains observed bottlenecks.

---

## Week 0 (½ Day) — Fast Setup (Don’t Skip)
**Goal:** Remove friction so Day 1 is productive.

*   Create AWS account + budget alerts
*   Launch EC2 G5 or P4d (on-demand)
*   **Install:**
    *   CUDA Toolkit
    *   NVIDIA Driver
    *   PyTorch (CUDA build)
*   **Verify:**
    *   `nvidia-smi`
    *   `torch.cuda.is_available()`

### Refs
*   **CUDA install:** [nvidia.com](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
*   **AWS GPU instances:** [amazon.com](https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html)

---

## WEEK 1 — Single-Node Cloud Inference (Hit the Ground Running)
**Theme:** *“Why is inference slow, and what actually fixes it?”*

You will serve a real model on a cloud GPU on Day 1.

---

### Days 1–2 — Transformer Inference Anatomy (Cloud GPU)
**Goal:** Understand where latency & memory go by observation, not slides.

#### Study
*   Transformer inference path (prefill vs decode)
*   KV Cache memory growth
*   **Attention = dominant bottleneck**

#### Hands-On
*   Deploy Llama-3-8B or Mistral-7B using:
    *   `transformers` + `accelerate`
*   **Measure:**
    *   VRAM usage
    *   Tokens/sec
    *   TTFT (time to first token)

#### Key Insight
You will see that attention + KV cache dominate memory.

#### Refs
*   **HF Accelerate:** [huggingface.co](https://huggingface.co/docs/accelerate)
*   **Transformer inference overview:** [huggingface.co](https://huggingface.co/blog/llm-inference)

---

### Days 3–4 — Modern Attention & Memory Optimization
**Goal:** Learn the attention variants that actually matter for inference.

#### Study
*   FlashAttention & FlashAttention-2
*   MQA vs GQA vs MHA
*   SDPA (PyTorch native)
*   KV cache mechanics

#### Hands-On
*   Enable FlashAttention in PyTorch
*   **Benchmark:**
    *   Standard attention vs FlashAttention
    *   Long context (2k → 8k)

#### Key Insight
FlashAttention is about memory movement, not math speed.

#### Refs
*   **FlashAttention paper:** [arxiv.org](https://arxiv.org/abs/2205.14135)
*   **FlashAttention-2:** [arxiv.org](https://arxiv.org/abs/2307.08691)
*   **PyTorch SDPA:** [pytorch.org](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

---

### Day 5 — Quantization That Actually Works on GPUs
**Goal:** Reduce VRAM without killing throughput.

#### Study
*   GPTQ vs AWQ (GPU-focused)
*   Why GGUF is not ideal for cloud GPUs
*   SmoothQuant & QLoRA (context only)

#### Hands-On
*   Quantize same model with:
    *   FP16
    *   GPTQ-4bit
    *   AWQ-4bit
*   **Compare:**
    *   VRAM
    *   Tokens/sec
    *   Output quality

#### Refs
*   **GPTQ:** [arxiv.org](https://arxiv.org/abs/2210.17323)
*   **AWQ:** [arxiv.org](https://arxiv.org/abs/2306.00978)
*   **SmoothQuant:** [arxiv.org](https://arxiv.org/abs/2211.10438)

---

## WEEK 2 — High-Performance Serving Engines (Real Systems)
**Theme:** *“Scripts don’t scale. Engines do.”*

---

### Days 1–2 — vLLM & PagedAttention
**Goal:** Understand why vLLM is the default choice today.

#### Study
*   Continuous batching
*   **PagedAttention** (solves KV cache fragmentation)
*   Throughput vs latency tradeoffs

#### Hands-On
*   Deploy vLLM server
*   **Tune:**
    *   `max_model_len`
    *   `tensor_parallel_size`
*   Load test concurrent users

#### Refs
*   **vLLM paper:** [arxiv.org](https://arxiv.org/abs/2309.06180)
*   **vLLM repo:** [github.com](https://github.com/vllm-project/vllm)

---

### Days 3–4 — TensorRT-LLM (NVIDIA Gold Standard)
**Goal:** Learn when hardware-specific compilation beats everything else.

#### Study
*   TensorRT graph compilation
*   Kernel fusion
*   FP8 / INT8 / INT4 paths
*   AWQ inside TensorRT-LLM

#### Hands-On
*   Compile a model with TensorRT-LLM
*   **Compare vs vLLM:**
    *   TTFT
    *   Throughput
    *   GPU utilization

#### Refs
*   **TensorRT-LLM:** [github.com](https://github.com/NVIDIA/TensorRT-LLM)
*   **NVIDIA inference optimization:** [nvidia.com](https://developer.nvidia.com/deep-learning-performance-training-inference)

---

### Day 5 — Benchmarking Like an Engineer
**Goal:** Measure what matters.
*   TTFT vs TPOT
*   Throughput under concurrency
*   GPU memory fragmentation

#### Tools
*   `locust`
*   `nvidia-smi dmon`
*   Prometheus-style metrics

---

## WEEK 3 — Distributed Inference & AWS Silicon
**Theme:** *“70B+ models don’t fit on one GPU.”*

---

### Days 1–2 — Parallelism That Actually Works
**Goal:** Choose the right parallelism for inference.

#### Study
*   Tensor Parallelism (TP)
*   Pipeline Parallelism (PP)
*   Data Parallelism (DP)
*   Communication costs (All-Reduce)

#### Hands-On
*   Run vLLM with TP=2–4
*   Measure scaling efficiency

#### Refs
*   **Megatron-LM:** [arxiv.org](https://arxiv.org/abs/1909.08053)
*   **NCCL overview:** [nvidia.com](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)

---

### Days 3–4 — AWS Inferentia & Trainium
**Goal:** Understand non-CUDA inference economics.

#### Study
*   Inferentia2 architecture
*   Neuron SDK compilation model
*   Cost/perf tradeoffs vs NVIDIA

#### Hands-On
*   Spin up inf2
*   Compile & run a model with Neuron SDK

#### Refs
*   **Neuron SDK:** [amazonaws.com](https://awsdocs-neuron.readthedocs-hosted.com/)
*   **Inf2 overview:** [amazon.com](https://aws.amazon.com/ec2/instance-types/inf2/)

---

### Day 5 — Multi-Node & Networking
**Goal:** Remove the hidden bottleneck: networking.
*   EFA (Elastic Fabric Adapter)
*   NCCL over EFA
*   **Ray** for orchestration

#### Refs
*   **EFA:** [amazon.com](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
*   **Ray:** [ray.io](https://docs.ray.io/en/latest/)

---

## WEEK 4 — Production SageMaker + Local / Edge Inference
**Theme:** *“Ship it, then simplify it.”*

---

### Days 1–2 — SageMaker Production Patterns
**Goal:** Turn experiments into endpoints.
*   Real-time vs Async endpoints
*   Auto-scaling policies
*   Multi-model vs single-model endpoints

#### Hands-On
*   Deploy vLLM or TGI via SageMaker SDK

#### Refs
*   **SageMaker inference:** [amazon.com](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)

---

### Day 3 — Observability & Cost Engineering
**Goal:** Make infra survivable.
*   CloudWatch metrics
*   GPU utilization alerts
*   Cost / 1k tokens analysis

---

### Day 4 — Local & Edge Inference (Finally)
**Goal:** Now that you understand cloud inference, simplify.

#### Study
*   `llama.cpp` internals
*   GGML vs GGUF
*   Ollama ergonomics

#### Hands-On
*   Run same model locally via:
    *   `llama.cpp`
    *   `llama-cpp-python`
    *   Ollama

#### Key Insight
GGUF exists because CPU ≠ GPU, not because it’s “better”.

#### Refs
*   **llama.cpp:** [github.com](https://github.com/ggerganov/llama.cpp)
*   **GGUF format:** [github.com](https://github.com/ggerganov/ggml)

---

### Day 5 — Capstone
**Deliverable:**
*   Cloud-hosted LLM endpoint
*   Quantized
*   Auto-scaled
*   Benchmarked
*   Cost-analyzed

---

## Final Outcome (What You’ll Be Able To Do)
After this month, you can:
*   Look at any LLM deployment and diagnose bottlenecks
*   **Choose between:**
    *   vLLM vs TensorRT-LLM vs TGI
*   **Decide:**
    *   CUDA vs Inferentia
    *   AWQ vs GPTQ
    *   TP vs PP
*   Build production inference systems, not demos
