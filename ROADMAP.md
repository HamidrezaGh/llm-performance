# LLM Performance Engineering — Roadmap

A structured learning path for becoming a production-level LLM performance engineer,
targeting roles at Anthropic, OpenAI, Google DeepMind, Meta, and NVIDIA.

**Principles:**
- Only claim results that are implemented and benchmarked.
- Every project includes profiling — Nsight Systems at minimum.
- Every benchmark reports: hardware, warmup runs, N iterations, mean, std, p50/p95/p99.
- Each phase builds on the previous one. Do not skip phases.

---

## Phase 1: GPU Architecture and CUDA Fundamentals

**Objective:** Build a deep mental model of GPU hardware and master low-level CUDA programming patterns
that all higher-level optimization rests on.

---

### 1.1 — CUDA Vector Addition ✅

| Field | Detail |
|---|---|
| **Goal** | Write a first CUDA kernel, understand host/device memory model, and benchmark against CPU and PyTorch baselines. |
| **Tools** | CUDA C++, nvcc, Python, CUDA events for timing |
| **Deliverables** | `vector_add.cu`, `benchmark.py`, Colab notebook with results table |
| **Metrics** | Kernel execution time (ms), speedup vs CPU, speedup vs PyTorch, effective memory bandwidth (GB/s) |
| **Status** | Complete |

---

### 1.2 — Tiled Matrix Multiplication with Shared Memory ✅

| Field | Detail |
|---|---|
| **Goal** | Implement tiled GEMM using `__shared__` memory. Understand memory coalescing, bank conflicts, and tile-size tradeoffs. |
| **Tools** | CUDA C++, nvcc, CUDA events, Python |
| **Deliverables** | `matrix_mul.cu` (naive + tiled), benchmark script, results comparison table |
| **Metrics** | Execution time (ms) for naive vs tiled vs PyTorch, GFLOPS achieved, % of peak memory bandwidth, speedup from tiling |
| **Status** | Complete |

---

### 1.3 — GPU Profiling with Nsight Systems and Nsight Compute

| Field | Detail |
|---|---|
| **Goal** | Profile the vector-add and tiled-matmul kernels from 1.1 and 1.2. Learn to read timelines, identify memory bottlenecks, measure occupancy, and produce roofline plots. |
| **Tools** | Nsight Systems (`nsys`), Nsight Compute (`ncu`), roofline model |
| **Deliverables** | Nsight timeline screenshots, roofline analysis chart, written analysis of whether each kernel is compute-bound or memory-bound, occupancy report |
| **Metrics** | SM occupancy (%), achieved memory bandwidth vs theoretical peak (%), arithmetic intensity (FLOP/byte), L1/L2 cache hit rates |
| **Why this matters** | Profiling is the single most important skill. Without it, optimization is guesswork. Every project after this one should include profiling. |

---

### 1.4 — Fused GEMM + Bias + ReLU Kernel

| Field | Detail |
|---|---|
| **Goal** | Fuse matrix multiply, bias addition, and ReLU activation into a single kernel. Understand kernel fusion as an optimization strategy to eliminate intermediate memory traffic. |
| **Tools** | CUDA C++, Nsight Compute, CUDA streams |
| **Deliverables** | `fused_gemm.cu` with fused vs unfused comparison, Nsight Compute profile showing reduced memory transactions |
| **Metrics** | Execution time reduction (%), global memory transactions (fused vs unfused), achieved GFLOPS, kernel launch overhead savings |

---

### 1.5 — Memory Bandwidth Analysis and Roofline Modeling

| Field | Detail |
|---|---|
| **Goal** | Build a benchmark suite that measures achieved memory bandwidth at different access patterns (coalesced, strided, random). Construct a roofline model for your target GPU. |
| **Tools** | CUDA C++, Nsight Compute, Python (matplotlib for plots) |
| **Deliverables** | Microbenchmark kernels for each access pattern, roofline chart with your kernels plotted, written analysis |
| **Metrics** | Achieved bandwidth (GB/s) per access pattern, % of theoretical peak (e.g., A100 = 2 TB/s HBM), arithmetic intensity for each kernel |

---

### 1.6 — Warp-Level Primitives and Reductions

| Field | Detail |
|---|---|
| **Goal** | Implement parallel reduction (sum, max) using warp shuffle intrinsics (`__shfl_down_sync`). Understand warp divergence and its performance cost. |
| **Tools** | CUDA C++, Nsight Compute |
| **Deliverables** | Reduction kernel (naive → shared memory → warp shuffle), benchmark across array sizes, divergence analysis |
| **Metrics** | Execution time per reduction approach, bandwidth utilization, warp execution efficiency (%) |

---

## Phase 2: Transformer Internals and Custom Kernels

**Objective:** Understand the transformer architecture at the GPU-kernel level and write custom
high-performance implementations of its core operations.

---

### 2.1 — Triton Softmax Kernel

| Field | Detail |
|---|---|
| **Goal** | Write a numerically stable fused softmax in Triton. Compare against PyTorch's `F.softmax` and understand online softmax (the foundation of Flash Attention). |
| **Tools** | Triton, PyTorch, Nsight Systems |
| **Deliverables** | `triton_softmax.py`, benchmark notebook, numerical correctness tests |
| **Metrics** | Execution time (ms) vs PyTorch baseline, memory bandwidth utilization, max absolute error vs reference |

---

### 2.2 — Triton LayerNorm Kernel

| Field | Detail |
|---|---|
| **Goal** | Implement fused LayerNorm in Triton with single-pass mean/variance computation. |
| **Tools** | Triton, PyTorch |
| **Deliverables** | `triton_layernorm.py`, comparison against `torch.nn.LayerNorm`, correctness tests |
| **Metrics** | Latency (ms) across hidden dimensions (768, 1024, 4096, 8192), throughput (elements/sec), speedup vs PyTorch |

---

### 2.3 — Triton Rotary Positional Embedding (RoPE) Kernel

| Field | Detail |
|---|---|
| **Goal** | Implement RoPE in Triton. Understand why positional encoding is compute-bound vs memory-bound. |
| **Tools** | Triton, PyTorch |
| **Deliverables** | `triton_rope.py`, benchmark across sequence lengths, correctness validation against HuggingFace implementation |
| **Metrics** | Latency (ms) at seq_len = {512, 2048, 8192, 32768}, speedup vs PyTorch |

---

### 2.4 — Naive Attention Kernel (Scaled Dot-Product)

| Field | Detail |
|---|---|
| **Goal** | Implement basic scaled dot-product attention from scratch in Triton. Compute Q\*K^T, apply causal mask, softmax, and V projection. Observe the O(n²) memory bottleneck firsthand — this establishes the baseline that Flash Attention improves on. |
| **Tools** | Triton, PyTorch |
| **Deliverables** | `naive_attention.py`, memory profiling showing O(n²) growth, benchmark across sequence lengths, correctness validation |
| **Metrics** | Latency (ms) at seq_len = {512, 1K, 2K, 4K, 8K}, peak memory (MB) showing quadratic growth, TFLOPS, numerical accuracy vs PyTorch |
| **Why this matters** | You must see the O(n²) memory wall yourself before Flash Attention's IO-aware tiling makes intuitive sense. |

---

### 2.5 — Flash Attention Implementation

| Field | Detail |
|---|---|
| **Goal** | Implement Flash Attention v2 in Triton. Understand tiled online softmax, IO-aware algorithm design, and why this is the most impactful kernel optimization for transformers. |
| **Tools** | Triton, PyTorch, Nsight Compute |
| **Deliverables** | `flash_attention.py`, benchmark against naive attention (2.4) and `torch.nn.functional.scaled_dot_product_attention`, memory usage comparison |
| **Metrics** | Latency (ms) at seq_len = {1K, 4K, 16K, 64K}, peak memory usage (MB), TFLOPS achieved, speedup vs naive O(n²) attention from 2.4 |
| **Why this matters** | Flash Attention is the single most impactful kernel innovation in modern transformer systems. Understanding it deeply is non-negotiable for inference roles. |

---

### 2.6 — Transformer Block Assembly: Plugging Custom Kernels into Llama

| Field | Detail |
|---|---|
| **Goal** | Replace PyTorch's default softmax, LayerNorm, and RoPE in a Llama-2 or Llama-3 model with your Triton kernels from 2.1–2.3. Measure end-to-end impact. |
| **Tools** | PyTorch, Triton, HuggingFace Transformers |
| **Deliverables** | Modified Llama forward pass, A/B benchmark (stock vs custom kernels), correctness validation (logit comparison) |
| **Metrics** | End-to-end inference latency (ms), tokens/sec, per-layer kernel time breakdown, max logit difference |

---

### 2.7 — Group GEMM / Batched GEMM for MoE

| Field | Detail |
|---|---|
| **Goal** | Implement grouped/batched matrix multiplication as used in Mixture-of-Experts layers. Understand expert routing and load balancing at the kernel level. |
| **Tools** | Triton or CUDA C++, cuBLAS batched GEMM for comparison |
| **Deliverables** | `group_gemm.py`, benchmark across expert counts (8, 16, 64), comparison vs cuBLAS |
| **Metrics** | GFLOPS per expert count, load imbalance overhead, latency vs cuBLAS |

---

## Phase 3: Inference Performance Engineering

**Objective:** Optimize end-to-end LLM inference — from torch.compile to KV cache management to
continuous batching to serving 70B+ models.

---

### 3.1 — torch.compile Deep Dive on Llama-8B ✅ (partial)

| Field | Detail |
|---|---|
| **Goal** | Benchmark `torch.compile` on Llama-3.1-8B across all compile modes (`default`, `reduce-overhead`, `max-autotune`). Analyze graph breaks and their performance impact. |
| **Tools** | PyTorch 2.x, torch.compile, `torch._dynamo`, Nsight Systems |
| **Deliverables** | Notebook with compile mode comparison, graph break analysis, Nsight timeline showing fused vs unfused regions |
| **Metrics** | Tokens/sec per compile mode, compile time (s), graph break count, latency reduction (%) |
| **Status** | Partial — baseline done, needs compile mode sweep and graph break analysis |

---

### 3.2 — KV Cache Optimization and Paged Attention

| Field | Detail |
|---|---|
| **Goal** | Implement a KV cache manager with paged memory allocation (inspired by vLLM's PagedAttention). Understand how KV cache is the primary memory bottleneck in autoregressive generation. |
| **Tools** | PyTorch, Triton (for paged attention kernel), Python |
| **Deliverables** | KV cache allocator with page tables, paged attention kernel, memory usage comparison (paged vs contiguous), benchmark at various sequence lengths and batch sizes |
| **Metrics** | Peak memory usage (GB), memory waste (%), max concurrent sequences at fixed GPU memory, tokens/sec at batch_size = {1, 8, 32, 128} |
| **Why this matters** | KV cache management determines how many requests a serving system can handle concurrently. This is the core of vLLM's 10-24x throughput improvement. |

---

### 3.3 — Minimal LLM Inference Engine

| Field | Detail |
|---|---|
| **Goal** | Build a minimal LLM inference engine from scratch that combines KV cache management, continuous batching, and token streaming into one cohesive system. Implement iteration-level scheduling (not request-level), memory-aware request admission, and async token output. |
| **Tools** | Python, PyTorch, asyncio |
| **Deliverables** | Inference engine with: KV cache allocator (fixed-size block pool), continuous batching scheduler with request queue, admission control and preemption, token-by-token streaming output. Load test harness with synthetic workloads (varying prompt/generation lengths). |
| **Metrics** | Throughput (tokens/sec) at varying request rates, p50/p95/p99 time-to-first-token (TTFT), p50/p95/p99 time-per-output-token (TPOT), max concurrent requests at fixed GPU memory, speedup vs static batching (target: ≥2x) |
| **Why this matters** | This is the project that ties KV cache, batching, and streaming together into a working system — the kind of thing you'd build on day one at a serving team. |

---

### 3.4 — Speculative Decoding

| Field | Detail |
|---|---|
| **Goal** | Implement speculative decoding using a draft model to accelerate autoregressive generation. Understand acceptance rate, draft length tuning, and when speculation helps vs hurts. |
| **Tools** | PyTorch, HuggingFace Transformers |
| **Deliverables** | Speculative decoding implementation with configurable draft length, benchmark across draft model sizes, acceptance rate analysis |
| **Metrics** | Tokens/sec speedup vs vanilla autoregressive, acceptance rate (%), optimal draft length, latency at batch_size = 1 |

---

### 3.5 — 70B Model Multi-GPU Serving (Tensor Parallelism)

| Field | Detail |
|---|---|
| **Goal** | Serve Llama-2-70B (or Llama-3-70B) across 4x A100 GPUs using tensor parallelism. Measure real serving performance under load. |
| **Tools** | vLLM or TensorRT-LLM, Ray, NCCL, Nsight Systems |
| **Deliverables** | Deployment scripts, throughput dashboard, Nsight timeline showing inter-GPU communication, load test results |
| **Metrics** | Tokens/sec at batch_size = {1, 8, 32}, p99 latency (ms), GPU utilization (%), inter-GPU communication overhead (%), scaling efficiency (1-GPU vs 4-GPU) |

---

### 3.6 — Quantization: FP8 and INT4 Inference

| Field | Detail |
|---|---|
| **Goal** | Implement and benchmark FP8 and INT4 quantized inference. Compare GPTQ, AWQ, and bitsandbytes approaches. Understand accuracy vs throughput tradeoffs. |
| **Tools** | Triton, bitsandbytes, auto-gptq, autoawq, PyTorch |
| **Deliverables** | Quantized model benchmarks, perplexity comparison (FP16 vs FP8 vs INT4), custom Triton dequantization kernel |
| **Metrics** | Tokens/sec per quantization method, memory reduction (%), perplexity degradation, peak memory usage |

---

## Phase 4: Distributed LLM Systems

**Objective:** Master distributed training and serving — data/tensor/pipeline parallelism,
fault tolerance, and multi-node scaling.

---

### 4.1 — Distributed Training with FSDP

| Field | Detail |
|---|---|
| **Goal** | Fine-tune a 7B model using PyTorch FSDP across multiple GPUs. Understand sharding strategies, activation checkpointing, and communication overhead. |
| **Tools** | PyTorch FSDP, NCCL, Nsight Systems, torchrun |
| **Deliverables** | Training script with FSDP, scaling analysis (1/2/4/8 GPUs), Nsight timeline showing communication vs compute overlap |
| **Metrics** | Training throughput (tokens/sec), GPU utilization (%), communication overhead (%), memory per GPU (GB), scaling efficiency |

---

### 4.2 — Pipeline and Tensor Parallelism Comparison

| Field | Detail |
|---|---|
| **Goal** | Implement tensor parallelism (TP) and pipeline parallelism (PP) on a transformer model. Analyze the tradeoffs: TP reduces per-layer latency but increases communication; PP reduces memory but introduces pipeline bubbles. |
| **Tools** | PyTorch, NCCL, Megatron-LM (reference), Nsight Systems |
| **Deliverables** | TP implementation (column/row parallel linear layers), PP implementation (micro-batch scheduling), comparison notebook |
| **Metrics** | Throughput (samples/sec) for TP vs PP vs TP+PP, pipeline bubble fraction (%), communication volume (GB), memory per GPU |

---

### 4.3 — Multi-Node Training and Network Profiling

| Field | Detail |
|---|---|
| **Goal** | Scale training beyond a single node (2+ nodes). Profile inter-node communication and identify network bottlenecks. |
| **Tools** | PyTorch DDP/FSDP, NCCL, InfiniBand/RoCE, Nsight Systems, `nccl-tests` |
| **Deliverables** | Multi-node training script, NCCL bandwidth benchmark, network latency analysis, comparison of intra-node (NVLink) vs inter-node (InfiniBand) |
| **Metrics** | All-reduce bandwidth (GB/s), training throughput at 1-node vs 2-node vs 4-node, communication-to-compute ratio, scaling efficiency (%) |

---

### 4.4 — Fault-Tolerant Serving Cluster

| Field | Detail |
|---|---|
| **Goal** | Deploy an LLM serving cluster with fault injection testing. Implement health checks, automatic failover, and request redistribution when a GPU node goes down. |
| **Tools** | vLLM, Ray Serve, Python, Docker |
| **Deliverables** | Cluster deployment scripts, fault injection test suite (kill node, network partition, OOM), throughput degradation analysis |
| **Metrics** | Recovery time (seconds), throughput during degraded state (%), request error rate during failover, steady-state vs degraded p99 latency |

---

### 4.5 — GPU-Aware Load Balancer

| Field | Detail |
|---|---|
| **Goal** | Build a load balancer that routes requests based on GPU memory utilization, queue depth, and KV cache occupancy. |
| **Tools** | Python (+ optional Rust for hot path), Ray, NVIDIA DCGM for GPU metrics |
| **Deliverables** | Load balancer with pluggable routing strategies (round-robin, least-loaded, KV-cache-aware), load test comparison |
| **Metrics** | p99 latency under load, throughput improvement vs round-robin (%), tail latency reduction, load imbalance across workers |

---

## Phase 5: Benchmarking, Profiling, and Performance Modeling

**Objective:** Develop rigorous benchmarking methodology, build profiling workflows, and create
performance models that predict system behavior before deployment.

---

### 5.1 — Benchmarking Methodology and Harness

| Field | Detail |
|---|---|
| **Goal** | Build a standardized benchmarking framework used across all projects. Define warmup protocol, iteration counts, statistical reporting, and reproducibility requirements. |
| **Tools** | Python, pytest-benchmark (optional), CUDA events, `torch.cuda.Event` |
| **Deliverables** | `benchmark/` module with: timing utilities, statistical summary (mean, std, p50, p95, p99), hardware info collection (GPU model, driver, CUDA version), JSON result output for comparison |
| **Metrics** | Coefficient of variation across runs (target: <5%), reproducibility across re-runs |
| **Why this matters** | Every number in this portfolio must be trustworthy. A benchmark harness ensures that. |

---

### 5.2 — GPU Profiling Workflow Guide

| Field | Detail |
|---|---|
| **Goal** | Create a reference profiling workflow: how to use Nsight Systems for timeline analysis, Nsight Compute for kernel-level metrics, and torch.profiler for PyTorch-level analysis. |
| **Tools** | Nsight Systems, Nsight Compute, torch.profiler, tensorboard |
| **Deliverables** | Step-by-step profiling guide with real examples from Phase 1-3 projects, annotated Nsight screenshots, common bottleneck patterns and how to identify them |
| **Metrics** | N/A — this is a methodology deliverable |

---

### 5.3 — Throughput-Latency Performance Model

| Field | Detail |
|---|---|
| **Goal** | Build a Python simulator that predicts LLM serving throughput and latency given: model size, GPU count, batch size, sequence length, quantization level, and network bandwidth. |
| **Tools** | Python, NumPy, matplotlib |
| **Deliverables** | Performance model with validated predictions, calibration against real benchmarks from Phase 3, interactive parameter sweep charts |
| **Metrics** | Prediction error vs actual benchmarks (<15% target), ability to predict optimal batch size, break-even point for adding GPUs |

---

### 5.4 — Scaling Laws Experiment

| Field | Detail |
|---|---|
| **Goal** | Empirically measure how inference throughput and latency scale with: GPU count (1→8), batch size (1→256), sequence length (128→32K), and model size (1B→70B). Produce scaling curves. |
| **Tools** | vLLM or TensorRT-LLM, matplotlib, benchmark harness from 5.1 |
| **Deliverables** | Scaling curves with analysis, identification of scaling bottlenecks (memory wall, communication overhead, pipeline bubbles) |
| **Metrics** | Throughput scaling efficiency (%), latency scaling behavior, optimal operating points per configuration |

---

### 5.5 — End-to-End System Benchmark Report

| Field | Detail |
|---|---|
| **Goal** | Produce a comprehensive benchmark report combining all optimizations from Phases 1–4. Show the cumulative impact of kernel fusion + quantization + KV cache optimization + continuous batching + tensor parallelism. |
| **Tools** | All tools from prior phases, LaTeX or Markdown for report |
| **Deliverables** | Published benchmark report with reproducible results, comparison against vLLM/TGI baselines, hardware cost analysis |
| **Metrics** | Tokens/sec/dollar, p99 latency at target throughput, memory efficiency (tokens served per GB), total speedup vs naive baseline |

---

## Proposed Directory Structure

```
llm-performance/
├── README.md
├── ROADMAP.md
├── requirements.txt
├── benchmark/                          # Shared benchmarking utilities (Phase 5.1)
│   ├── __init__.py
│   ├── timer.py
│   ├── stats.py
│   └── hardware_info.py
│
├── 01-gpu-fundamentals/                # Phase 1
│   ├── 01-cuda-vector-add/
│   ├── 02-cuda-tiled-matmul/
│   ├── 03-nsight-profiling/            # NEW — profiling workflow
│   ├── 04-fused-gemm-bias-relu/
│   ├── 05-memory-bandwidth-roofline/   # NEW — roofline analysis
│   └── 06-warp-reductions/             # NEW — warp primitives
│
├── 02-transformer-kernels/             # Phase 2
│   ├── 01-triton-softmax/
│   ├── 02-triton-layernorm/
│   ├── 03-triton-rope/
│   ├── 04-naive-attention/             # Baseline before Flash Attention
│   ├── 05-flash-attention/             # Critical
│   ├── 06-kernel-replacement-llama/
│   └── 07-group-gemm/
│
├── 03-inference-optimization/          # Phase 3
│   ├── 01-torch-compile-llama8b/
│   ├── 02-kv-cache-paged-attention/    # NEW — critical
│   ├── 03-continuous-batching/
│   ├── 04-speculative-decoding/        # NEW
│   ├── 05-70b-multi-gpu-serving/
│   └── 06-quantization-fp8-int4/
│
├── 04-distributed-systems/             # Phase 4
│   ├── 01-fsdp-training/              # NEW — distributed training
│   ├── 02-tp-pp-comparison/           # NEW — parallelism strategies
│   ├── 03-multi-node-training/        # NEW — scaling beyond 1 node
│   ├── 04-fault-tolerant-cluster/
│   └── 05-gpu-load-balancer/
│
└── 05-benchmarking-profiling/          # Phase 5
    ├── 01-benchmark-harness/           # NEW — standardized benchmarking
    ├── 02-profiling-guide/             # NEW — Nsight workflow
    ├── 03-performance-model/
    ├── 04-scaling-experiments/         # NEW
    └── 05-system-benchmark-report/     # NEW — capstone
```

---

## Execution Order and Time Estimates

| Phase | Projects | Estimated Time | Prerequisites |
|---|---|---|---|
| **Phase 1** | 1.1–1.6 | 6–8 weeks | None |
| **Phase 2** | 2.1–2.6 | 6–8 weeks | Phase 1 |
| **Phase 3** | 3.1–3.6 | 8–10 weeks | Phase 2 |
| **Phase 4** | 4.1–4.5 | 8–10 weeks | Phase 3 |
| **Phase 5** | 5.1–5.5 | 4–6 weeks | Phases 1–4 (5.1 should be started in Phase 1) |

**Total: ~32–42 weeks** for the complete portfolio.

**Note:** Phase 5.1 (benchmark harness) should be built early in Phase 1 and used throughout.

---

## Key Additions Compared to Current Roadmap

| Addition | Why It Matters |
|---|---|
| **Nsight profiling project** | Profiling is the #1 skill gap. Every optimization starts with a profile. |
| **Roofline model analysis** | Distinguishing compute-bound vs memory-bound is fundamental to choosing the right optimization. |
| **Warp-level primitives** | Required for writing high-performance reductions, which appear in every real kernel. |
| **Flash Attention** | The most impactful kernel innovation in transformers. Not including it is a gap. |
| **KV cache / Paged Attention** | The core of vLLM's breakthrough. This is what inference engineers work on daily. |
| **Speculative decoding** | Active area of research and deployment at every major lab. |
| **Distributed training (FSDP)** | Training at scale is a core requirement, not just serving. |
| **TP/PP comparison** | Understanding parallelism tradeoffs is essential for system design. |
| **Benchmark harness** | Without rigorous methodology, no benchmark result is credible. |
| **Scaling experiments** | Demonstrates systems thinking — understanding how components interact at scale. |

---

## Advanced Project Cross-Reference

These 12 advanced projects map to specific roadmap items:

| # | Advanced Project | Roadmap Item |
|---|---|---|
| 1 | CUDA Matrix Multiplication Optimization | 1.2 Tiled Matmul, 1.4 Fused GEMM |
| 2 | GPU Memory Bandwidth Benchmark | 1.5 Memory Bandwidth Roofline |
| 3 | Transformer Attention Kernel | 2.4 Naive Attention Kernel |
| 4 | Flash Attention Implementation | 2.5 Flash Attention |
| 5 | KV Cache Optimization | 3.2 KV Cache + Paged Attention |
| 6 | Minimal LLM Inference Engine | 3.3 Minimal LLM Inference Engine |
| 7 | Speculative Decoding Engine | 3.4 Speculative Decoding |
| 8 | Transformer GPU Profiling | 1.3 Nsight Profiling, 5.2 Profiling Guide |
| 9 | Kernel Fusion Optimization | 1.4 Fused GEMM + Bias + ReLU |
| 10 | Tensor Parallel Transformer | 4.2 TP/PP Comparison |
| 11 | Distributed Inference Benchmark | 5.4 Scaling Experiments |
| 12 | LLM Serving System | 3.5 70B Serving, 4.4 Fault Tolerant Cluster |

---

## Capstone: Mini-vLLM — LLM Inference Engine from Scratch

This is the project that elevates the portfolio from "good exercises" to "this person can build
real infrastructure." Build a simplified but functional LLM inference engine that integrates
every skill from Phases 1–5 into a single system.

Systems like vLLM achieve 10–24x throughput improvements over naive serving by optimizing
memory management and KV cache handling. Building even a simplified version demonstrates
the kind of end-to-end systems thinking that frontier labs hire for.

### Architecture

```
┌─────────────────────────────────────────────────┐
│                   API Server                     │
│              (HTTP + streaming)                   │
├─────────────────────────────────────────────────┤
│                   Scheduler                      │
│   ┌─────────────┐  ┌──────────────────────┐     │
│   │ Request Queue│  │ Admission Controller │     │
│   └──────┬──────┘  └──────────┬───────────┘     │
│          │                    │                   │
│   ┌──────▼────────────────────▼───────────┐     │
│   │        Continuous Batching Engine       │     │
│   │  (iteration-level, not request-level)  │     │
│   └──────────────────┬─────────────────────┘     │
├──────────────────────┼──────────────────────────┤
│                 GPU Worker                        │
│   ┌──────────────────▼─────────────────────┐     │
│   │           KV Cache Manager              │     │
│   │  ┌────────────┐  ┌──────────────────┐  │     │
│   │  │ Page Table  │  │ Block Allocator  │  │     │
│   │  └────────────┘  └──────────────────┘  │     │
│   └──────────────────┬─────────────────────┘     │
│   ┌──────────────────▼─────────────────────┐     │
│   │         Model Execution Engine          │     │
│   │   (prefill + decode, custom kernels)    │     │
│   └────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

### Components

| Component | Responsibility |
|---|---|
| **API Server** | Accept requests via HTTP, stream tokens back as they are generated |
| **Scheduler** | Manage request queue, decide which requests to run each iteration, handle preemption |
| **Continuous Batcher** | Group prefill and decode requests into efficient batches, re-batch every iteration |
| **KV Cache Manager** | Paged block allocation for KV cache, track per-request page tables, reclaim memory on completion |
| **GPU Worker** | Execute the model forward pass, manage CUDA streams, report memory state back to scheduler |
| **Model Runner** | Load the model, run prefill (prompt processing) and decode (token generation) steps, plug in custom Triton kernels |

### Key Design Decisions

- **Paged KV cache** — fixed-size blocks allocated on demand, eliminating memory fragmentation and waste
- **Iteration-level scheduling** — re-evaluate the batch every decode step, not just at request arrival
- **Prefill/decode separation** — prefill is compute-bound (process full prompt), decode is memory-bound (generate one token) — schedule them differently
- **Preemption** — when memory is exhausted, swap or recompute lower-priority requests
- **Streaming output** — tokens returned to the client as they are generated, not after full completion

### Deliverables

- Working inference engine serving a 7B model on a single GPU
- Paged KV cache with block allocator and page tables
- Continuous batching scheduler with admission control
- HTTP API with streaming token output
- Load test harness with synthetic traffic (Poisson arrivals, variable prompt/generation lengths)
- Benchmark comparison against HuggingFace `generate()` baseline

### Metrics

| Metric | Target |
|---|---|
| Throughput (tokens/sec) | ≥3x vs naive `generate()` at high concurrency |
| Time-to-first-token (TTFT) p99 | <500ms at 10 concurrent requests |
| Memory utilization | >90% KV cache occupancy under load (minimal waste) |
| Max concurrent requests | 4x+ vs static batching at same GPU memory |
| Token streaming latency | First token streamed within one decode step |

### What This Proves

- You understand the full inference serving stack, not just individual kernels
- You can integrate KV cache, batching, scheduling, and GPU execution into a working system
- You can reason about memory-bound vs compute-bound phases and schedule accordingly
- You can build something that works under real load, not just on toy benchmarks

---

## README Integrity Rule

**Do not list a project as complete or benchmarked in the main README until the code, benchmarks,
and analysis are committed and reproducible.** Placeholder folders should be listed as "Planned"
with no performance claims. This is the most important credibility signal for hiring managers
reviewing the portfolio.
