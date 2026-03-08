# LLM Performance Engineering

**CUDA | Triton | Inference Optimization | Distributed Systems | GPU Profiling**

A hands-on portfolio of GPU performance engineering projects, progressing from low-level CUDA kernels to production LLM inference and distributed training systems.

**Target roles:** Performance Engineer / LLM Systems Engineer at Anthropic, OpenAI, Google DeepMind, Meta, NVIDIA

See [`ROADMAP.md`](./ROADMAP.md) for the full project plan with goals, deliverables, and metrics for each project.

---

## Completed Projects

### CUDA Vector Addition — 66x Speedup over CPU
Custom CUDA kernel for 100M-element vector addition. Grid-stride loop, explicit memory management, CUDA event timing.

| Implementation | Time (ms) | Speedup vs CPU |
|---|---:|---:|
| CPU (sequential) | 320.20 | 1x |
| PyTorch | 51.24 | 6.25x |
| **CUDA kernel** | **4.81** | **66.6x** |

Hardware: A100 (Google Colab) | [`01-gpu-fundamentals/01-cuda-vector-add/`](./01-gpu-fundamentals/01-cuda-vector-add/)

---

### Tiled Matrix Multiplication — 6.4x Speedup via Shared Memory
Tiled GEMM using `__shared__` memory vs naive CUDA kernel. Demonstrates memory coalescing and tile-size optimization.

| Implementation | Time (ms) | Relative Speed |
|---|---:|---:|
| Naive CUDA | 168.39 | 1.0x |
| Tiled CUDA (`__shared__`) | 26.48 | 6.4x vs naive |
| PyTorch (`@`) | 134.71 | 5.1x slower than tiled |

Hardware: A100 (Google Colab) | [`01-gpu-fundamentals/02-cuda-tiled-matmul/`](./01-gpu-fundamentals/02-cuda-tiled-matmul/)

---

### torch.compile on Llama-3.1-8B-Instruct
Baseline vs compiled inference performance using `torch.compile`. First LLM inference benchmark in the portfolio. 4-bit quantization with bitsandbytes.

Hardware: A100 (Google Colab) | [`03-inference-optimization/01-torch-compile-llama8b/`](./03-inference-optimization/01-torch-compile-llama8b/)

---

## Repository Structure

```
01-gpu-fundamentals/              Phase 1: GPU Architecture & CUDA
  01-cuda-vector-add/             ✅ Complete
  02-cuda-tiled-matmul/           ✅ Complete
  03-nsight-profiling/               Planned — GPU profiling workflow
  04-fused-gemm-bias-relu/           Planned — Kernel fusion
  05-memory-bandwidth-roofline/      Planned — Roofline model analysis
  06-warp-reductions/                Planned — Warp shuffle primitives

02-transformer-kernels/           Phase 2: Transformer Internals
  01-triton-softmax/                 Planned — Fused softmax in Triton
  02-triton-layernorm/               Planned — Fused LayerNorm
  03-triton-rope/                    Planned — Rotary embeddings
  04-naive-attention/                Planned — Scaled dot-product attention baseline
  05-flash-attention/                Planned — Flash Attention v2
  06-kernel-replacement-llama/       Planned — Custom kernels in Llama
  07-group-gemm/                     Planned — Batched GEMM for MoE

03-inference-optimization/        Phase 3: Inference Performance
  01-torch-compile-llama8b/       ✅ Partial — needs compile mode sweep
  02-kv-cache-paged-attention/       Planned — PagedAttention
  03-continuous-batching/            Planned — Minimal LLM inference engine
  04-speculative-decoding/           Planned — Draft model acceleration
  05-70b-multi-gpu-serving/          Planned — Tensor parallel serving
  06-quantization-fp8-int4/          Planned — FP8/INT4 quantization

04-distributed-systems/           Phase 4: Distributed LLM Systems
  01-fsdp-training/                  Planned — Distributed fine-tuning
  02-tp-pp-comparison/               Planned — Parallelism tradeoffs
  03-multi-node-training/            Planned — Multi-node scaling
  04-fault-tolerant-cluster/         Planned — Fault injection testing
  05-gpu-load-balancer/              Planned — GPU-aware routing

05-benchmarking-profiling/        Phase 5: Benchmarking & Profiling
  01-benchmark-harness/              Planned — Standardized framework
  02-profiling-guide/                Planned — Nsight workflow guide
  03-performance-model/              Planned — Throughput/latency simulator
  04-scaling-experiments/            Planned — Scaling curves
  05-system-benchmark-report/        Planned — Capstone report

06-capstone-mini-vllm/            Capstone: LLM Inference Engine from Scratch
                                     Planned — Paged KV cache, continuous batching,
                                     streaming tokens, scheduler, GPU worker.
                                     A mini-vLLM tying all phases together.
```

---

## Skills Demonstrated (so far)

- CUDA kernel development (grid-stride loops, shared memory tiling)
- Host/device memory management
- CUDA event-based timing and benchmarking
- GPU performance comparison (CUDA vs PyTorch vs CPU)
- LLM inference with torch.compile and quantization

## Skills In Progress

- GPU profiling (Nsight Systems / Nsight Compute)
- Triton kernel development
- Flash Attention
- KV cache optimization
- Continuous batching
- Distributed training (FSDP, tensor/pipeline parallelism)

---

## Hardware

All benchmarks run on **NVIDIA A100** via Google Colab unless otherwise noted.

## Setup

```bash
git clone https://github.com/HamidrezaGh/llm-performance.git
cd llm-performance
pip install -r requirements.txt
```

See individual project READMEs for specific build and run instructions.

For Mac users developing on remote GPUs, see [`REMOTE_POD_SETUP.md`](./REMOTE_POD_SETUP.md).
