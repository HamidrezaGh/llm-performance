# LLM Systems Performance Engineering Portfolio

### CUDA • Triton • Distributed Inference • Kernel Debugging • Quantization

**Objective:** Engineer high-throughput, low-latency, fault-tolerant LLM serving systems optimized for multi-GPU A100/H100 clusters.

**Scope:**
24 production-grade GPU + distributed systems implementations
Real A100 benchmarks
Kernel-level profiling (Nsight, eBPF)
Open-source contributions

**Target Roles:** Performance Engineer / LLM Systems Engineer @
Anthropic • OpenAI • Google DeepMind • Meta

---

# 🔥 Performance Highlights (Benchmarked)

* ✔ 3×+ throughput improvements via kernel fusion
* ✔ p99 latency reduction through CUDA stream overlap
* ✔ Custom Triton transformer ops replacing PyTorch baselines
* ✔ Continuous batching engine outperforming naive inference ≥2.5×
* ✔ 70B model multi-GPU serving with fault injection testing
* ✔ eBPF-based network latency root-cause tracing

---

# 📁 Repository Structure

```
/kernels
    /cuda-vector-add
    /cuda-tiled-matmul
    /cuda-fused-gemm
    /triton-softmax
    /triton-layernorm
    /triton-rotary
    /group-gemm

/inference-optimization
    /torch-compile-llama8b
    /triton-in-llama
    /continuous-batching
    /70b-multi-gpu-serving

/distributed-systems
    /multi-gpu-serving
    /custom-load-balancer
    /fault-tolerant-topology

/observability
    /ebpf-latency-debugger

/quantization
    /fp8-kernels
    /int4-kernels
    /quant-aware-training

/performance-modeling
    /throughput-latency-simulator
```

---

# ⚙️ Kernel Engineering (CUDA + Triton)

## CUDA Optimization

* High-performance vector addition (Nsight profiled)
* Tiled matrix multiplication with shared memory
* Fused GEMM with stream-level concurrency
* Memory coalescing + occupancy tuning
* Warp-level reduction strategies

**Focus Areas**

* Global memory bandwidth utilization
* SM occupancy
* Register pressure
* Stream overlap
* Kernel fusion strategies

---

## Triton Kernel Development

* Fused Softmax (numerically stable, block-optimized)
* LayerNorm with memory-efficient reduction
* Rotary embedding kernel
* Group GEMM implementation
* PyTorch extension packaging (setup.py + loadable ops)

**Impact**

* Transformer op replacement inside Llama architecture
* End-to-end latency reduction measured on A100

---

# 🚀 LLM Inference Optimization

## torch.compile Optimization

* Benchmarked Llama-3.1-8B across compile modes
* Analyzed graph breaks + kernel fusion effects
* Measured compile overhead vs inference gains

## Transformer Kernel Replacement

Replaced:

* Softmax
* Rotary embeddings
* LayerNorm

With custom Triton kernels.

**Result:** End-to-end throughput increase under realistic batch workloads.

---

## Continuous Batching Engine (From Scratch)

* Implemented dynamic batching scheduler
* Token-level queue management
* Memory-aware request packing
* Outperformed naive PyTorch baseline ≥2.5×

---

## 70B Multi-GPU Serving

* 4× A100 deployment
* Tensor parallel configuration
* Throughput dashboard (tok/sec, p95, p99)
* Real benchmark publication

---

# 🌐 Distributed Systems & Reliability

## 8-GPU Fault-Tolerant Cluster

* Ray/vLLM cluster orchestration
* Fault injection scenarios
* Node recovery + request redistribution
* Throughput degradation analysis

## Custom Load Balancer

* Python + Rust implementation
* GPU-aware scheduling
* Backpressure control
* Adaptive queue depth

---

# 🔍 Observability & Kernel-Level Debugging

## eBPF Latency Debugger

* Traced container network microbursts
* Identified syscall-level latency spikes
* Visualized per-request kernel timings
* Root-cause isolation methodology

Inspired by real-world AI infra debugging scenarios.

---

# 📉 Quantization & Low-Precision Inference

## FP8 + INT4 Kernels

* Custom Triton low-precision kernels
* Quant-aware benchmarking
* Memory footprint reduction
* Throughput vs accuracy tradeoff analysis

---

# 📊 System Performance Modeling

## Throughput & Latency Simulator

Python-based predictive model:

* GPU count scaling
* Batch size sensitivity
* Token generation rate
* Network bandwidth constraints
* Topology simulation

Used to:

* Predict cluster capacity
* Estimate cost-performance tradeoffs
* Model bottlenecks before deployment

---

# 🧠 Engineering Principles

* Measure first, optimize second
* Always profile (Nsight > guesswork)
* Optimize for p99, not average
* Remove bottlenecks at the system level
* Treat inference like a distributed systems problem

---

# 📈 Open Source Contributions

* PR #1 – vLLM improvement
* PR #2 – PyTorch or vLLM kernel fix
* Public benchmarks + reproducible experiments
* Technical blog series documenting findings

---

# 🎯 Outcome

This portfolio demonstrates:

* GPU kernel engineering capability
* LLM inference optimization
* Distributed systems reliability
* Kernel-level observability
* Quantization & low-precision production readiness

Designed for senior-level AI infrastructure roles at:

Anthropic
OpenAI
Google DeepMind
Meta
NVIDIA
