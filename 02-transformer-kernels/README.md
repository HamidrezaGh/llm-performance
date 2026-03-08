# Phase 2 — Transformer Internals and Custom Kernels

Write high-performance Triton implementations of core transformer operations and plug them into a real LLM.

| Lab | Topic | Status |
|-----|-------|--------|
| [`01-triton-softmax/`](./01-triton-softmax/) | Fused softmax (numerically stable) | Planned |
| [`02-triton-layernorm/`](./02-triton-layernorm/) | Fused LayerNorm | Planned |
| [`03-triton-rope/`](./03-triton-rope/) | Rotary positional embeddings | Planned |
| [`04-naive-attention/`](./04-naive-attention/) | Naive scaled dot-product attention | Planned |
| [`05-flash-attention/`](./05-flash-attention/) | Flash Attention v2 | Planned |
| [`06-kernel-replacement-llama/`](./06-kernel-replacement-llama/) | Custom kernels inside Llama | Planned |
| [`07-group-gemm/`](./07-group-gemm/) | Grouped GEMM for MoE | Planned |
