# 04 — Naive Attention Kernel (Scaled Dot-Product)

**Status:** Planned

**Goal:** Implement basic scaled dot-product attention from scratch in Triton. Compute Q*K^T, apply causal mask, softmax, and V projection. Observe the O(n^2) memory bottleneck firsthand — this establishes the baseline that Flash Attention improves on.

**Tools:** Triton, PyTorch

**Deliverables:**
- `naive_attention.py` — from-scratch attention kernel
- Memory profiling showing O(n^2) attention matrix materialization
- Benchmark across sequence lengths
- Correctness validation against `F.scaled_dot_product_attention`

**Metrics:**
- Latency (ms) at seq_len = {512, 1K, 2K, 4K, 8K}
- Peak memory usage (MB) — should show quadratic growth
- TFLOPS achieved
- Numerical accuracy vs PyTorch reference
