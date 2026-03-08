# 04 — Flash Attention Implementation

**Status:** Planned

**Goal:** Implement Flash Attention v2 in Triton. Understand tiled online softmax, IO-aware algorithm design, and why this is the most impactful kernel optimization for transformers.

**Tools:** Triton, PyTorch, Nsight Compute

**Deliverables:**
- `flash_attention.py` — Triton Flash Attention kernel
- Benchmark against naive attention and `torch.nn.functional.scaled_dot_product_attention`
- Memory usage comparison (Flash vs naive)
- Correctness validation

**Metrics:**
- Latency (ms) at seq_len = {1K, 4K, 16K, 64K}
- Peak memory usage (MB)
- TFLOPS achieved
- Speedup vs naive O(n^2) attention
