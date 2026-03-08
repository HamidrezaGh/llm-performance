# 03 — Triton Rotary Positional Embedding (RoPE) Kernel

**Status:** Planned

**Goal:** Implement RoPE in Triton. Analyze whether positional encoding is compute-bound or memory-bound at different sequence lengths.

**Tools:** Triton, PyTorch

**Deliverables:**
- `triton_rope.py`
- Benchmark across sequence lengths
- Correctness validation against HuggingFace implementation

**Metrics:**
- Latency (ms) at seq_len = {512, 2048, 8192, 32768}
- Speedup vs PyTorch baseline
