# 02 — Triton LayerNorm Kernel

**Status:** Planned

**Goal:** Implement fused LayerNorm in Triton with single-pass mean/variance computation.

**Tools:** Triton, PyTorch

**Deliverables:**
- `triton_layernorm.py`
- Comparison against `torch.nn.LayerNorm`
- Correctness tests

**Metrics:**
- Latency (ms) across hidden dimensions (768, 1024, 4096, 8192)
- Throughput (elements/sec)
- Speedup vs PyTorch
