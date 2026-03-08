# 01 — Triton Softmax Kernel

**Status:** Planned

**Goal:** Write a numerically stable fused softmax in Triton. Compare against PyTorch's `F.softmax`. Understand online softmax — the foundation of Flash Attention.

**Tools:** Triton, PyTorch, Nsight Systems

**Deliverables:**
- `triton_softmax.py`
- Benchmark notebook
- Numerical correctness tests (max absolute error vs reference)

**Metrics:**
- Execution time (ms) vs PyTorch baseline
- Memory bandwidth utilization (%)
- Speedup across input sizes
