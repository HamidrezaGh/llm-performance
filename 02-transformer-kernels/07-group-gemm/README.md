# 06 — Group GEMM / Batched GEMM for MoE

**Status:** Planned

**Goal:** Implement grouped/batched matrix multiplication as used in Mixture-of-Experts layers. Understand expert routing and load balancing at the kernel level.

**Tools:** Triton or CUDA C++, cuBLAS batched GEMM (baseline)

**Deliverables:**
- `group_gemm.py`
- Benchmark across expert counts (8, 16, 64)
- Comparison vs cuBLAS batched GEMM

**Metrics:**
- GFLOPS per expert count
- Load imbalance overhead
- Latency vs cuBLAS baseline
