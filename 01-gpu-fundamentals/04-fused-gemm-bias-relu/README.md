# 04 — Fused GEMM + Bias + ReLU Kernel

**Status:** Planned

**Goal:** Fuse matrix multiply, bias addition, and ReLU activation into a single kernel. Demonstrate kernel fusion as a strategy to eliminate intermediate global memory traffic.

**Tools:** CUDA C++, Nsight Compute, CUDA streams

**Deliverables:**
- `fused_gemm.cu` with fused vs unfused comparison
- Nsight Compute profile showing reduced memory transactions
- Stream overlap demonstration

**Metrics:**
- Execution time reduction (%)
- Global memory transactions (fused vs unfused)
- Achieved GFLOPS
- Kernel launch overhead savings
