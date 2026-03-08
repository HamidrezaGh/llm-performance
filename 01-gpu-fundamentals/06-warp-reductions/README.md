# 06 — Warp-Level Primitives and Reductions

**Status:** Planned

**Goal:** Implement parallel reduction (sum, max) using warp shuffle intrinsics (`__shfl_down_sync`). Compare naive shared-memory reduction vs warp-shuffle reduction. Analyze warp divergence cost.

**Tools:** CUDA C++, Nsight Compute

**Deliverables:**
- Reduction kernels: naive → shared memory → warp shuffle
- Benchmark across array sizes
- Warp divergence analysis

**Metrics:**
- Execution time per reduction approach
- Bandwidth utilization (%)
- Warp execution efficiency (%)
