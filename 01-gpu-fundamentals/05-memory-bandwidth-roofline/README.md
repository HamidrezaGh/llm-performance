# 05 — Memory Bandwidth Analysis and Roofline Modeling

**Status:** Planned

**Goal:** Build a microbenchmark suite measuring achieved memory bandwidth across different access patterns (coalesced, strided, random). Construct a roofline model for the target GPU.

**Tools:** CUDA C++, Nsight Compute, Python (matplotlib)

**Deliverables:**
- Microbenchmark kernels for each access pattern
- Roofline chart with kernels from prior projects plotted
- Written analysis of bottleneck classification

**Metrics:**
- Achieved bandwidth (GB/s) per access pattern
- % of theoretical peak (e.g., A100 HBM = 2 TB/s)
- Arithmetic intensity for each kernel
