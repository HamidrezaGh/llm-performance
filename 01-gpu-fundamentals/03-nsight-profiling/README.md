# 03 — GPU Profiling with Nsight Systems and Nsight Compute

**Status:** Planned

**Goal:** Profile the kernels from projects 1.1 and 1.2 using Nsight Systems (timeline analysis) and Nsight Compute (kernel-level metrics). Build a roofline plot and determine whether each kernel is compute-bound or memory-bound.

**Tools:** Nsight Systems (`nsys`), Nsight Compute (`ncu`)

**Deliverables:**
- Nsight timeline captures for vector-add and tiled-matmul
- Roofline analysis chart
- Written analysis: compute-bound vs memory-bound classification
- SM occupancy and memory bandwidth reports

**Metrics:**
- SM occupancy (%)
- Achieved memory bandwidth vs theoretical peak (%)
- Arithmetic intensity (FLOP/byte)
- L1/L2 cache hit rates
