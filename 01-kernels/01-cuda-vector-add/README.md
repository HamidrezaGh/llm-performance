# 01 - High-Performance CUDA Vector Addition Kernel

**Goal**: Implement and benchmark a basic CUDA kernel for vector addition against CPU baseline.

**Hardware**: Google Colab (A100 GPU)

**Results**:

| Implementation     | Time (ms)   | Speedup      |
|--------------------|-------------|--------------|
| CPU (Sequential)   | 320.20 ms   | 1×           |
| CUDA Kernel        | 4.81 ms     | **66.6×**    |

**Key Takeaways**:
- Successfully implemented grid-stride loop design for large vectors (100M elements)
- Demonstrated significant performance gain using GPU parallelization
- Proper memory allocation and data transfer between host and device

**Techniques Used**:
- CUDA kernel launch with optimal block/thread configuration
- CUDA event-based accurate timing
- Memory coalescing considerations

**Next Steps** (Week 2):
- Add PyTorch comparison
- Nsight Systems profiling + memory analysis
- Shared memory optimization (tiled matrix multiplication)

**Files**:
- `vector_add.cu` → Main CUDA implementation