# 01 - High-Performance CUDA Vector Addition Kernel

**Goal**: Implement a CUDA kernel for vector addition (100M elements) and benchmark against both PyTorch and CPU baselines.

**Hardware**: Google Colab (A100 GPU)

**Results**:

| Implementation     | Time (ms)    | Speedup vs CPU | Speedup vs PyTorch |
|--------------------|--------------|----------------|--------------------|
| CPU (Sequential)   | 320.20 ms    | 1×             | -                  |
| PyTorch            | 51.24 ms     | 6.25×          | 1×                 |
| **CUDA Kernel**    | **4.81 ms**  | **66.6×**      | **10.7×**          |

**Key Takeaways**:
- Successfully wrote and executed my first CUDA kernel with proper memory management.
- Demonstrated massive performance gains by moving computation to the GPU.
- CUDA significantly outperforms both native Python loops and PyTorch's `+` operator on large vectors.

**Techniques Used**:
- Grid-stride loop for handling large inputs
- CUDA event-based accurate timing
- Explicit host/device memory transfers

**Setup**: See [`CUDA-Colab-Setup-Guide.md`](CUDA-Colab-Setup-Guide.md) in the root for full instructions on running CUDA code in Colab.

**Next Steps (Week 2)**:
- Implement tiled matrix multiplication with shared memory
- Add Nsight Systems profiling
- Analyze memory bandwidth and occupancy

**Files**:
- `vector_add.cu` — Custom CUDA kernel
- `benchmark.py` — PyTorch comparison