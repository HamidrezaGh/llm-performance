# 01 - CUDA Vector Addition Kernel

**Goal**: Implement a high-performance vector addition kernel and benchmark against CPU baseline.

**Results**:
- CUDA Kernel: **XX.XX ms**
- CPU Baseline: **XXX.XX ms**
- **Speedup**: **X.X×**

**Hardware Used**: A100 (via Colab / Vast.ai)

**Key Techniques**:
- Grid-stride loop design
- Proper memory allocation and transfer
- CUDA event-based timing

**Next Steps**:
- Add PyTorch comparison
- Nsight Systems profiling
- Memory coalescing optimization (Week 2)

**Files**:
- `vector_add.cu` → Main CUDA kernel