# 02 — KV Cache Optimization and Paged Attention

**Status:** Planned

**Goal:** Implement a KV cache manager with paged memory allocation (inspired by vLLM's PagedAttention). Understand how KV cache is the primary memory bottleneck in autoregressive generation.

**Tools:** PyTorch, Triton, Python

**Deliverables:**
- KV cache allocator with page tables
- Paged attention kernel
- Memory usage comparison: paged vs contiguous allocation
- Benchmark at various sequence lengths and batch sizes

**Metrics:**
- Peak memory usage (GB)
- Memory waste (%)
- Max concurrent sequences at fixed GPU memory
- Tokens/sec at batch_size = {1, 8, 32, 128}
