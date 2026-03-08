# 06 — Quantization: FP8 and INT4 Inference

**Status:** Planned

**Goal:** Benchmark FP8 and INT4 quantized inference. Compare GPTQ, AWQ, and bitsandbytes approaches. Write a custom Triton dequantization kernel. Analyze accuracy vs throughput tradeoffs.

**Tools:** Triton, bitsandbytes, auto-gptq, autoawq, PyTorch

**Deliverables:**
- Quantized model benchmarks across methods
- Perplexity comparison (FP16 vs FP8 vs INT4)
- Custom Triton dequantization kernel
- Memory footprint analysis

**Metrics:**
- Tokens/sec per quantization method
- Memory reduction (%)
- Perplexity degradation
- Peak memory usage (GB)
