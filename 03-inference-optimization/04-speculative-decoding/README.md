# 04 — Speculative Decoding

**Status:** Planned

**Goal:** Implement speculative decoding using a draft model to accelerate autoregressive generation. Analyze acceptance rates, draft length tuning, and when speculation helps vs hurts.

**Tools:** PyTorch, HuggingFace Transformers

**Deliverables:**
- Speculative decoding implementation with configurable draft length
- Benchmark across draft model sizes
- Acceptance rate analysis

**Metrics:**
- Tokens/sec speedup vs vanilla autoregressive
- Acceptance rate (%)
- Optimal draft length
- Latency at batch_size = 1
