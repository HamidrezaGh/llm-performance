# 05 — Transformer Block Assembly: Custom Kernels in Llama

**Status:** Planned

**Goal:** Replace PyTorch's default softmax, LayerNorm, and RoPE in a Llama model with the custom Triton kernels from projects 2.1-2.3. Measure end-to-end impact.

**Tools:** PyTorch, Triton, HuggingFace Transformers

**Deliverables:**
- Modified Llama forward pass with custom kernels
- A/B benchmark: stock vs custom kernels
- Correctness validation (logit comparison)

**Metrics:**
- End-to-end inference latency (ms)
- Tokens/sec
- Per-layer kernel time breakdown
- Max logit difference vs reference
