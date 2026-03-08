# 03 — Minimal LLM Inference Engine

**Status:** Planned

**Goal:** Build a minimal LLM inference engine from scratch that combines KV cache management, continuous batching, and token streaming into a single coherent system. Implement iteration-level scheduling (not request-level), memory-aware request admission, and async token output.

**Tools:** Python, PyTorch, asyncio

**Deliverables:**
- Inference engine with:
  - KV cache allocator (fixed-size block pool)
  - Continuous batching scheduler with request queue
  - Admission control and preemption support
  - Token-by-token streaming output
- Load test harness with synthetic workloads (varying prompt/generation lengths)
- Comparison vs static batching baseline

**Metrics:**
- Throughput (tokens/sec) at varying request rates
- p50/p95/p99 time-to-first-token (TTFT)
- p50/p95/p99 time-per-output-token (TPOT)
- Max concurrent requests at fixed GPU memory
- Speedup vs static batching (target: >=2x)
