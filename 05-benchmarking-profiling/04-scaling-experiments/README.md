# 04 — Scaling Experiments

**Status:** Planned

**Goal:** Empirically measure how inference throughput and latency scale with GPU count (1-8), batch size (1-256), sequence length (128-32K), and model size (1B-70B). Produce scaling curves.

**Tools:** vLLM or TensorRT-LLM, matplotlib, benchmark harness

**Deliverables:**
- Scaling curves with analysis
- Identification of scaling bottlenecks (memory wall, communication overhead, pipeline bubbles)
- Optimal operating point recommendations

**Metrics:**
- Throughput scaling efficiency (%)
- Latency scaling behavior
- Optimal batch size per configuration
