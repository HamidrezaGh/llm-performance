# 03 — Throughput-Latency Performance Model

**Status:** Planned

**Goal:** Build a Python simulator that predicts LLM serving throughput and latency given model size, GPU count, batch size, sequence length, quantization level, and network bandwidth.

**Tools:** Python, NumPy, matplotlib

**Deliverables:**
- Performance model with validated predictions
- Calibration against real benchmarks from Phase 3
- Interactive parameter sweep charts

**Metrics:**
- Prediction error vs actual benchmarks (<15% target)
- Ability to predict optimal batch size
- Break-even point for adding GPUs
