# 05 — 70B Model Multi-GPU Serving

**Status:** Planned

**Goal:** Serve Llama-2-70B (or Llama-3-70B) across 4x A100 GPUs using tensor parallelism. Measure real serving performance under load.

**Tools:** vLLM or TensorRT-LLM, Ray, NCCL, Nsight Systems

**Deliverables:**
- Deployment scripts
- Throughput dashboard (tokens/sec, p95, p99)
- Nsight timeline showing inter-GPU communication
- Load test results

**Metrics:**
- Tokens/sec at batch_size = {1, 8, 32}
- p99 latency (ms)
- GPU utilization (%)
- Inter-GPU communication overhead (%)
- Scaling efficiency: 1-GPU vs 4-GPU
