# 02 — Pipeline and Tensor Parallelism Comparison

**Status:** Planned

**Goal:** Implement tensor parallelism (TP) and pipeline parallelism (PP) on a transformer model. Analyze tradeoffs: TP reduces per-layer latency but increases communication; PP reduces memory but introduces pipeline bubbles.

**Tools:** PyTorch, NCCL, Megatron-LM (reference), Nsight Systems

**Deliverables:**
- TP implementation (column/row parallel linear layers)
- PP implementation (micro-batch scheduling)
- Comparison notebook with tradeoff analysis

**Metrics:**
- Throughput (samples/sec) for TP vs PP vs TP+PP
- Pipeline bubble fraction (%)
- Communication volume (GB)
- Memory per GPU (GB)
