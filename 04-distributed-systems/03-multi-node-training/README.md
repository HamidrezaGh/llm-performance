# 03 — Multi-Node Training and Network Profiling

**Status:** Planned

**Goal:** Scale training beyond a single node. Profile inter-node communication and identify network bottlenecks. Compare intra-node (NVLink) vs inter-node (InfiniBand) bandwidth.

**Tools:** PyTorch DDP/FSDP, NCCL, nccl-tests, Nsight Systems

**Deliverables:**
- Multi-node training script
- NCCL bandwidth benchmark
- Network latency analysis
- Intra-node vs inter-node comparison

**Metrics:**
- All-reduce bandwidth (GB/s)
- Training throughput at 1-node vs 2-node vs 4-node
- Communication-to-compute ratio
- Scaling efficiency (%)
