# 01 — Distributed Training with FSDP

**Status:** Planned

**Goal:** Fine-tune a 7B model using PyTorch FSDP across multiple GPUs. Understand sharding strategies, activation checkpointing, and communication overhead.

**Tools:** PyTorch FSDP, NCCL, Nsight Systems, torchrun

**Deliverables:**
- FSDP training script
- Scaling analysis (1/2/4/8 GPUs)
- Nsight timeline showing communication vs compute overlap

**Metrics:**
- Training throughput (tokens/sec)
- GPU utilization (%)
- Communication overhead (%)
- Memory per GPU (GB)
- Scaling efficiency (%)
