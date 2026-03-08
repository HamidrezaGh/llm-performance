# 05 — GPU-Aware Load Balancer

**Status:** Planned

**Goal:** Build a load balancer that routes requests based on GPU memory utilization, queue depth, and KV cache occupancy.

**Tools:** Python, Ray, NVIDIA DCGM

**Deliverables:**
- Load balancer with pluggable routing strategies (round-robin, least-loaded, KV-cache-aware)
- Load test comparison across strategies

**Metrics:**
- p99 latency under load
- Throughput improvement vs round-robin (%)
- Tail latency reduction
- Load imbalance across workers
