# 04 — Fault-Tolerant Serving Cluster

**Status:** Planned

**Goal:** Deploy an LLM serving cluster with fault injection testing. Implement health checks, automatic failover, and request redistribution when a GPU node goes down.

**Tools:** vLLM, Ray Serve, Python, Docker

**Deliverables:**
- Cluster deployment scripts
- Fault injection test suite (kill node, network partition, OOM)
- Throughput degradation analysis

**Metrics:**
- Recovery time (seconds)
- Throughput during degraded state (%)
- Request error rate during failover
- Steady-state vs degraded p99 latency
