# Capstone — Mini-vLLM: LLM Inference Engine from Scratch

**Status:** Planned

**Goal:** Build a simplified but functional LLM inference engine that integrates every skill from Phases 1–5. This is a mini-vLLM: paged KV cache, continuous batching, streaming token output, a scheduler, and a GPU worker — all from scratch.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   API Server                     │
│              (HTTP + streaming)                   │
├─────────────────────────────────────────────────┤
│                   Scheduler                      │
│   ┌─────────────┐  ┌──────────────────────┐     │
│   │ Request Queue│  │ Admission Controller │     │
│   └──────┬──────┘  └──────────┬───────────┘     │
│          │                    │                   │
│   ┌──────▼────────────────────▼───────────┐     │
│   │        Continuous Batching Engine       │     │
│   │  (iteration-level, not request-level)  │     │
│   └──────────────────┬─────────────────────┘     │
├──────────────────────┼──────────────────────────┤
│                 GPU Worker                        │
│   ┌──────────────────▼─────────────────────┐     │
│   │           KV Cache Manager              │     │
│   │  ┌────────────┐  ┌──────────────────┐  │     │
│   │  │ Page Table  │  │ Block Allocator  │  │     │
│   │  └────────────┘  └──────────────────┘  │     │
│   └──────────────────┬─────────────────────┘     │
│   ┌──────────────────▼─────────────────────┐     │
│   │         Model Execution Engine          │     │
│   │   (prefill + decode, custom kernels)    │     │
│   └────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

## Components

| Component | Responsibility |
|---|---|
| **API Server** | Accept requests via HTTP, stream tokens back as generated |
| **Scheduler** | Manage request queue, decide which requests run each iteration, handle preemption |
| **Continuous Batcher** | Group prefill and decode requests into efficient batches, re-batch every iteration |
| **KV Cache Manager** | Paged block allocation, per-request page tables, reclaim memory on completion |
| **GPU Worker** | Execute model forward pass, manage CUDA streams, report memory state to scheduler |
| **Model Runner** | Load model, run prefill and decode steps, plug in custom Triton kernels |

## Tools

Python, PyTorch, Triton, asyncio, FastAPI or aiohttp

## Deliverables

- Working inference engine serving a 7B model on a single GPU
- Paged KV cache with block allocator and page tables
- Continuous batching scheduler with admission control
- HTTP API with streaming token output
- Load test harness (Poisson arrivals, variable prompt/generation lengths)
- Benchmark comparison against HuggingFace `generate()` baseline

## Metrics

| Metric | Target |
|---|---|
| Throughput (tokens/sec) | >=3x vs `generate()` at high concurrency |
| TTFT p99 | <500ms at 10 concurrent requests |
| Memory utilization | >90% KV cache occupancy under load |
| Max concurrent requests | 4x+ vs static batching at same GPU memory |
| Token streaming latency | First token within one decode step |

## Prerequisites

Phases 1–4 of the roadmap. This project ties everything together.
