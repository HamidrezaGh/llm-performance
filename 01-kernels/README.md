# 🚀 CUDA & NVIDIA GPU Architecture – Fundamentals

**This file is the main learning source for CUDA.** Read it first to build your mental model before diving into the hands-on labs in the subfolders.

---

# 📂 Labs in This Folder

| Lab | Topic |
|-----|-------|
| [`01-cuda-vector-add/`](./01-cuda-vector-add/) | First CUDA kernel – vector addition |
| [`02-cuda-tiled-matmul/`](./02-cuda-tiled-matmul/) | Tiled matrix multiplication |
| [`03-cuda-fused-gemm/`](./03-cuda-fused-gemm/) | Fused GEMM (matrix multiply) |
| [`04-triton-softmax/`](./04-triton-softmax/) | Triton – softmax kernel |
| [`05-triton-layernorm/`](./05-triton-layernorm/) | Triton – layer normalization |
| [`06-triton-rotary/`](./06-triton-rotary/) | Triton – rotary embeddings |
| [`07-group-gemm/`](./07-group-gemm/) | Grouped GEMM |

---

# 0️⃣ What is CUDA? (Software Basics)

**CUDA** (Compute Unified Device Architecture) is NVIDIA's platform for programming GPUs. It extends C/C++ with:

* **Kernels** – functions that run on the GPU (marked with `__global__`)
* **APIs** – `cudaMalloc`, `cudaMemcpy`, `cudaFree` to manage GPU memory
* **Launch syntax** – `kernel<<<blocks, threads>>>(args)` to run code in parallel

## Host vs Device

| Term | Meaning |
|------|---------|
| **Host** | The CPU and its RAM. Your `main()` runs here. |
| **Device** | The GPU and its memory. Kernels run here. |

Data lives in separate spaces. You must **copy** data between host and device explicitly.

## Typical CUDA Program Flow

```text
1. Allocate memory on HOST    (malloc)
2. Allocate memory on DEVICE (cudaMalloc)
3. Copy HOST → DEVICE       (cudaMemcpy)
4. Launch kernel on GPU     (kernel<<<...>>>)
5. Copy DEVICE → HOST       (cudaMemcpy)
6. Free all memory          (cudaFree, free)
```

The first lab ([`01-cuda-vector-add/`](./01-cuda-vector-add/)) implements this exact flow with detailed comments.

---

# 1️⃣ Big Picture

NVIDIA GPU architecture has:

* **Hardware Architecture (Physical components)**
* **CUDA Execution Model (Software abstraction)**
* **Memory Hierarchy**
* **Thread Organization Model**

---

# 🏗 2️⃣ Hardware Architecture (Physical GPU)

```text
NVIDIA GPU
│
├── Host Interface
│     ├── PCIe / NVLink
│     └── Copy Engines (DMA)
│
├── Global Memory System
│     ├── HBM / GDDR Memory
│     ├── Memory Controllers
│     └── L2 Cache (shared by all SMs)
│
├── GPCs (Graphics Processing Clusters)
│     └── SMs (Streaming Multiprocessors)
│
└── Display / Graphics Engine (if applicable)
```

---

# 🔥 3️⃣ Streaming Multiprocessor (SM) – The Core Compute Unit

The **SM is the main compute engine** of the GPU.

```text
Streaming Multiprocessor (SM)
│
├── Warp Scheduler(s)
├── Dispatch Units
│
├── Execution Units
│     ├── CUDA Cores (FP32 / INT32 ALUs)
│     ├── FP64 Units
│     ├── Tensor Cores (Matrix / AI)
│     ├── SFUs (Special Function Units)
│     └── Load/Store Units
│
├── Register File (per thread)
├── Shared Memory (per block)
├── L1 Cache (often unified with shared memory)
└── Instruction Cache
```

Key idea:

> The SM executes **warps**, not individual threads.

---

# 🧠 4️⃣ CUDA Execution Model (Software Hierarchy)

CUDA organizes parallelism like this:

```text
Grid
 ├── Block 0
 │     ├── Warp 0
 │     │     ├── Thread 0
 │     │     ├── Thread 1
 │     │     └── ...
 │     ├── Warp 1
 │     └── ...
 ├── Block 1
 └── ...
```

---

## 🔹 Thread

* Smallest execution unit
* Has its own registers
* Executes instructions

---

## 🔹 Warp (32 Threads)

* 32 threads executed in lockstep
* Smallest scheduling unit in hardware
* Executed by warp scheduler

Formula:

```cpp
warpId = threadId / 32;
laneId = threadId % 32;
```

---

## 🔹 Block (Thread Block)

* Group of threads
* Runs entirely on one SM
* Can use shared memory
* Can synchronize (`__syncthreads()`)

Blocks are **independent scheduling units**.

---

## 🔹 Grid

* All blocks launched by a kernel
* Entire workload of one kernel launch

Kernel launch:

```cpp
kernel<<<numBlocks, threadsPerBlock>>>();
```

---

# 🗂 5️⃣ Memory Hierarchy (Fast → Slow)

```text
Registers          (per thread, inside SM)  ⚡ fastest
    ↓
Shared Memory      (per block, inside SM)
    ↓
L1 Cache           (per SM)
    ↓
L2 Cache           (shared across GPU)
    ↓
Global Memory      (HBM / GDDR)
    ↓
Host Memory        (CPU RAM)
```

---

## 🔹 Memory Scope Summary

| Memory Type   | Scope      | Lifetime |
| ------------- | ---------- | -------- |
| Registers     | Per thread | Thread   |
| Shared Memory | Per block  | Block    |
| L1 Cache      | Per SM     | Dynamic  |
| L2 Cache      | Whole GPU  | Dynamic  |
| Global Memory | Whole GPU  | Kernel   |

---

# ⚙️ 6️⃣ Execution Flow (What Happens When You Launch a Kernel)

```text
CPU launches kernel
      ↓
Grid created
      ↓
Blocks distributed to SMs
      ↓
Each block split into warps
      ↓
Warp scheduler picks ready warps
      ↓
Execution units execute instructions
```

Important:

* SM executes **warps**
* Blocks stay on one SM until finished
* Multiple blocks can reside on one SM

---

# 📊 7️⃣ Parallelism Levels

| Level        | Description                  |
| ------------ | ---------------------------- |
| Thread-Level | Individual execution         |
| Warp-Level   | SIMD execution of 32 threads |
| Block-Level  | Cooperative execution        |
| Grid-Level   | Massive parallel workload    |
| Multi-SM     | True hardware parallelism    |

---

# 🧩 8️⃣ Key Architectural Concepts

### 🔹 SIMT (Single Instruction, Multiple Threads)

* Threads execute independently
* But warp executes same instruction at a time

---

### 🔹 Warp Divergence

If threads in a warp take different branches:

```cpp
if (threadIdx.x % 2 == 0)
```

Warp must serialize execution → performance drop.

---

### 🔹 Occupancy

Number of active warps per SM.

Limited by:

* Registers
* Shared memory
* Max threads per SM
* Max blocks per SM

---

### 🔹 Latency Hiding

GPU hides memory latency by:

* Switching between warps
* Running many warps per SM

---

# 🏭 9️⃣ Conceptual Hardware + Software Mapping

```text
GPU
 ├── SM 0
 │     ├── Block A
 │     │     ├── Warp 0
 │     │     └── Warp 1
 │     └── Block B
 │
 ├── SM 1
 │     └── Block C
 │
 └── Shared L2 Cache
```

---

# 🧮 🔟 Summary Cheat Sheet

| Concept       | Hardware? | Description              |
| ------------- | --------- | ------------------------ |
| GPU           | Yes       | Whole device             |
| SM            | Yes       | Compute unit             |
| CUDA Core     | Yes       | ALU                      |
| Tensor Core   | Yes       | Matrix engine            |
| Thread        | No        | Smallest execution unit  |
| Warp          | Semi      | 32-thread execution unit |
| Block         | No        | Group of threads         |
| Grid          | No        | Kernel workload          |
| Shared Memory | Yes       | On-chip fast memory      |
| Global Memory | Yes       | Off-chip DRAM            |

---

# 🏁 Final Mental Model

Think of it like a factory:

* GPU = Factory
* SM = Workshop
* Block = Job assigned to a workshop
* Warp = 32 workers moving in sync
* Thread = Individual worker
* Shared memory = Local whiteboard in workshop
* Global memory = Warehouse
