# 02 - Tiled CUDA Matrix Multiplication (Shared Memory)

## Goal
Implement and benchmark a tiled CUDA matrix multiplication kernel using `__shared__` memory against:
- a naive CUDA kernel
- PyTorch (`A @ B`)

## Current Configuration
- Matrix size: `N = 2048` (defined in both `matrix_mul.cu` and `benchmark.py`)
- Tile size: `TILE_SIZE = 32`
- Kernels in `matrix_mul.cu`:
  - `naiveMatMul`
  - `tiledMatMul`

## Files
- `matrix_mul.cu`: naive + tiled CUDA kernels, plus CUDA event timing
- `benchmark.py`: PyTorch baseline timing

## Example Results (Google Colab A100)
These values are sample numbers from one run and will vary by GPU, driver, and runtime state.

| Implementation | Time (ms) | Relative Speed |
|---|---:|---:|
| Naive CUDA | 168.39 | 1.0x |
| Tiled CUDA (`__shared__`) | 26.48 | 6.4x vs naive |
| PyTorch (`@`) | 134.71 | 5.1x slower than tiled |

## Run in Colab
```bash
!git clone https://github.com/HamidrezaGh/llm-performance.git
%cd llm-performance

!nvcc -O3 -o matrix_mul 01-kernels/02-cuda-tiled-matmul/matrix_mul.cu -Wno-deprecated-gpu-targets
!./matrix_mul
!python 01-kernels/02-cuda-tiled-matmul/benchmark.py
```

## Run Locally (Linux/macOS with CUDA)
```bash
cd llm-performance
nvcc -O3 -o matrix_mul 01-kernels/02-cuda-tiled-matmul/matrix_mul.cu -Wno-deprecated-gpu-targets
./matrix_mul
python3 01-kernels/02-cuda-tiled-matmul/benchmark.py
```

## Notes
- Timing in `matrix_mul.cu` uses CUDA events.
- The current CUDA program reports runtime only; it does not validate numerical correctness between kernels.
- For reproducible comparisons, run multiple iterations and discard first-run warm-up.
