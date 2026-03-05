# CUDA Development Setup Guide for Google Colab

This guide explains how to compile and run CUDA (.cu) files in Google Colab.

## Step-by-Step Setup

### 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2. Enable GPU Runtime
- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** = **GPU** (preferably T4 or A100)
- Click **Save**

### 3. Clone Your Repository
Run this in the first cell:

```python
!git clone https://github.com/HamidrezaGh/llm-performance.git
%cd llm-performance
```

### 4. Install CUDA Compiler (nvcc) — Optional
Colab usually has `nvcc` pre-installed. If you get "nvcc not found", run:

```python
!apt-get update
!apt-get install -y nvidia-cuda-toolkit
```

### 5. Compile the CUDA File
```python
!nvcc -o vector_add 01-kernels/01-cuda-vector-add/vector_add.cu
```

### 6. Run the Compiled Program
```python
!./vector_add
```

### 7. Run the Python Benchmark
Install PyTorch first (Colab includes it, but ensure it's available), then run:

```python
!pip install torch -q
!python 01-kernels/01-cuda-vector-add/benchmark.py
```

## Tips

- Re-run the `apt-get install` step if you get "nvcc not found"
- Use `!nvidia-smi` to check GPU status
- For better timing, use `cudaEvent` in your kernels instead of `clock()`
- Restart runtime (**Runtime** → **Restart runtime**) if you get CUDA version errors
