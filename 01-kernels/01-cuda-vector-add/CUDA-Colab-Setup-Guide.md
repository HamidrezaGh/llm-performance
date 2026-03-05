# CUDA Development Setup Guide for Google Colab

This guide explains how to compile and run CUDA (.cu) files in Google Colab.

## Step-by-Step Setup

### 1. Open Google Colab
Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2. Enable GPU Runtime
- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** = **GPU** (preferably A100 or T4)
- Click Save

### 3. Clone Your Repository
Run this in the first cell:
```python
!git clone https://github.com/HamidrezaGh/llm-performance.git
%cd llm-performance

### 4. Install CUDA Compiler (nvcc)

Run this command:
Python!apt-get update
!apt-get install -y cuda-toolkit

### 5. Compile a CUDA File
Example for Week 1:
Python!nvcc -o vector_add 01-kernels/01-cuda-vector-add/vector_add.cu

### 6. Run the Compiled Program
Python!./vector_add

### 7. Run Python Scripts
Python!python 01-kernels/01-cuda-vector-add/benchmark.py

## Tips

Re-run the apt-get install if you get "nvcc not found"
Use !nvidia-smi to check GPU status
For better timing, always use cudaEvent in your kernels (as shown in Week 1)
Restart runtime if you get CUDA version errors
