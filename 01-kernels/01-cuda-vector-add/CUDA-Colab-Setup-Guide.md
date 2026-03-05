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

### 4. Compile the CUDA kernel
!nvcc -o vector_add 01-kernels/01-cuda-vector-add/vector_add.cu

### 5. Run it
!./vector_add
