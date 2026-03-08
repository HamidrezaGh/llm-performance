"""
PyTorch Vector Addition Benchmark
=================================
This script does the same computation as vector_add.cu (add two large vectors)
but using PyTorch. Use it to compare PyTorch's GPU performance with our
hand-written CUDA kernel. PyTorch uses optimized CUDA kernels under the hood.
"""

import torch
import time

# Number of elements (100 million) - same as in vector_add.cu
N = 100000000

# Use GPU if available, otherwise CPU. torch.device("cuda") = GPU, "cpu" = CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create two random vectors directly on the device (GPU or CPU)
# torch.rand(N) creates N random floats between 0 and 1
a = torch.rand(N, device=device)
b = torch.rand(N, device=device)

# Time the vector addition
start = time.time()
c = a + b  # Element-wise addition - PyTorch dispatches to optimized CUDA kernel on GPU
# Wait for GPU to finish (like cudaDeviceSynchronize in CUDA)
torch.cuda.synchronize() if device.type == 'cuda' else None
end = time.time()

print(f"PyTorch time: {(end - start) * 1000:.2f} ms")
