import torch
import time

N = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A = torch.rand(N, N, device=device)
B = torch.rand(N, N, device=device)

# PyTorch baseline
start = time.time()
C = A @ B
torch.cuda.synchronize()
print(f"PyTorch time: { (time.time() - start)*1000 :.2f} ms")