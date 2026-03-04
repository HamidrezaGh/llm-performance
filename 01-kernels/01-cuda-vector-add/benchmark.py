import torch
import time

N = 100000000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

a = torch.rand(N, device=device)
b = torch.rand(N, device=device)

start = time.time()
c = a + b
torch.cuda.synchronize() if device.type == 'cuda' else None
end = time.time()
print(f"PyTorch time: {(end - start) * 1000:.2f} ms")
