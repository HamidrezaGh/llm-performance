# Remote GPU Pod Setup (RunPod + Mac)

This guide is for developing on a Mac while running code on a remote NVIDIA GPU pod.

## 0) What runs where

- Run on your Mac: SSH connection commands, VS Code Remote SSH.
- Run on the pod: install packages, create venv, run benchmarks, train/inference code.

## 1) One-time setup on your Mac

1. Ensure you have an SSH key:

```bash
ls ~/.ssh/id_ed25519 ~/.ssh/id_ed25519.pub
```

2. Copy your public key:

```bash
pbcopy < ~/.ssh/id_ed25519.pub
```

3. In RunPod, add this key in `User Settings -> SSH Public Keys`.

## 2) Create the pod

1. Create a new pod from a PyTorch/CUDA image.
2. Enable SSH access.
3. Attach persistent storage (volume disk or network volume).
4. Start the pod and open `Connect -> SSH`.
5. Copy the exact SSH command shown by RunPod.

Important: keep your repo under `/workspace` so it survives pod stop/start.

Example format:

```bash
ssh <pod-user>@ssh.runpod.io -p <port> -i ~/.ssh/id_ed25519
```

## 3) First connection and bootstrap

Run on your Mac:

```bash
ssh -A <pod-user>@ssh.runpod.io -p <port> -i ~/.ssh/id_ed25519
```

Then run on the pod:

```bash
apt-get update && apt-get install -y git tmux python3-venv
cd /workspace
git clone https://github.com/HamidrezaGh/llm-performance.git
cd llm-performance
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install numpy
```

Why HTTPS clone first: avoids GitHub SSH key issues inside the pod. You can switch to SSH later if needed.

## 4) Sanity checks

Run on the pod:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `True <GPU_NAME>`.

## 5) Daily workflow

1. Connect from Mac:

```bash
ssh -A <pod-user>@ssh.runpod.io -p <port> -i ~/.ssh/id_ed25519
```

2. Activate project on pod:

```bash
cd /workspace/llm-performance
source .venv/bin/activate
```

3. Use `tmux` for long jobs:

```bash
tmux new -s llm
```

Detach with `Ctrl-b` then `d`, reattach with:

```bash
tmux attach -t llm
```

## 6) Run the first CUDA kernel (vector add)

Run on the pod:

```bash
cd /workspace/llm-performance
source .venv/bin/activate
cd 01-gpu-fundamentals/01-cuda-vector-add
nvcc -O3 -o vector_add vector_add.cu -Wno-deprecated-gpu-targets
./vector_add
python benchmark.py
```

## 7) Optional: push to GitHub from pod

Option A (recommended): stay on HTTPS remote and use GitHub CLI auth.

```bash
gh auth login
git remote -v
```

Option B: use SSH push with agent forwarding (`ssh -A`), then test:

```bash
ssh -T git@github.com
```

## 8) Sync code between local, GitHub, and pod

Use GitHub as the single sync point.

Default flow:

- local -> GitHub -> pod
- pod -> GitHub -> local

Update pod with local changes:

Run on local machine:

```bash
cd /Users/hamidrezaghaderi/Documents/repos/llm-performance
git checkout <branch>
git add .
git commit -m "your message"
git push
```

Run on pod:

```bash
cd /workspace/llm-performance
git fetch origin
git checkout <branch>
git pull --ff-only origin <branch>
```

If you made changes on pod:

Run on pod:

```bash
cd /workspace/llm-performance
git checkout <branch>
git add .
git commit -m "pod changes"
git push
```

Run on local machine:

```bash
cd /Users/hamidrezaghaderi/Documents/repos/llm-performance
git checkout <branch>
git pull --ff-only
```

Safety rules:

- Always run `git status` before pull/push.
- Do not edit the same file on both machines before syncing.
- Keep pod repo at `/workspace/llm-performance` to preserve data across stop/start.

## 9) VS Code Remote SSH (optional)

Add to `~/.ssh/config` on Mac:

```sshconfig
Host llm-gpu
  HostName ssh.runpod.io
  User <pod-user>
  Port <port>
  IdentityFile ~/.ssh/id_ed25519
  ForwardAgent yes
  StrictHostKeyChecking accept-new
```

Then connect in VS Code Remote SSH to `llm-gpu`.

## 10) Stop vs terminate (cost control)

- `Exit SSH only`: pod keeps running, billing continues.
- `Stop pod`: all running processes are terminated. Compute billing stops.
- `Stop pod`: data not stored on volume disk is lost.
- `Stop pod`: files under `/workspace` are preserved if `/workspace` is on your volume disk.
- `Stop pod`: idle disk billing continues (usually much cheaper than active GPU billing).
- `Terminate pod`: pod resources are removed. Separate persistent volumes may still bill until deleted.

Before stopping or terminating:

1. Commit and push important changes.
2. Confirm files are on persistent storage.

## 11) Quick troubleshooting

- `Permission denied (publickey)` when cloning from GitHub:
  - Use HTTPS clone, or configure GitHub auth in pod.
- `requirements.txt not found`:
  - You are in the wrong directory. Run `cd /workspace/llm-performance`.
- `torch.cuda.is_available() == False`:
  - Wrong pod image or GPU not attached. Check in RunPod dashboard.
- `nvcc: command not found`:
  - Use a CUDA development image or install the CUDA toolkit in the pod.
