# 03 - torch.compile Optimization on Llama-3.1-8B-Instruct

**Goal**  
Measure baseline vs compiled inference performance on Llama-3.1-8B-Instruct using `torch.compile`.  
First real LLM inference benchmark in the portfolio.

**Hardware**  
Google Colab (A100 GPU)

**Setup Instructions**  
See `CUDA-Colab-Setup-Guide.md` in root for general CUDA setup.  
Additional requirements for this week:

```bash
!pip install -q transformers torch accelerate bitsandbytes huggingface_hub

How to run in Colab

Enable GPU runtime (Runtime → Change runtime type → GPU → A100 if available)
Log in to Hugging Face (required for Llama-3.1 gated model):
Go to https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct → Accept license
Create Read token: https://huggingface.co/settings/tokens
In Colab left sidebar → Secrets → Add HF_TOKEN with your token

Run cells in order: install → login → load model → baseline → compiled → results

Results (your run – March 2025)






Run TypeLatency (ms)Tokens/secSpeedupBaseline36,458 ms11.0 TPS1.00×torch.compile~36,458 ms*~11.0 TPS~1.01×
*Note: speedup currently low due to quantization + short warmup. See "Next Steps" for tuning.