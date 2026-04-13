# GPU Cloud Setup Guide for Autoresearch

## Overview

Run the autoresearch experiment loop on a cloud H100 with Claude Code (Max plan).
This gives ~10x more training steps per 5-minute run, torch.compile, Flash Attention 3,
and the full architecture design space (no MPS constraints).

**Estimated cost:** £2.50-3.50/hr for the GPU instance. Claude Code uses your Max plan subscription.

---

## Option A: RunPod (recommended — simplest)

### 1. Create account and add SSH key

1. Sign up at https://runpod.io
2. Go to **Settings → SSH Keys**, add your public key (`~/.ssh/id_ed25519.pub`)

### 2. Launch a pod

Via the web UI:
1. **Pods → Deploy**
2. Select **H100 SXM** or **H100 PCIe** (80 GB, ~$2.50/hr)
3. Template: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
4. Volume: 50 GB (enough for code + cached data)
5. Click **Deploy On-Demand**

Or via CLI:
```bash
pip install runpodctl
runpodctl config --apikey YOUR_RUNPOD_API_KEY
runpodctl pod create --name autoresearch --gpu "NVIDIA H100 SXM" \
  --image runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 \
  --volumeSize 50
```

### 3. SSH in

The SSH command is shown in the RunPod dashboard under **Connect → SSH**.
It will look something like:
```bash
ssh root@<pod-id>.runpod.io -p <port> -i ~/.ssh/id_ed25519
```

---

## Option B: Lambda Labs

### 1. Create account and add SSH key

1. Sign up at https://lambda.ai
2. Go to **Settings → SSH Keys**, add your public key

### 2. Launch an instance

1. **Instances → Launch Instance**
2. Select **1x H100** (~$2.49/hr)
3. Choose region, attach SSH key, click **Launch**

### 3. SSH in

```bash
ssh ubuntu@<INSTANCE_IP>
```

IP shown in the Lambda dashboard once running.

---

## Option C: GCP (if you already have GPU quota)

**Warning:** If you've never used GPUs on GCP, you likely have zero GPU quota.
Request it via Console → IAM & Admin → Quotas → search "GPUs (all regions)".
Approval can take hours to days.

```bash
gcloud compute instances create autoresearch-gpu \
  --zone=us-central1-a \
  --machine-type=a3-highgpu-1g \
  --maintenance-policy=TERMINATE \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True"
```

```bash
gcloud compute ssh autoresearch-gpu --zone=us-central1-a
```

---

## Once you're SSH'd in (all providers)

### 4. Install system dependencies

```bash
# Node.js (for Claude Code)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo bash -
sudo apt-get install -y nodejs

# uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 5. Install Claude Code and authenticate

```bash
npm install -g @anthropic-ai/claude-code

# Authenticate with your Anthropic Max plan account
# This will give you a URL to open in your browser
claude auth login
```

### 6. Clone the repo and install Python dependencies

```bash
git clone <YOUR_REPO_URL> autoresearch
cd autoresearch
git checkout autoresearch/mar23

# Install Python deps — the CUDA torch index works on Linux
uv sync
```

### 7. Upgrade PyTorch to 2.9.1 (templates ship 2.8)

```bash
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### 8. Verify GPU

```bash
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name()}'); print(f'CUDA: {torch.version.cuda}'); print(f'torch: {torch.__version__}')"
```

Expected output:
```
GPU: NVIDIA H100 80GB HBM3
CUDA: 12.8
torch: 2.9.1+cu128
```

### 9. Reset train.py to the original CUDA version

The branch has our MPS-patched `train.py`. For the H100, revert to the original
and re-apply only the confirmed findings:

```bash
# Get the original CUDA train.py
git checkout master -- train.py

# Verify it runs (quick sanity check — kill after ~30 seconds)
timeout 30 python train.py || true
```

### 10. Apply confirmed findings from MPS experiments

Edit `train.py` to apply the two changes that transferred:

```python
# Line ~536: reduce warmdown
WARMDOWN_RATIO = 0.3    # was 0.5 — confirmed better with limited steps

# In init_weights(), change x0_lambdas init:
self.x0_lambdas.fill_(0.2)  # was 0.1 — confirmed better for gradient flow
```

Commit this as the H100 baseline:
```bash
git add train.py
git commit -m "feat: apply MPS-confirmed findings (warmdown 0.3, x0_lambdas 0.2)"
```

### 11. Install the arxiv MCP server

```bash
uv tool install arxiv-mcp-server
```

Then add it to Claude Code:
```bash
claude mcp add arxiv $(which arxiv-mcp-server) -- --storage-path ~/.arxiv-mcp-server/papers
```

### 12. Run the baseline and start the autonomous loop

```bash
claude --dangerously-skip-permissions
```

Paste this prompt:

> Read program.md and start the experiment loop. This is an H100 GPU — torch.compile
> and Flash Attention 3 are available. Use python (not python3.12) to run training.
> The CUDA train.py is active with DEPTH=8, TOTAL_BATCH_SIZE=2^19.
> Run the baseline first, then begin the research loop.
> Use the /scholar skill for paper discovery and arxiv MCP for full-text reading.
> NEVER STOP.

### 13. Detach and let it run

Use `tmux` or `screen` so the session persists when you disconnect:

```bash
# Before launching Claude Code:
tmux new -s research

# Inside tmux, run claude
claude --dangerously-skip-permissions

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t research
```

---

## Monitoring from your Mac

The tracking files are in git. From your local machine:

```bash
# Pull results periodically
cd autoresearch
git fetch origin autoresearch/mar23
git diff autoresearch/mar23 -- results.tsv research_log.md findings.md
```

Or SSH in and read them directly:
```bash
ssh <your-gpu-box> "cat ~/autoresearch/results.tsv"
ssh <your-gpu-box> "tail -30 ~/autoresearch/research_log.md"
```

---

## Shutting down

**Don't forget to stop the instance when done!**

- RunPod: Dashboard → Pods → Stop/Terminate
- Lambda: Dashboard → Instances → Terminate
- GCP: `gcloud compute instances stop autoresearch-gpu --zone=us-central1-a`

At ~$2.50/hr, leaving it running accidentally costs ~$60/day.

---

## Key differences: H100 vs MPS

| | MPS (M-series Mac) | H100 |
|---|---|---|
| DEPTH | 4 (reduced) | 8 (original) |
| Params | 11.5M | ~50M |
| Steps/5min | ~320 | ~950 |
| TOTAL_BATCH_SIZE | 2^16 (65K) | 2^19 (524K) |
| torch.compile | No | Yes |
| Flash Attention 3 | No (SDPA) | Yes |
| Sliding window | No effect | Works (SSSL pattern) |
| VRAM | 26 GB shared | 80 GB dedicated |
| Manual attention | 40% overhead | Negligible |
| HEAD_DIM flexibility | 128 only | Any |
