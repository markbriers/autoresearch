# Task Definition: GPT Language Model Training

## Objective

Minimise validation bits-per-byte (val_bpb) on a small GPT language model.

- **Command:** `.venv/bin/python train.py > run.log 2>&1`
- **Metric extraction:** grep for `^val_bpb:` and `^peak_vram_mb:` in run.log
- **Metric name:** val_bpb
- **Direction:** minimise (lower is better)
- **Secondary metrics:** peak VRAM (MB), wall-clock time (seconds)

## Intervention Space

The agent modifies `train.py`, which contains the full GPT model, optimiser (Muon + AdamW), and training loop.

**What you CAN change:** Attention mechanism, activation function, residual connection structure, normalisation, positional encoding, value embedding structure, MLP structure, training loop logic (e.g., auxiliary losses, curriculum effects), or any other architectural modification with a paper-grounded rationale. If a new architectural component requires its own optimiser group, use the same LR/betas as the nearest existing group (e.g., new scalar parameters use SCALAR_LR).

**What you CANNOT change (frozen configuration):**

```
DEPTH = 10
ASPECT_RATIO = 64       # model_dim = 640
HEAD_DIM = 128           # 5 heads
DEVICE_BATCH_SIZE = 128
TOTAL_BATCH_SIZE = 2**18 # 262K tokens/step
STEP_BUDGET = 1800
TIME_BUDGET = 1200       # 20-min safety kill

EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.03
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.1
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.7
FINAL_LR_FRAC = 0.01

WINDOW_PATTERN = "SSSL"
softcap = 12
x0_lambdas init = 0.2
resid_lambdas init = 1.0
```

## Evaluation

- **Noise floor:** approximately 0.002 val_bpb (measured from one seed-variance spot check)
- **Confirmation threshold:** 0.003 (1.5x noise floor). A change must reduce val_bpb by at least 0.003 to be CONFIRMED.
- **INCONCLUSIVE:** delta between -0.003 and +0.001
- **REFUTED:** delta > +0.001 or CRASH (OOM, divergence, torch.compile failure)
- **Resource constraints:**
  - VRAM < 76 GB (baseline uses ~68GB on H100 80GB; leaves headroom)
  - Wall-clock < 1200 seconds (20-minute safety kill)
  - No NaN/Inf in training loss

## Domain Context

This is a 10-layer, 640-dimension GPT model (85.9M parameters) trained with the Muon optimiser (Newton-Schulz orthogonalisation for matrix parameters) and AdamW (for scalar/vector parameters). Training is step-bounded at 1800 optimiser steps with a warmdown schedule.

Key facts from prior experiments (200+ runs):

- ReluSquared is the current activation. SwiGLU beats it per-step. xIELU (Huang 2024) beats both.
- Value embeddings are essential (~50% of params; removing causes +0.030 regression).
- QK-norm (RMSNorm on Q, K after RoPE) is essential for convergence.
- Differential attention has failed three times across multiple versions.
- Per-dimension scaling conflicts with Muon's Newton-Schulz orthogonalisation.
- Auxiliary losses are extremely dangerous with Muon (confirmed twice — catastrophic regression).
- torch.compile is fragile to new scalar parameters, graph changes, and nn.Linear biases.
- Positional encoding changes have minimal effect at 2048 context length.
- EMA weight averaging is redundant with the warmdown schedule.
- The winning pattern from prior work: learnable scaling with identity/near-passthrough initialisation.
- VRAM ceiling after stacked improvements is approximately 71-72GB.
- Confirmed improvements interact subadditively (approximately 37% stacking discount). Some pairs interfere (e.g., SwiGLU + factored embeddings). Others compose well (e.g., PD residuals + factored embeddings).
- QK-norm plus softcap creates a rigid logit regime that resists additional scalar modifications (per-head learnable temperature causes regression).

## Subsystem Taxonomy

Categories for the subsystem tracker. The Evaluator may create fine-grained subcategories as patterns emerge (e.g., "attention/additive-params" vs "attention/gating").

- attention
- activation
- residuals
- positional
- MLP
- normalisation
- embeddings
- training-loop

## Cross-Domain Requirement

- **Domain A (your field):** Machine learning, deep learning, neural network architecture, optimisation
- **Domain B (outside your field):** Signal processing, control theory, information theory, neuroscience, physics, biology, telecommunications, compressed sensing, statistical mechanics, ecology, economics, materials science, fluid dynamics, or any other non-ML discipline

Each hypothesis must pair one paper from Domain A with one from Domain B. Two papers from different ML subfields (e.g., "activation functions" and "attention mechanisms") does NOT count as cross-domain. The Evaluator will reject non-compliant contracts.
