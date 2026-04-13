# autoresearch: from engineering to research

A fork of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) that extends the autonomous ML experiment loop with academic literature access, cross-domain hypothesis synthesis, and a three-subagent protocol designed to make a coding agent behave more like a researcher than an engineer.

The original autoresearch gives a coding agent a GPT training script and lets it optimise val_bpb autonomously. This fork asks: what happens when you also give it access to the scientific literature and force it to read papers from outside machine learning?

The full write-up is in [docs/blog-post.md](docs/blog-post.md).

## What this fork adds

### Three-subagent protocol (.claude/agents/)

Three Claude Code subagents with isolated context windows and restricted tool access:

- **Researcher** (.claude/agents/researcher.md) — searches papers via OpenAlex and arxiv MCP, generates cross-domain hypotheses as sprint contracts with quantified success criteria. Must pair one ML paper with one non-ML paper (signal processing, neuroscience, control theory, etc.). Includes adversarial self-critique and P(success) estimates.

- **Evaluator** (.claude/agents/evaluator.md) — reviews sprint contracts, judges results against hard thresholds (val_bpb delta < -0.003 for CONFIRMED, default INCONCLUSIVE). No Bash, no web access. Computes independent P(success) estimates, belief divergence, surprisal scores. Maintains a subsystem tracker with pivot directives and prediction calibration notes. The knowledge curator of the programme.

- **Implementer** (.claude/agents/implementer.md) — implements approved contracts, runs training, records raw numbers. No web access, no paper reading, no self-assessment. Cannot write CONFIRMED/REFUTED.

### Orchestrator and loop

- `.claude/commands/run-v7.md` — orchestrates one cycle: Research, Pre-Eval, Implement, Post-Eval, git snapshot
- `run_v7_loop.sh` — external bash loop for unattended operation, fresh Claude Code session per cycle, exponential backoff on errors, graceful stop via `touch STOP`

### Literature access

- `.claude/commands/scholar.md` — OpenAlex API skill for relevance-ranked search across hundreds of millions of academic works
- arxiv MCP server for full paper reading (search, download, read in markdown)

### Step-bounded evaluation

Replaced Karpathy's wall-clock time budget with a fixed step budget (1800 optimiser steps). This removes the throughput bias that caused the original benchmark to systematically favour sparse/fast activations over dense/slow ones. See the blog post for how this reversed the "ReluSquared > SwiGLU" finding that had survived 117 experiments.

## Results

Nine protocol iterations, 200+ experiments, ~£100 total compute on RunPod H100s. Full results in `results/v1` through `results/v9`.

| Version | Key change | Runs | Confirmed | Notable finding |
|---------|-----------|------|-----------|-----------------|
| v1 | Literature access added | 117 | 0 research | Agent defaults to engineering (3/117 paper-driven) |
| v2 | Phase separation | 48 | 0 research | Better hypotheses, all fail experimentally |
| v3 | Hard counter | 4 | 0 | Throughput bias kills all creative ideas |
| v4 | Step-bounded training | 22 | 0 research | SwiGLU > ReluSquared reversal; wider MLP works |
| v5 | Frozen HPs, research-only | 13 | 3 | Partial RoPE via OFDM, depthwise conv, attn temp |
| v6 | Adversarial gates, belief tracking | 39 | 8 | All known techniques; rediscovery problem |
| v7 | Three subagents | 4 | 3 | Within-ML pairings; cross-domain too loose |
| v8 | Non-ML Domain B enforced | 7 | 3 | ShrinkReLU (wavelet shrinkage), learned softcap (Jaynes) |
| v9 | Bayesian surprise | 17 | 4 | PD residual scaling (control theory); subadditive stacking |

### Key findings

- **Mechanisms are VARIANT-level novel** — novel configurations of known families (PD residuals, ShrinkReLU), not wholly new mechanisms
- **Empirical findings are arguably NOVEL** — 37% subadditive stacking discount for architectural modifications; QK-norm + softcap rigidity against learnable temperature
- **The harness matters more than the model** — most of the work was in prompt constraints, tool restrictions, context isolation, and Bayesian surprise integration
- **Cross-domain enforcement was the single most effective change** — one line requiring non-ML papers shifted hypothesis character immediately

## Repository structure

```
train.py                    — GPT training script (agent modifies this)
prepare.py                  — data prep and evaluation utilities (fixed)
program.md                  — latest agent instructions

.claude/
  agents/
    researcher.md           — Researcher subagent definition
    evaluator.md            — Evaluator subagent definition
    implementer.md          — Implementer subagent definition
  commands/
    run-v7.md               — cycle orchestrator
    scholar.md              — OpenAlex paper search skill

run_v7_loop.sh              — external bash loop for unattended runs

docs/
  blog-post.md              — full write-up (~5000 words)
  gpu-cloud-setup.md        — RunPod H100 setup guide
  plans/                    — implementation plans

results/
  v1/                       — v1 results (single loop, MPS + H100)
  v2/                       — v2 results (phase separation)
  v3/                       — v3 results (hard counter)
  v4/                       — v4 results (step-bounded)
  v5/                       — v5 results (frozen HPs, research-only)
  v7/                       — v7 results (three subagents)
  v8/                       — v8 results (cross-domain enforcement)
  v9/                       — v9 results (Bayesian surprise)
```

## Quick start

Follow Karpathy's original setup, then add the literature tools:

```bash
# Standard setup
uv sync
uv run prepare.py

# Add arxiv MCP
uv tool install arxiv-mcp-server
claude mcp add arxiv $(which arxiv-mcp-server) -- --storage-path ~/.arxiv-mcp-server/papers

# Single cycle (interactive)
claude -p "/run-v7"

# Unattended (loops until you touch STOP)
chmod +x run_v7_loop.sh
nohup ./run_v7_loop.sh &
```

For cloud GPU setup (RunPod H100), see [docs/gpu-cloud-setup.md](docs/gpu-cloud-setup.md).

## Design choices (extending Karpathy's)

- **Step-bounded, not time-bounded.** Fixed 1800 optimiser steps instead of 5-minute wall clock. Removes throughput bias that confounds architectural comparison.
- **Frozen hyperparameters.** From v5 onwards, all HPs are locked. The agent can only modify architecture. This isolates the research contribution from engineering.
- **Three subagents, not one.** Researcher generates, Implementer tests, Evaluator judges. The agent that proposes an idea cannot judge whether it worked.
- **Cross-domain enforcement.** Each hypothesis must cite one ML paper and one paper from outside ML/DL/optimisation. This prevents "engineering with citations."
- **Bayesian surprise.** Independent P(success) estimates from Researcher and Evaluator. Belief divergence prioritises hypotheses where the agents disagree. High-surprisal results get exploration directives.
- **External bash loop.** Fresh Claude Code session per cycle. Zero shared context between cycles. Files are the only persistent memory.

## Upstream

This is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The original train.py, prepare.py, and benchmark design are Karpathy's work (MIT licence). This fork adds the methodology, literature access, subagent definitions, results, and write-up.

## Licence

MIT
