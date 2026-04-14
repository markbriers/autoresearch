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
task.md                     — YOUR task definition (write this — see below)
train.py                    — GPT training script (default target for the example)
prepare.py                  — data prep and evaluation utilities (GPT-specific)
program.md                  — reference documentation

.claude/
  agents/
    researcher.md           — Researcher subagent (generic protocol, reads task.md)
    evaluator.md            — Evaluator subagent (generic protocol, reads task.md)
    implementer.md          — Implementer subagent (generic protocol, reads task.md)
  commands/
    setup-task.md           — interactive task definition wizard (/setup-task)
    run-cycle.md            — cycle orchestrator (/run-cycle)
    scholar.md              — OpenAlex paper search skill (/scholar)

run_loop.sh                 — external bash loop for unattended runs

examples/
  gpt-training/
    task.md                 — worked example: GPT val_bpb optimisation
    README.md               — pointer to full write-up

docs/
  blog-post.md              — full write-up (~5000 words)
  gpu-cloud-setup.md        — RunPod H100 setup guide
  plans/                    — design and implementation plans

results/                    — historical results from GPT experiment (v1-v9)
```

## Quick start

### Using the GPT example (Karpathy's benchmark)

```bash
# Standard setup
uv sync
uv run prepare.py

# Copy the example task definition
cp examples/gpt-training/task.md ./task.md

# Add arxiv MCP for paper reading
uv tool install arxiv-mcp-server
claude mcp add arxiv $(which arxiv-mcp-server) -- --storage-path ~/.arxiv-mcp-server/papers

# Single cycle (interactive)
claude -p "/run-cycle"

# Unattended (loops until you touch STOP)
chmod +x run_loop.sh
nohup ./run_loop.sh &
```

### Using your own task

The easiest way to get started is the interactive wizard:

```bash
claude
# Then type: /setup-task
```

This walks you through six questions about your task and generates a validated `task.md`. Alternatively, write one manually with six required sections (see `examples/gpt-training/task.md` for reference):

1. **Objective** — command to run, metric to extract, direction (min/max)
2. **Intervention Space** — what the agent can modify, what is frozen
3. **Evaluation** — noise floor, confirmation threshold, resource constraints
4. **Domain Context** — prior knowledge about your problem domain
5. **Subsystem Taxonomy** — categories for tracking interventions
6. **Cross-Domain Requirement** — your field (Domain A) and what counts as outside (Domain B)

Then run `/run-cycle` or `run_loop.sh` as above. The subagents read `task.md` at the start of each cycle and adapt to your task automatically.

For cloud GPU setup (RunPod H100), see [docs/gpu-cloud-setup.md](docs/gpu-cloud-setup.md).

## Design choices (extending Karpathy's)

- **Domain-agnostic protocol.** The three subagents and orchestrator are generic. All domain-specific content lives in `task.md`, which the user writes. The same protocol works for ML training, compiler optimisation, molecular simulation, or anything else with a measurable objective.
- **Three subagents, not one.** Researcher generates, Implementer tests, Evaluator judges. The agent that proposes an idea cannot judge whether it worked.
- **Cross-domain enforcement.** Each hypothesis must cite one paper from the user's field and one from outside. The user defines what "outside" means in `task.md`. This prevents "engineering with citations."
- **Bayesian surprise.** Independent P(success) estimates from Researcher and Evaluator. Belief divergence prioritises hypotheses where the agents disagree. High-surprisal results get exploration directives.
- **Sprint contracts with hard thresholds.** Success criteria are defined before implementation, not after. The confirmation threshold is set by the user based on their noise floor.
- **External bash loop.** Fresh Claude Code session per cycle. Zero shared context between cycles. Files are the only persistent memory.

## Upstream

This is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The original train.py, prepare.py, and benchmark design are Karpathy's work (MIT licence). This fork adds the methodology, literature access, subagent definitions, results, and write-up.

## Licence

MIT
