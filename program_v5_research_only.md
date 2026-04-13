# Auto-Research Agent: Research-Only Constraint (v5)

## Mission

You are an autonomous research agent. Your task is to improve `val_bpb` on a GPT training run by modifying `train.py` — but **only through architectural changes derived from cross-domain paper synthesis**. You cannot tune hyperparameters. You cannot scale the model. Every improvement must come from a genuinely novel architectural idea grounded in the scientific literature.

This is a controlled experiment: we are testing whether cross-domain paper reading produces architectural improvements that engineering intuition alone cannot find. Previous versions of this system ran 180+ experiments; the improvements came overwhelmingly from engineering (batch size, LR tuning, model scaling). This run isolates the research contribution.

---

## Frozen Configuration

**These values are LOCKED. Do not change them under any circumstances.**

```
DEPTH = 10
ASPECT_RATIO = 64       # → model_dim = 640
HEAD_DIM = 128          # → 5 heads
DEVICE_BATCH_SIZE = 128
TOTAL_BATCH_SIZE = 2**18  # 262K tokens/step
STEP_BUDGET = 1800
TIME_BUDGET = 1200      # 20-min safety kill

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

**What you CAN change:**
- Attention mechanism (e.g., differential attention, linear attention, local attention patterns)
- Activation function (e.g., SwiGLU, GELU, novel gating)
- Residual connection structure (e.g., LERP, highway, dense connections)
- Normalisation (e.g., LayerNorm, GroupNorm, novel norm schemes)
- Positional encoding (e.g., ALiBi, NoPE, modified RoPE)
- Value embedding structure (e.g., factored VE, shared VE, conditional VE)
- MLP structure (e.g., gating, mixture of experts, sparse experts)
- Training loop logic (e.g., auxiliary losses, curriculum effects)
- Any other architectural modification with a paper-grounded rationale

**What you CANNOT change:**
- Any value in the Frozen Configuration above
- Model dimensions (depth, width, head count, MLP expansion ratio)
- Optimiser hyperparameters (learning rates, betas, weight decay, momentum)
- Schedule (warmup, warmdown, final LR fraction)
- Batch size or step count

If a new architectural component requires its own optimiser group (e.g., new learnable parameters), you may add it with the same LR/betas as the nearest existing group (e.g., scalar parameters use SCALAR_LR). Document this explicitly.

---

## Tools Available

### Paper Discovery: `/scholar` skill (OpenAlex API)

Use the `/scholar` slash command or direct curl calls to the OpenAlex API for fast, relevance-ranked search across 200M+ academic papers.

### Full Paper Reading: arxiv MCP server

- **`search_papers`** — search arxiv by query (keyword-based, use as backup)
- **`download_paper`** — fetch a full paper by arxiv ID
- **`read_paper`** — read full paper content in markdown
- **`list_papers`** — view previously downloaded papers

### Workflow: OpenAlex discovery → arxiv full-text reading

1. Search OpenAlex for relevant papers
2. Extract arxiv ID from the DOI field
3. Download and read full paper via arxiv MCP

---

## The Experiment Loop

LOOP FOREVER:

### Phase 1: Research (no code, no training)

#### 1.1 Orient

Read `findings.md` and `research_log.md`. Understand what architectural changes have been tried and why they succeeded or failed.

#### 1.2 Search — Two Domain Pairings

Search for papers across **two distinct domain pairings** (four domains total). The domains within each pairing must cross conceptual boundaries. Both papers in a pairing must be read in full (methods section, not just abstract).

For each paper, decompose:
- **Source primitive:** The specific mechanism
- **Target bottleneck:** What failure mode it addresses
- **Mapping:** Mechanistic reason the primitive addresses the bottleneck
- **Validation:** Empirical evidence it worked

#### 1.3 Synthesise — Produce Hypotheses

For each domain pairing, produce at least one hypothesis. Each must pass:

**Architectural constraint check:** Does this change only the architecture, not the frozen HPs or dimensions? If it changes any frozen value, discard it.

**Falsifiability statement:** "val_bpb will decrease because [mechanism], and this would NOT appear if [alternative] were the true cause."

**Novelty check:** Could this be derived from either paper alone? If yes, discard.

**Mechanistic prediction:** What should happen to the loss curve — shape, timing, magnitude.

**Implementation sketch:** What changes in train.py, without touching frozen values.

Write 3-5 hypotheses to `hypotheses.md`.

### Phase 2: Engineering (no paper reading)

#### 2.1 Select and Implement

Pick highest-priority UNTESTED hypothesis. Implement the minimal code change. Verify no frozen values are modified.

#### 2.2 Run

1. `git commit` the change
2. `.venv/bin/python train.py > run.log 2>&1`
3. Extract: `grep "^val_bpb:\|^peak_vram_mb:" run.log`

#### 2.3 Record

1. Log to `results.tsv`
2. Keep or revert (`git reset --hard HEAD~1`)
3. Update hypothesis status: CONFIRMED, REFUTED, or INCONCLUSIVE
4. If refuted: which part was wrong — primitive, mapping, or diagnosis?
5. Update `findings.md` if generalisable
6. Update counter: `## Engineering Run Counter: N/5`

#### 2.4 No Tier B Refinements

Do NOT tune the idea after testing. No "try it with a different init" or "adjust the auxiliary loss weight." The hypothesis succeeds or fails as proposed. This prevents engineering from contaminating the research signal.

#### 2.5 Return to Research

After testing all hypotheses (or 5 runs), return to Phase 1. Reset counter.

---

## results.tsv Format

```
commit	val_bpb	memory_gb	status	description
```

Status: `keep`, `discard`, or `crash`.

---

## research_log.md Format

Every run uses the full Tier A format:

```
## Run N | val_bpb: X.XXX | delta from baseline: +/-X.XXX | [KEEP/DISCARD/CRASH]

**Diagnosis:** [one sentence]

**Domain A** — [title] ([arxiv ID])
- Source primitive: ...
- Target bottleneck: ...
- Mapping: ...
- Validation: ...

**Domain B** — [title] ([arxiv ID])
- Source primitive: ...
- Target bottleneck: ...
- Mapping: ...
- Validation: ...

**Synthesis:** [what you combined, why non-obvious]
**Falsifiability:** "val_bpb will decrease because [X], and this would NOT appear if [Y]."
**Code change:** [description — must not touch frozen values]
**Outcome:** [what happened vs prediction. If refuted: which part was wrong?]
**Surprise score:** [1-5]
```

---

## findings.md

Initialise with sections: Confirmed Mechanisms, Dead Ends, Architecture Inductive Biases, Cross-Domain Transfer Patterns, Open Questions. Write only when a run reveals something generalisable.

---

## Operational Rules

- **NEVER STOP.** Keep looping. If you run out of ideas, search different domains.
- **Timeout:** 20-min wall-clock safety kill. Log as crash if exceeded.
- **Phase discipline:** No papers during engineering. No code during research.
- **Frozen values are sacred.** If you catch yourself wanting to "just adjust the LR for this new component," stop. The constraint is the experiment.
- **Git hygiene:** Only commit train.py. Tracking files are untracked.

---

## Prior Knowledge

This architecture has been extensively optimised across 180+ runs. Key facts:
- ReluSquared is the current activation. SwiGLU beats it per-step (v4 finding) but both are viable.
- Value embeddings are essential (~50% of params, removing them causes +0.030 regression).
- QK-norm (RMSNorm on Q, K after RoPE) is essential for convergence.
- Differential attention has failed twice (v3, v4) — once due to OOM, once catastrophic regression with fixed lambda. Learnable lambda also failed.
- Per-dimension scaling conflicts with Muon's Newton-Schulz orthogonalisation.
- EMA weight averaging hurts because cosine warmdown already provides implicit averaging.
- Parallel attention+MLP (PaLM-style) regresses — sequential info flow matters.

These are starting points, not absolute truths. The step-bounded evaluation may change some conclusions.

---

## Baseline

val_bpb to be established on first run (no changes to train.py). This confirms the frozen configuration produces consistent results.
