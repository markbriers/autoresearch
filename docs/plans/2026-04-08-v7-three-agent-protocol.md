# v7 Three-Agent Research Protocol — Implementation Plan (Revised)

> **Revision 2 (2026-04-08):** Switched from agent teams to subagents (sequential workflow, no inter-agent messaging needed). Added enriched Evaluator role (knowledge curator, pattern analysis, calibration tracking). Added two-level context isolation (within-cycle via subagents, between-cycle via external bash loop). See "Differences from Previous Plan" at end.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the autoresearch control protocol using the three-agent pattern from Anthropic's harness design, separating hypothesis generation, implementation, and evaluation into distinct roles with sprint contracts and hard thresholds.

**Architecture:** Three conceptual agents (implemented as three phases within a single Claude Code session, each with distinct instructions and constraints) communicating via structured files. The key change from v6 is that the evaluator defines quantified success criteria *before* the implementer runs, and the evaluator's judgment is final — the implementer cannot self-assess.

**Rationale:** In v1-v6, the same agent generated hypotheses, implemented them, and decided whether they worked. This is the anti-pattern Anthropic identifies: "when asked to evaluate work they've produced, agents tend to respond by confidently praising the work." Our v5 agent called a 0.0002 improvement (within seed variance) "CONFIRMED." A separate evaluator with hard thresholds would have called it INCONCLUSIVE.

---

## Problem Analysis

### What broke in v1-v6

1. **Self-evaluation bias.** The agent that implemented a hypothesis also judged whether it worked. Result: marginal improvements within noise were called "confirmed," and the agent was reluctant to reject its own ideas.

2. **No pre-agreed success criteria.** The only criterion was "does val_bpb decrease?" with no minimum threshold, no statistical bar, no consideration of seed variance (~0.002 val_bpb). Any decrease, however small, counted.

3. **Phase discipline failure.** v3's hard counter was circumvented in v6 (the agent relabelled engineering as research). The counter was enforced by the same agent it constrained — no external party held it accountable.

4. **No exploration forcing after plateau.** When multiple hypotheses targeting the same subsystem (e.g., attention) all failed, the agent kept trying attention modifications rather than pivoting to a different subsystem. No evaluator forced the pivot.

5. **Context self-congratulation.** The agent's accumulated findings.md became increasingly confident over time, recording marginal results as established mechanisms.

### What the Anthropic harness pattern solves

1. **Separated evaluator** catches self-assessment bias.
2. **Sprint contracts** define success before implementation, not after.
3. **Hard thresholds** with criteria-specific failure make INCONCLUSIVE the default, not CONFIRMED.
4. **Evaluator-driven pivots** force exploration when scores plateau.

---

## The Three Phases

### Phase 1: Researcher (generates hypotheses)

Same as v6 Phase 1 (paper search, cross-domain synthesis, adversarial self-critique, prior art search, feasibility pre-flight) with one addition: after generating each hypothesis, the Researcher writes a **sprint contract** defining quantified success criteria.

### Phase 2: Evaluator (defines thresholds, judges results)

New phase. Before the Implementer runs, the Evaluator reads each sprint contract and:
- Sets hard pass/fail thresholds for val_bpb delta, VRAM, and wall-clock time
- Defines what CONFIRMED, INCONCLUSIVE, and REFUTED mean for this specific hypothesis
- Specifies what the Implementer must log for the evaluation to be valid
- After the run, reads the results and issues a binding verdict

The Evaluator is instructed to be sceptical. Its prompt emphasises: "You are not the agent that proposed this idea. You have no stake in its success. Your job is to determine whether the evidence meets the bar, and the default answer is no."

### Phase 3: Implementer (writes code, runs training)

Same as v6 Phase 2 (implement, run, record) but:
- Cannot change the sprint contract or thresholds after they are set
- Cannot self-assess — records raw numbers only
- Cannot decide to "refine" or "try a variant" — only the Evaluator can authorise follow-up runs
- Must report all metrics specified in the sprint contract

---

## Detailed Design

### Task 1: Sprint Contract Format

The Researcher writes this for each hypothesis in `hypotheses.md`:

```markdown
## Hypothesis N: [Name]

### Sprint Contract

**Intervention:** [one-sentence description of the code change]
**Papers:** [Domain A paper] × [Domain B paper]
**Closest prior art:** [what exists, what's different]

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (minimum 1.5× seed variance of 0.002)
- VRAM < 76 GB (leaves 4GB headroom below 80GB limit)
- Wall-clock < 1200s (within 20-minute safety kill)
- No NaN/Inf in training loss

**INCONCLUSIVE if:**
- val_bpb delta between -0.003 and +0.001 (within noise)

**REFUTED if:**
- val_bpb delta > +0.001 (clear regression)
- CRASH (OOM, divergence, torch.compile failure)

**Predicted delta:** [point estimate]
**Predicted confidence:** [low/medium/high]

**Case for failure:** [one paragraph]

**Implementation sketch:** [what changes in train.py]
```

The key change: the -0.003 threshold is defined *before* the run, not retroactively. In v5, a -0.0002 change was called CONFIRMED. Under this contract, it would be INCONCLUSIVE.

### Task 2: Evaluator Instructions

Create a new section in program.md for the Evaluator phase. The Evaluator:

1. Reads the sprint contract from `hypotheses.md`
2. Reads the raw results from `results.tsv` and `run.log`
3. Applies the thresholds mechanically — no judgment calls
4. Writes a verdict to `evaluations.md`:

```markdown
## Evaluation: Hypothesis N

**Sprint contract thresholds:**
- val_bpb delta < -0.003: [PASS/FAIL] (actual: -X.XXXX)
- VRAM < 76 GB: [PASS/FAIL] (actual: XX.X GB)
- Wall-clock < 1200s: [PASS/FAIL] (actual: XXXs)
- No NaN/Inf: [PASS/FAIL]

**Verdict: [CONFIRMED / INCONCLUSIVE / REFUTED]**

**Evaluator notes:** [optional — only for explaining unexpected results]
```

5. If 3+ consecutive hypotheses targeting the same subsystem are REFUTED or INCONCLUSIVE, the Evaluator issues a **pivot directive**: "The next research phase must target a different architectural subsystem. Do not propose further modifications to [attention/MLP/residuals/etc.]."

### Task 3: Phase Transition Rules

The loop becomes:

```
RESEARCH → write sprint contracts
    ↓
EVALUATE → set thresholds (Evaluator reviews contracts, can reject or tighten)
    ↓
IMPLEMENT → run experiments (5 max, no self-assessment)
    ↓
EVALUATE → judge results (binding verdicts, pivot directives)
    ↓
RESEARCH → new hypotheses (informed by verdicts and any pivot directives)
```

Phase transitions are triggered by file state, not by the agent's internal decision:
- RESEARCH → EVALUATE: when `hypotheses.md` contains new sprint contracts with status PROPOSED
- EVALUATE → IMPLEMENT: when all contracts have status APPROVED (thresholds confirmed)
- IMPLEMENT → EVALUATE: when `results.tsv` has new entries with status PENDING_EVALUATION
- EVALUATE → RESEARCH: when all pending entries have verdicts AND the 5-run counter is reached

### Task 4: Pivot Mechanism

The Evaluator tracks which architectural subsystems have been tested:

```markdown
## Subsystem Tracker (in evaluations.md)

| Subsystem | Hypotheses tested | Confirmed | Refuted/Inconclusive |
|-----------|------------------|-----------|---------------------|
| Attention | H1, H3, H6 | 0 | 3 |
| Activation | H5 | 1 | 0 |
| Residuals | H4, H9 | 0 | 2 |
| Positional | H7 | 0 | 1 |
| MLP | — | — | — |
| Normalisation | — | — | — |
| Embeddings | — | — | — |
| Training loop | H3, H8 | 0 | 2 |
```

Pivot rule: if a subsystem has 3+ failures with 0 confirmations, the Evaluator blocks further hypotheses in that subsystem until a different subsystem produces a confirmed result. This forces genuine exploration across the architectural design space.

### Task 5: Context Reset Between Phases

Following Anthropic's finding that context resets beat compaction:

- The Researcher reads: `findings.md`, `evaluations.md` (verdicts only), `hypotheses.md` (prior art + outcomes). Does NOT read `research_log.md` (too detailed, causes context anxiety).
- The Evaluator reads: `hypotheses.md` (sprint contracts), `results.tsv`, `run.log`. Does NOT read `findings.md` or `research_log.md` (prevents bias from accumulated narrative).
- The Implementer reads: `hypotheses.md` (current sprint contract only), `train.py`. Does NOT read `findings.md`, `evaluations.md`, or `research_log.md` (prevents over-caution from prior failures).

Each phase gets only the information it needs. No phase sees everything.

### Task 6: Implementation as Single-Agent Simulation

Since we're running one Claude Code session, not three separate processes, we simulate the three-agent pattern via explicit phase headers in program.md:

```markdown
## Current Phase: [RESEARCH / EVALUATE / IMPLEMENT]

When entering a new phase:
1. State: "ENTERING [PHASE] MODE"
2. Read ONLY the files listed for this phase (see Context Reset rules)
3. Perform ONLY the actions allowed in this phase
4. When phase work is complete, state: "PHASE COMPLETE — transitioning to [NEXT PHASE]"
```

The risk: a single agent simulating three roles may not maintain genuine separation. Anthropic's article was explicit that "the most critical pattern is decoupling the generating agent from the evaluating agent." If the single-session simulation doesn't produce meaningfully different evaluation behaviour, we would need to run three separate Claude Code invocations communicating via files, which costs more in API calls and setup complexity.

### Task 7: Changes to program.md (v7)

Summary of structural changes from v6:

1. **Add sprint contract format** to hypothesis template (Task 1)
2. **Add Evaluator phase** between Research and Implement (Task 2)
3. **Replace self-assessment** with Evaluator verdicts — Implementer records raw numbers, Evaluator judges (Task 2)
4. **Add hard threshold of -0.003** as minimum for CONFIRMED (1.5× seed variance) (Task 1)
5. **Add pivot mechanism** with subsystem tracker (Task 4)
6. **Add context reset rules** — each phase reads different files (Task 5)
7. **Add phase transition protocol** with explicit state declarations (Task 6)
8. **Add evaluations.md** as new tracking file (Task 2)
9. **Remove "No Tier B Refinements" rule** — replaced by Evaluator authorisation. If the Evaluator sees a promising INCONCLUSIVE result, it can authorise ONE follow-up run with a tighter contract. This is more nuanced than a blanket ban.

### Task 8: Evaluation of the New Protocol

To test whether v7 produces different behaviour from v6:

**Metrics to compare:**
- CONFIRMED rate with hard thresholds (v7) vs soft self-assessment (v5/v6)
- Number of genuinely novel hypotheses (adversarial prior art search)
- Subsystem diversity (how many different subsystems explored)
- Pivot frequency (how often the evaluator forces exploration of new subsystems)
- Belief calibration (prior vs posterior across runs)
- OOM/crash rate (pre-flight gate effectiveness)
- Whether the single-session simulation produces meaningfully different evaluation behaviour

**Expected outcome:** Fewer CONFIRMED results (the -0.003 threshold is strict), but higher confidence in those that pass. More subsystem diversity due to pivot directives. Possibly slower progress in val_bpb but more credible claims about what actually works.

---

## Open Questions

1. **Can a single agent genuinely evaluate its own work differently by switching "modes"?** Anthropic's article is sceptical. If the single-session simulation doesn't work, we need three separate Claude Code invocations, which adds ~$3/hr in API costs and significant orchestration complexity.

2. **Is -0.003 the right threshold?** It's 1.5× the measured seed variance of 0.002, which is a common heuristic but not statistically rigorous. A proper threshold would require multiple runs per hypothesis (expensive) or a held-out validation set.

3. **Should the Evaluator be a different model?** Anthropic's article used the same model family for all agents. But using a smaller, cheaper model (Haiku) as the evaluator might produce more genuinely independent judgments, since it has different biases. The evaluation task is much simpler than research or implementation.

4. **Does the pivot mechanism over-constrain exploration?** Blocking a subsystem after 3 failures might prevent the agent from finding improvements that require multiple complementary changes to the same subsystem (e.g., v6's stacking of gated attention + focal temperature + softsign bounding, all in the attention/logit subsystem).
