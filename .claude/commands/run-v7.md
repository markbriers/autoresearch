---
description: Run one complete v7 research cycle using three subagents (Researcher, Evaluator, Implementer)
---

You are the orchestrator for one cycle of the autoresearch v7 three-agent protocol. You coordinate three subagents sequentially. Each subagent runs in its own context window and communicates via files on disk.

## Pre-flight Checks

Before starting, verify the environment:

1. Check tmux is installed: `which tmux`
2. Check train.py exists: `ls train.py`
3. Check Python venv: `ls .venv/bin/python`

If any check fails, print the error and exit.

## Initialise Tracking Files (first cycle only)

If `evaluations.md` does not exist, create it with:
- Header: `# Evaluations`
- Empty subsystem tracker table (columns: Subsystem, Tested, Confirmed, Refuted/Inconclusive, Status) with rows for attention, activation, residuals, positional, MLP, normalisation, embeddings, training-loop, all set to OPEN
- Empty Calibration Notes section
- Empty Pivot Directives section

If `findings.md` does not exist, create it with sections: Confirmed Mechanisms, Dead Ends, Architecture Inductive Biases, Cross-Domain Transfer Patterns, Open Questions.

If `hypotheses.md` does not exist, create it with: `# Hypotheses` and `## Engineering Run Counter: 0/5 | Phase: RESEARCH`

If `results.tsv` does not exist, create it with the header: `commit	val_bpb	memory_gb	status	description`

If `research_log.md` does not exist, create it with: `# Research Log`

## Establish Baseline (first cycle only)

If `results.tsv` contains only the header line (no data rows), run the baseline:
1. `.venv/bin/python train.py > run.log 2>&1`
2. Extract val_bpb and peak_vram_mb from run.log
3. Append to results.tsv: `baseline	[val_bpb]	[memory_gb]	keep	Baseline: unmodified train.py`
4. Record in research_log.md

## Determine Cycle State

Read `hypotheses.md` to determine where in the cycle we are:
- Hypotheses with status PROPOSED: skip to Pre-Evaluation
- Hypotheses with status APPROVED: skip to Implementation
- Hypotheses with status PENDING_EVALUATION: skip to Post-Evaluation
- Otherwise: start with Research

This allows the cycle to resume if a previous cycle crashed mid-way.

## Phase 1: Research

Spawn a subagent using the `researcher` agent type:

Prompt: "You are the Researcher. This is the research phase. Read findings.md, evaluations.md, and hypotheses.md for context. Search for papers across two cross-domain pairings. Generate 3-5 sprint contracts for novel architectural hypotheses. Write them to hypotheses.md with status PROPOSED. Use /scholar for paper discovery and arxiv MCP for full-text reading. When done, report how many contracts you wrote and which subsystems they target."

Wait for the subagent to complete. Verify hypotheses.md contains new PROPOSED entries.

## Phase 2: Pre-Evaluation

Spawn a subagent using the `evaluator` agent type:

Prompt: "MODE: PRE-RUN. Review all sprint contracts with status PROPOSED in hypotheses.md. Read evaluations.md for the subsystem tracker and pivot directives. For each contract: check feasibility, check pivot compliance, check for repetition of prior failures. Update each hypothesis status to APPROVED or REJECTED (with reason) in hypotheses.md. Report how many you approved and rejected."

Wait for the subagent to complete. Count APPROVED entries in hypotheses.md.

If 0 approved: spawn the Researcher again with feedback from the rejections, asking it to revise. Then re-run Pre-Evaluation. If still 0 after retry, exit cleanly (the bash loop starts a fresh cycle).

## Phase 3: Implementation

Spawn a subagent using the `implementer` agent type:

Prompt: "Implement all APPROVED hypotheses in hypotheses.md (up to 5). For each: read the sprint contract, modify train.py, git commit, run .venv/bin/python train.py > run.log 2>&1, record raw results to results.tsv with status pending_evaluation, write raw observations to research_log.md. Do NOT interpret results. Report how many runs you completed."

Wait for the subagent to complete. Verify results.tsv has new pending_evaluation entries.

## Phase 4: Post-Evaluation

Spawn a subagent using the `evaluator` agent type:

Prompt: "MODE: POST-RUN. Follow the full post-run sequence: (B.1) Read results.tsv and run.log, apply sprint contract thresholds mechanically, write verdicts to evaluations.md, update hypotheses.md status. Do this BEFORE reading findings.md. (B.2) NOW read findings.md and check for patterns across experiments. Write enriched failure analyses. (B.3) Update the subsystem tracker with fine-grained categories. (B.4) Assess prediction calibration and write calibration notes. (B.5) Issue pivot directives if any subsystem has 3+ failures with 0 confirmations. (B.6) Authorise follow-ups for borderline INCONCLUSIVE results if warranted. Report verdicts, any pivot directives issued, and calibration observations."

Wait for the subagent to complete.

## Phase 5: Git Snapshot

Stage and commit all tracking files:

```bash
git add hypotheses.md evaluations.md findings.md results.tsv research_log.md
git commit -m "cycle snapshot: [summary from evaluator]" --allow-empty
```

## Exit

Exit cleanly. The external bash loop (run_v7_loop.sh) starts a fresh session for the next cycle.
