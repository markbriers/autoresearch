---
description: Run one complete research cycle using three subagents (Researcher, Evaluator, Implementer)
---

You are the orchestrator for one cycle of the autonomous research protocol. You coordinate three subagents sequentially. Each subagent runs in its own context window and communicates via files on disk.

## Pre-flight Checks

Before starting, verify:

1. `task.md` exists in the working directory
2. `task.md` contains all six required section headings: `## Objective`, `## Intervention Space`, `## Evaluation`, `## Domain Context`, `## Subsystem Taxonomy`, `## Cross-Domain Requirement`
3. Git is initialised in the working directory

If any check fails, print a clear error message explaining what is missing and exit.

## Initialise Tracking Files (first cycle only)

If `evaluations.md` does not exist, create it with:
- Header: `# Evaluations`
- Empty subsystem tracker table using categories from the Subsystem Taxonomy section of task.md (columns: Subsystem, Tested, Confirmed, Refuted/Inconclusive, Status — all set to OPEN)
- Empty Calibration Notes section
- Empty Pivot Directives section

If `findings.md` does not exist, create it with sections: Confirmed Mechanisms, Dead Ends, Inductive Biases, Cross-Domain Transfer Patterns, Open Questions.

If `hypotheses.md` does not exist, create it with: `# Hypotheses` and `## Engineering Run Counter: 0/5 | Phase: RESEARCH`

If `results.tsv` does not exist, create it with the header: `commit	metric_value	resource_usage	status	description`

If `research_log.md` does not exist, create it with: `# Research Log`

## Establish Baseline (first cycle only)

If `results.tsv` contains only the header line (no data rows), run the baseline:
1. Read the Objective section of task.md for the run command and metric extraction method
2. Execute the command, redirecting output to run.log
3. Extract the metric and resource usage as specified in task.md
4. Append to results.tsv: `baseline	[metric_value]	[resource_usage]	keep	Baseline: unmodified`
5. Record in research_log.md

## Determine Cycle State

Read `hypotheses.md` to determine where in the cycle we are:
- Hypotheses with status PROPOSED: skip to Pre-Evaluation
- Hypotheses with status APPROVED: skip to Implementation
- Hypotheses with status PENDING_EVALUATION: skip to Post-Evaluation
- Otherwise: start with Research

This allows the cycle to resume if a previous cycle crashed mid-way.

## Phase 1: Research

Spawn a subagent using the `researcher` agent type:

Prompt: "You are the Researcher. Read task.md first, then findings.md, evaluations.md, and hypotheses.md. Generate 3-5 sprint contracts following the cross-domain requirement in task.md. Write them to hypotheses.md with status PROPOSED. Use /scholar for paper discovery and arxiv MCP for full-text reading. Report how many contracts you wrote and which subsystems they target."

Wait for completion. Verify hypotheses.md contains new PROPOSED entries.

## Phase 2: Pre-Evaluation

Spawn a subagent using the `evaluator` agent type:

Prompt: "MODE: PRE-RUN. Read task.md first. Review all sprint contracts with status PROPOSED in hypotheses.md. Check feasibility against task.md constraints, check pivot compliance, check cross-domain compliance, check for repetition. Write independent P(success) estimates. Update each status to APPROVED or REJECTED. Report counts."

Wait for completion. If 0 approved, spawn the Researcher again with rejection feedback. If still 0 after retry, exit cleanly.

## Phase 3: Implementation

Spawn a subagent using the `implementer` agent type:

Prompt: "Read task.md first. Implement all APPROVED hypotheses in hypotheses.md (up to 5). For each: read the contract, modify the target file(s) specified in task.md, git commit, run the command from task.md's Objective section, record raw results to results.tsv and research_log.md. Do NOT interpret results. Report how many runs you completed."

Wait for completion. Verify results.tsv has new pending_evaluation entries.

## Phase 4: Post-Evaluation

Spawn a subagent using the `evaluator` agent type:

Prompt: "MODE: POST-RUN. Read task.md first for thresholds. Follow the full post-run sequence: (B.1) mechanical verdicts from results.tsv against task.md thresholds, (B.2) pattern analysis after reading findings.md, (B.3) subsystem tracker update using task.md categories, (B.4) calibration and surprisal analysis, (B.5) pivot and exploration directives, (B.6) follow-up authorisation. Report verdicts and any directives."

Wait for completion.

## Phase 5: Git Snapshot

Stage and commit all tracking files:

```bash
git add hypotheses.md evaluations.md findings.md results.tsv research_log.md
git commit -m "cycle snapshot: [summary from evaluator]" --allow-empty
```

## Exit

Exit cleanly. The external bash loop (run_loop.sh) starts a fresh session for the next cycle.
