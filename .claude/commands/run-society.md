---
description: Run one cycle of the society experiment (single-researcher or multi-lab arm)
---

You are the orchestrator for one cycle of the society experiment. Read `arm.env` in the working directory first. It sets `ARM` (single or society), `LABS` (space-separated lab names), `THEMES` (one per lab, same order, separated by `;`) and `PER_CYCLE` (contracts per cycle, the same in both arms so compute is matched).

Subagents communicate only through files. Do not paste one subagent's output into another's prompt beyond what is written below.

## Pre-flight

1. `.venv/bin/python` exists and `git rev-parse champion` succeeds. If either fails, print the problem and exit.
2. Make sure tracking files exist: `hypotheses.md` (`# Hypotheses`), `results.tsv` (header `commit\tval_bpb\tmemory_gb\tstatus\tdescription`), `research_log.md`, `findings.md`, and `evaluations.md` with the subsystem tracker described in `.claude/commands/run-v7.md`. Create `labs/<lab>/` for each lab.
3. Sync the main checkout's `train.py` to the champion so researchers read current code: `git show champion:train.py > train.py`.
4. If no `champion` row in `results.tsv` refers to the current `git rev-parse --short champion`, run the champion once at seed 42 in a worktree (`git worktree add --detach worktrees/champ champion`, run `../../.venv/bin/python train.py > run.log 2>&1` inside it, remove the worktree) and append `[short sha]\t[val_bpb]\t[memory_gb]\tchampion\tchampion reference (seed 42)`.
5. Run `python3 society/board.py render`.

## Phase 1: Research

If `ARM=single`: spawn one `lab-researcher` with:
"LAB: solo. THEME: any. VISIBILITY: OPEN. Write PER_CYCLE contracts to labs/solo/proposals.md."

If `ARM=society`: spawn one `lab-researcher` per lab, all in parallel in a single message, each with:
"LAB: [lab]. THEME: [theme]. VISIBILITY: BOARD. Write 1 contract to labs/[lab]/proposals.md." (Use PER_CYCLE divided by the number of labs, rounded down, minimum 1.)

Then run `python3 society/board.py merge`.

## Phase 2: Pre-evaluation

Spawn an `evaluator` with: "MODE: PRE-RUN. Review all PROPOSED contracts in hypotheses.md. Check feasibility for the profile in train.py, check the Domain B rule, and reject repeats of anything already tested. Write your independent P(success) and the belief divergence on each. Set each to APPROVED or REJECTED. Approve at most PER_CYCLE." If none are approved, ask the researcher(s) whose contracts were rejected to revise once, merge, and re-run pre-evaluation. If still none, exit.

## Phase 3: Implementation

Spawn a `society-implementer` with: "Implement every APPROVED hypothesis in hypotheses.md following your instructions."

## Phase 4: Post-evaluation

Spawn an `evaluator` with the same MODE: POST-RUN prompt as in `.claude/commands/run-v7.md` Phase 4, adding: "Deltas are against the most recent champion row in results.tsv."

## Phase 5: Audit

For each hypothesis the Evaluator marked CONFIRMED this cycle, in order of most negative delta first:
`.venv/bin/python society/seed_audit.py --hypothesis N --commit $(git rev-parse exp/HN)`
Each later audit is applied on top of whatever the earlier ones promoted, so this is also the stacking test. Write one line per audit to `research_log.md` with the decision and the delta_mean and z it printed.

## Phase 6: Board and snapshot

Run `python3 society/board.py render`. Append a line to `cycles.log`: cycle number, ARM, hypotheses tested, confirmed, promoted, current champion sha. Then exit. The loop script starts a fresh session for the next cycle.
