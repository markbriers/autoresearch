# One researcher or a society of labs?

This directory turns the v9 protocol into a controlled comparison between two ways of organising the agents, at equal compute. It applies the main findings of "A Society of AI Scientists" (Ahmad et al., 2026, [alphaXiv](https://www.alphaxiv.org/abs/2611.society-ai-scientists)), a twelve-week study of 936 agents across 35 laboratories:

1. Single-agent loops that refine their own work stop gaining after round 1. Across 19 multi-round studies, 25 of 27 improving arms showed diminishing gains after the first round, and 0 of 5 direct comparisons favoured recursive over independent proposals.
2. Showing agents each other's scores causes herding. In their multi-agent search trial, 88% of later trials crowded around the current leader when scores were shared, compared with 30% for independent agents. Sharing what had been tried *without* the scores broke the herding and restored broad search.
3. Checks that run code catch what reading text misses. When referee objections were run as scripts on deposited data, 49.4% of headline conclusions were overturned.

The v9 run looks like finding 1. It found most of its results early, and by cycles 4 to 6 the Evaluator was rating most outcomes as low surprisal. The blog put this down to the agents calibrating. The other possible explanation is a single researcher narrowing onto its own leaderboard. This experiment separates the two.

## The two arms

Both arms use the same Evaluator, the same Implementer, the same multi-seed audit and the same number of contracts per cycle (4). They differ only in who proposes and what they can see.

| | single | society |
|---|---|---|
| Proposers | one `lab-researcher` (LAB=solo, THEME=any) | four `lab-researcher`s in parallel, one contract each |
| Themes | none | attention and positional; MLP and activation; residual stream and normalisation; embeddings, output head and training loop |
| Can see | `findings.md`, `evaluations.md`, `hypotheses.md` (every verdict and number), as in v9 | `board.md` (every lab's attempts, no outcomes or scores) and its own lab's verdicts |
| Context | fresh per cycle | fresh per cycle, and no lab sees another's reasoning |

`board.py` builds the board and each lab's own view mechanically, so hiding scores does not depend on the prompt alone. The researchers still have Bash, so the prompt also tells them not to read the other files.

## The audit (both arms)

In v9 a single run with delta below -0.003 counted as CONFIRMED. With seed s.d. of about 0.002, a difference of two single runs has s.d. of about 0.0028, so that gate is roughly one sigma, and later work stacked on top of it. Here a CONFIRMED result only becomes part of the shared `champion` once `seed_audit.py` has:

1. applied the change on top of the current champion (so every promotion is also a stacking test);
2. run the champion and the candidate on 3 seeds each;
3. found a mean improvement beyond 0.003 that is also at least two standard errors from zero.

Every audit seed run counts against the arm's run budget, so the extra rigour has a cost and the comparison stays honest.

## Running on the laptop (Apple Silicon)

`train.py` now has a `laptop` profile: a 4-layer model with 16K tokens per step and a 1000-step budget, using MPS, with no torch.compile and no FlashAttention. It is selected automatically when there is no CUDA device.

```bash
uv sync                         # on macOS this now resolves torch from PyPI (MPS); uv.lock will update
uv run prepare.py --num-shards 4
# 1. Calibrate: three seeds of the unmodified baseline. Tells you seconds per run and seed s.d.
git branch -f champion HEAD
.venv/bin/python society/seed_audit.py --baseline-only
```

If one run takes much more than about 8 minutes, lower the step budget for the whole experiment with `AR_STEP_BUDGET=600`, or shorten evaluation with `AR_EVAL_TOKENS=10485760`. Set either one in the environment of both arms and never change it mid-experiment. If the seed s.d. is well above 0.002, raise the threshold with `AR_THRESHOLD`, for example to 1.5 times the s.d.

```bash
# 2. Create the two arms from the same commit
society/new_arm.sh single  ../arm-single
society/new_arm.sh society ../arm-society

# 3. Run them one after the other with the same budget (the laptop has one GPU)
(cd ../arm-single  && RUN_BUDGET=30 ../autoresearch/society/run_arm.sh)
(cd ../arm-society && RUN_BUDGET=30 ../autoresearch/society/run_arm.sh)

# 4. Compare
python3 society/compare_arms.py ../arm-single ../arm-society
```

With 30 runs per arm at 6 to 8 minutes each, each arm takes about 3 to 4 hours, so both fit in one night. Claude API usage for the agents is separate from compute and is roughly the same for both arms.

The laptop results answer the organisational question (does a society out-search one researcher?) on a smaller model. They are not comparable with the H100 numbers in `results/v1` to `results/v9`.

## Running on an H100

Do the same steps with `AR_PROFILE=h100`, which uses the v9 frozen hyperparameters (depth 10, 1800 steps). The architecture is the one in `train.py` on master, not byte-identical to `results/v9/train.py`, so the arms establish their own baseline. Expect about 15 minutes per run. At 40 runs per arm, that is about 20 GPU-hours for both arms.

## What to read out

`compare_arms.py` reports, per arm: the runs used, single-run confirmations, multi-seed promotions, the final champion's mean val_bpb and its gain over the start, the subsystems tested with their spread (entropy), and confirmations in the first and second halves of the run.

These are predictions, not results. If the paper's findings carry over:

- the society arm spreads its tests more evenly across subsystems (higher entropy);
- it keeps confirming in the second half, where the single arm flattens;
- it ends with a lower audited champion.

If the single arm matches or beats the society, that supports the v9 reading that the system had converged, rather than herded. That is also worth reporting.

## Caveats

- One arm each is a single sample. If the gap is small, run a second pair with the arm order swapped before claiming anything.
- Labs read the shared champion `train.py`, so a promotion is visible to them as code even though no score is. That matches the paper's shared testbeds, and it is the one channel of leader information that remains.
- The paper is observational and its herding result comes from a small optimisation task, not from training runs. This experiment is a test of whether it transfers.

## Files

| File | Purpose |
|---|---|
| `../.claude/agents/lab-researcher.md` | Researcher used by both arms; visibility and theme set per call |
| `../.claude/agents/society-implementer.md` | Implements each contract on top of `champion` in its own worktree |
| `../.claude/commands/run-society.md` | One cycle: research, pre-eval, implement, post-eval, audit, board |
| `board.py` | Merges lab proposals; renders the score-free board and per-lab views |
| `seed_audit.py` | Multi-seed gate between CONFIRMED and PROMOTED; advances `champion` |
| `new_arm.sh`, `run_arm.sh` | Create an arm and run it to its run budget |
| `compare_arms.py` | Side-by-side summary of the arms |
