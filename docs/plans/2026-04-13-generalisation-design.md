# Generalisation Design: Domain-Agnostic Autoresearch

**Goal:** Refactor the three-subagent protocol so it works with any objective function, not just GPT val_bpb. The user writes a `task.md` describing their task; the agent definitions and orchestrator are generic.

**Architecture:** Extract all domain-specific content from `.claude/agents/*.md` into `task.md`. Agent definitions keep protocol logic only. Subagents read both their definition (protocol) and `task.md` (domain) at runtime. The GPT experiment becomes a worked example at `examples/gpt-training/task.md`.

## What changes

### task.md (user writes this)

Freeform markdown with six required sections (validated by pre-flight check):

1. **Objective** — command to run, metric extraction, metric name, direction (min/max)
2. **Intervention Space** — what can be modified, what is frozen
3. **Evaluation** — noise floor, confirmation threshold (1.5x noise floor), resource constraints
4. **Domain Context** — prior knowledge, what has been tried, what works/fails
5. **Subsystem Taxonomy** — categories for the subsystem tracker and pivot directives
6. **Cross-Domain Requirement** — what is "your field" (Domain A) vs "outside" (Domain B)

No YAML, no frontmatter. Subagents read these sections as natural language. Pre-flight only verifies the six headings exist.

### Agent definitions (become generic)

- **Researcher** — remove frozen config, prior knowledge, ML-specific self-critique. Replace with "read task.md sections: Domain Context, Cross-Domain Requirement, Subsystem Taxonomy." Keep: sprint contract format, information gain analysis, adversarial debate, P(success).
- **Evaluator** — remove hardcoded threshold (-0.003), VRAM/wall-clock limits, transformer subsystem categories. Replace with "read task.md sections: Evaluation, Subsystem Taxonomy." Initialise tracker from task.md categories. Keep: mechanical verdicts, pattern analysis, independent P(success), surprisal, pivot directives, calibration.
- **Implementer** — remove hardcoded train.py command, frozen config list, val_bpb-specific revert rules. Replace with "read task.md sections: Objective, Intervention Space." Keep: minimal implementation, raw numbers, no self-assessment.

### Orchestrator (run-v7.md)

Add pre-flight check: verify task.md exists and contains six required section headings. Everything else stays the same.

### Bash loop (run_v7_loop.sh)

No changes.

### Tracking files

No structural changes. findings.md, evaluations.md, hypotheses.md, results.tsv, research_log.md are all domain-agnostic. The Evaluator initialises subsystem tracker categories from task.md instead of hardcoding.

## File structure after refactor

```
task.md                         — user writes this
examples/
  gpt-training/
    task.md                     — worked example (our GPT experiment)
    README.md                   — pointer to docs/blog-post.md
.claude/agents/
  researcher.md                 — generic protocol
  evaluator.md                  — generic protocol
  implementer.md                — generic protocol
.claude/commands/
  run-cycle.md                  — orchestrator (renamed from run-v7.md)
  scholar.md                    — OpenAlex paper search skill
run_loop.sh                     — bash wrapper (renamed from run_v7_loop.sh)
docs/
  blog-post.md                  — write-up
  gpu-cloud-setup.md            — cloud setup guide
results/                        — historical results from GPT experiment
```

## What stays exactly the same

- Three-subagent separation (Researcher/Evaluator/Implementer)
- Sprint contract format with quantified success criteria
- Bayesian surprise (independent P(success), belief divergence, surprisal)
- Adversarial self-critique and prior art search
- Cross-domain enforcement (one paper from user's field, one from outside)
- External bash loop with per-cycle context isolation
- File-based persistent memory (findings, evaluations, hypotheses, results, log)
- Subsystem tracker with pivot directives
- Stacking analysis and compositionality testing
- Evaluator as knowledge curator (pattern analysis, calibration, exploration directives)

## Implementation sequence

1. Create feature branch `feat/generalise`
2. Create `examples/gpt-training/task.md` by extracting domain content from current agents
3. Rewrite `.claude/agents/researcher.md` — remove domain, add task.md references
4. Rewrite `.claude/agents/evaluator.md` — remove domain, add task.md references
5. Rewrite `.claude/agents/implementer.md` — remove domain, add task.md references
6. Update orchestrator — rename to `run-cycle.md`, add task.md pre-flight check
7. Rename `run_v7_loop.sh` to `run_loop.sh`
8. Update README.md with generalised quick start
9. Test: copy `examples/gpt-training/task.md` to `./task.md` and verify the orchestrator still works (dry run)
