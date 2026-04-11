# Teaching an AI Agent to Read Papers: What Happens When You Give an Autonomous ML Engineer Access to the Scientific Literature

**TLDR.** I extended Karpathy's autoresearch benchmark with academic literature access and tried, across eight iterations, to make the agent behave like a researcher rather than an engineer. The agent never produced a novel architectural mechanism. It did, when sufficiently constrained, produce novel chains of reasoning that arrived at known mechanisms via theoretical results from outside machine learning (wavelet shrinkage theory, compressed sensing, statistical mechanics). Most of the work was in the harness, not the model. The single most effective change was a one-line prompt rule requiring at least one paper in each cross-domain pairing to come from outside ML. Total cost: about £75. The code, all results, and the research logs are in the repository.

---

In early March 2026, Andrej Karpathy released [autoresearch](https://github.com/karpathy/autoresearch), an open-source benchmark that hands a coding agent (Claude Code, in his original setup) a small GPT training script and lets it optimise val_bpb autonomously. The agent has shell access via Bash, can edit files, and runs in a tight loop: read code, modify, train for a fixed wall-clock budget, measure bits-per-byte, keep improvements, revert failures. On a given machine you can run roughly twelve experiments per hour with no human in the loop.

Watching the agent work, I noticed something. Every improvement was an engineering optimisation: adjust the learning rate, reduce the batch size, tweak the warmdown. The agent behaved like a systematic junior engineer, not a researcher. It never consulted the literature. It never formed a hypothesis grounded in theory. It never made a surprising connection between fields. This is not a Claude-specific failure — it's how coding agents in general behave under task-completion framing.

This got me thinking. Could I augment the loop with access to the scientific literature, and if so, would the agent actually use it to do something that looked more like research?

I'm not the only person asking this. [Paper Lantern](https://www.paperlantern.ai/blog/auto-research-case-study), an MCP server giving agents access to two million CS papers, ran a controlled comparison on the same benchmark and reported a 4.05% improvement with literature access versus 3.67% without. Their agent treats the literature as a lookup table — searching for known techniques and applying them. I wanted to test something harder: whether the agent could synthesise ideas across unrelated fields and produce reasoning that a competent engineer would not have arrived at by staring at the code.

What follows is a case study in eight iterations. The main contribution is not a new architecture but a set of observations about how benchmark design, process constraints, and prompt-level framing determine whether an agent explores creatively or defaults to safe engineering.


## The First Few Iterations (v1–v3): How Hard the Defaults Pull

I extended the agent's tool access. I gave it an arxiv MCP server (search papers, download, read full text in markdown), a custom skill wrapping the OpenAlex API for relevance-ranked search across hundreds of millions of academic works, and a system prompt that demanded cross-domain synthesis before any code change. I added a three-tier persistent memory system: a research log capturing every run, a findings file distilling generalisable insights, and a long-term memory file promoted across sessions.

It mostly didn't matter. Across the first 117 experiments (40 on a MacBook, 77 on a RunPod H100), only three were paper-driven. The rest were the same hyperparameter tuning Karpathy's original system produces. The new tools and memory were available; the agent rarely chose to use them. Its expected-value calculation always favoured "use Bash to edit train.py and adjust the learning rate" over "use the arxiv MCP to read a paper from control theory and try to connect it to attention."

In v2 I separated the loop into two phases with a hard counter: Phase 1 reads papers and writes hypotheses, Phase 2 implements them, and after five engineering runs the agent must return to research. This produced better hypotheses, but they all failed experimentally. The failure analyses were precise and useful — but by v3 I had tested about 17 paper-driven hypotheses across multiple runs and none had produced a confirmed improvement that engineering alone wouldn't have found. The underlying problem was something else.


## The Throughput Trap (v4)

The culprit was the time budget. When training is bounded by wall-clock time, every architectural change is evaluated through a throughput lens. A wider MLP that processes 17% fewer steps in the same time appears to regress, even if each step is more informative. The fix was embarrassingly simple: replace the time budget with a step budget. Instead of "train for 7.5 minutes," "train for exactly 1800 optimiser steps."

The results were immediate and slightly humbling. The accumulated finding that "ReluSquared is better than SwiGLU," recorded as established fact across more than a hundred experiments, reversed. SwiGLU is slower per step (it uses all neurons; ReluSquared is ~95% sparse), so under wall-clock evaluation it always appeared worse. Under step-bounded evaluation it won by 0.004. Within this benchmark, switching from wall-clock to fixed-step evaluation reversed several earlier conclusions. This isn't a claim that fixed-compute evaluation is generally a mistake — it's standard in scaling-law work. But in a wall-clock-bounded loop where the agent can change both the architecture and the step count simultaneously, throughput can swamp per-step learning quality and reverse apparent architecture rankings.


## The First Clean Experiment (v5)

Armed with step-bounded training, I designed an experiment to isolate the research contribution. All hyperparameters were frozen — learning rates, batch size, warmdown, weight decay, softcap. All model dimensions were frozen — depth, width, head count, MLP expansion. The agent could only modify the architecture itself, through changes grounded in cross-domain paper synthesis. No follow-up tuning. If an idea worked at the default hyperparameters, it worked. If not, the failure analysis recorded why.

A "confirmed improvement" here means a change that reduced val_bpb relative to the frozen baseline in a single run. Seed variance is approximately 0.002 val_bpb (measured in one spot check), so improvements smaller than that should be treated as directional signals rather than statistically robust findings. Multi-seed validation would strengthen all these claims.

v5 produced three architectural changes that stacked: a pre-attention causal depthwise convolution (-0.0017), per-head learnable attention scaling (-0.0002), and partial RoPE (-0.0035). Cumulative reduction from baseline: 0.0054.

The partial RoPE result was the interesting one. The agent's reasoning paired positional encoding analysis with OFDM (orthogonal frequency-division multiplexing) from telecommunications. By analogy with OFDM's principle of dedicating orthogonal subcarriers to different information types, the agent proposed allocating 75% of head dimensions to position-aware matching and 25% to position-free content matching. Partial rotary allocation has clear prior art (GPT-NeoX's `rotary_pct`), but I'd never seen anyone derive it from telecommunications subcarrier theory. The mechanism is known. The reasoning path isn't.


## The Rediscovery Problem (v6 → v7)

v5's three confirmed improvements were all known techniques. Pre-attention convolution exists in Conformer and Primer. Learnable attention scaling is well-precedented. Partial RoPE is in GPT-NeoX. The system was finding things that worked, but finding them by retracing paths other researchers had already walked.

For v6 I added adversarial self-critique, quantified belief tracking, and stronger prior art search (drawing on Google's AI Co-Scientist and AI2's AutoDiscovery). v6 produced eight confirmed improvements stacking from 0.9595 to 0.9496. All eight were still known techniques. By v7 I was convinced the bottleneck was the single-session phase simulation: when one agent generates a hypothesis, implements it, and judges the result, it has a stake in every step. [Anthropic's harness design article](https://www.anthropic.com/engineering/harness-design-long-running-apps) makes exactly this point about coding agents grading their own work.

I split the protocol into three Claude Code subagents — Researcher, Implementer, Evaluator — each defined in its own `.claude/agents/*.md` file with a distinct system prompt and a restricted tool list. Each subagent runs in its own isolated context window. The Researcher has full tool access (Bash, Read, Edit, Write, WebFetch, WebSearch, plus the arxiv MCP). The Implementer has Bash and file editing tools but no web access — it cannot read papers or second-guess the design. The Evaluator has *only* Read and Edit; no Bash, no web, no MCP. It physically cannot run code, search, or do anything that creates attachment to outcomes. It can only read files and write verdicts against a hard threshold. I wrapped the whole thing in an external bash loop that starts a fresh Claude Code session per cycle, so there is zero shared context between cycles — the persistent memory lives entirely in the tracking files on disk.

v7 ran one cycle. Three of four hypotheses were confirmed. The problem: the cross-domain pairings were weak. The Researcher was pairing Huang 2024 with Shazeer 2020 (two activation-function papers) and calling it cross-domain. The "two distinct domains" instruction had a loophole — different ML subfields qualified.


## The One-Line Fix (v8)

v8 was the cheapest and most effective change of the entire project. I added one sentence to the Researcher's system prompt: each domain pairing must contain exactly one ML paper and one paper from outside machine learning, deep learning, and optimisation. Valid Domain B sources include signal processing, control theory, information theory, neuroscience, physics, biology, telecommunications, compressed sensing, statistical mechanics. The Evaluator gained a matching rejection rule in its own system prompt — non-compliant contracts get blocked at the approval gate before the Implementer ever sees them.

The effect was immediate. The Researcher's next batch paired Shazeer's GLU variants with Candes and Tao on compressed sensing. Ma's MEGA attention with Brown's 1963 exponential smoothing. Gemma-2's softcap with Carandini and Heeger on divisive normalisation in cortex. DeepNet's residual scaling with Mallat's multi-resolution wavelet analysis. Gemma-2's logit softcap with Jaynes's 1957 maximum entropy paper. The pairings were finally genuinely cross-domain.

Three hypotheses were confirmed before the pod ran out of funds. The cleanest was **ShrinkReLU**, derived from Donoho and Johnstone's 1994 wavelet shrinkage paper. The agent's reasoning chain: Donoho and Johnstone proved that soft thresholding is minimax-optimal for removing Gaussian noise from a signal, with the optimal threshold scaling with the noise level. Map this onto MLP activation design — small activations are likely noise from imperfect earlier layers; large activations are signal. Make the threshold learnable per layer so each layer adapts its denoising strength. The implementation: replace `F.relu(x).square()` with `F.relu(x - tau).square()` where tau is a softplus-parameterised learnable scalar. Ten extra parameters total. The result was a val_bpb reduction of 0.0058.

I think ShrinkReLU is the cleanest example of the "known technique via novel reasoning path" pattern across all eight versions. Soft thresholding is textbook signal processing. Learnable ReLU biases have been explored. But the specific combination, justified by wavelet shrinkage theory as activation denoising in a Muon-optimised transformer, doesn't appear in the literature in that form. The theoretical justification is load-bearing. An engineer trying "add a learnable bias to ReLU" faces an infinite design space — positive, negative, per-channel, per-layer, fixed, scheduled. Donoho and Johnstone tell you specifically: positive, per-layer, initialised small, learned, because optimal denoising thresholds scale with the noise level at each stage of processing. The theory collapses the design space to a single specific configuration with a principled reason for why it works.

The second confirmed result was a dimension-reduced SwiGLU, with the hidden-dimension choice justified by Candes and Tao's bounds on sparse signal recovery (sparse recovery needs only enough measurements to satisfy the recovery threshold, so 1728 rather than the full 4× expansion was sufficient). The third was a learnable softcap temperature derived from Jaynes's maximum entropy principle: the optimal entropy constraint should match the true entropy of the target distribution, which varies with training progress, so the softcap shouldn't be a fixed constant.

v8 also surfaced a problem I didn't get to solve. Stacking attempts on top of confirmed improvements often catastrophically regressed (ShrinkReLU plus learnable RMSNorm gain produced +0.136 val_bpb), apparently because both modify the same activation statistics. The Evaluator would have issued pattern directives about stacking risk in later cycles, but the pod ran out of funds before cycle 3.


## Summary

| Version | Budget | Runs | Paper-driven | Best delta | Notes |
|---------|--------|------|--------------|------------|-------|
| v1 | 5 min wall-clock | 117 | 3 | -0.062 (eng) | Single loop; agent defaults to engineering |
| v2 | 7.5 min wall-clock | 48 | 5 | -0.024 (eng) | Phase separation, counter drift |
| v3 | 7.5 min wall-clock | 4 | 4 | 0 | Hard counter; throughput kills creative ideas |
| v4 | 1800 steps | 22 | 5 | -0.059 (eng) | Step budget reverses SwiGLU finding |
| v5 | 1800 steps, frozen HPs | 13 | 13 | -0.005 (research) | 3 confirmed; partial RoPE via OFDM |
| v6 | 1800 steps, frozen HPs | 39 | 39 | -0.010 (research) | 8 confirmed; all known techniques |
| v7 | 1800 steps, frozen HPs | 4 | 4 | -0.005 (research) | Three subagents; within-ML pairings |
| v8 | 1800 steps, frozen HPs | 7 | 7 | -0.007 (research) | Non-ML Domain B enforced; ShrinkReLU |

"Confirmed" from v5 onwards means val_bpb decreased by more than 0.003 (1.5× seed variance) relative to the frozen baseline in a single run, without multi-seed validation.


## What I Actually Learned

**None of the interventions are novel mechanisms.** Every confirmed improvement across 200+ experiments has a clear precedent in the literature. An ML engineer familiar with the field could have proposed any of them without reading a telecommunications textbook or a 1957 statistical mechanics paper.

**The agent did, at its best, produce novel reasoning paths.** OFDM-to-partial-RoPE. Donoho-to-ShrinkReLU. Jaynes-to-learnable-softcap. In each case the intervention is mechanistically justified by a non-ML theoretical result, and the justification narrows the design space from infinite variations to a single specific configuration with a principled reason for why it works. Whether this constitutes "research" depends on your definition. Discovering mechanisms no one has proposed: no. Arriving at known mechanisms through chains of reasoning that connect previously unrelated fields: yes, when forced.

**The agent will default to engineering unless every loophole is closed.** A single-loop agent does Tier B tuning. Phase separation gets circumvented. A hard counter gets relabelled. A two-domain rule gets interpreted as different ML subfields. The load-bearing element was always the constraint that was hardest for the agent to reinterpret creatively. A sceptical reader might also note that 200+ adaptive experiments against a single validation metric look a lot like benchmark gaming. They'd be right — the frozen-HP constraint in v5+ prevents the most obvious form, but architectural changes are still selected against the same metric. Multi-seed validation would strengthen all of this; I didn't do it.

**The deeper issue is exploration-exploitation, not autoresearch specifically.** Colleagues working with coding agents (Cursor, Devin, Codex, Claude Code) in other contexts report the same dynamic: the agent settles into execution mode (tight feedback, clear next steps, reliable small gains) and resists switching to research mode (ambiguous goals, deferred reward, higher variance). Coding agents are post-trained to be helpful task-completing systems — rewarded for solving the problem in front of them, producing working code, converging on an answer. That training objective directly selects for exploitation. The research mode I wanted (read broadly, form speculative hypotheses, tolerate ambiguity, value information gain from failed experiments) is essentially the opposite of what the post-training optimises for. One could imagine a post-training objective that rewards epistemic curiosity directly — weighting reward by Bayesian surprise, so a calibrated prediction that fails interestingly is worth more than a safe prediction that succeeds boringly. The mathematical machinery exists in active inference and Bayesian optimal experimental design. As far as I know, nobody has applied it to coding-agent post-training.

**Single-line summary:** The system did not generate novel architectures. It did generate novel reasoning paths to known architectures, when forced. Most of the work of making a coding agent do research is not in the model's reasoning capability — it's in the harness: the system prompts, the tool restrictions, the context isolation between subagents, the persistent memory format, and the specific prompt-level constraints that close the loopholes the agent finds when given any flexibility.

---

The code, results files, research logs, and hypothesis records across all eight versions are at [github.com/markbriers/autoresearch](https://github.com/markbriers/autoresearch) on the `autoresearch/mar23` branch. Total compute cost: approximately £75. Karpathy's original is at [github.com/karpathy/autoresearch](https://github.com/karpathy/autoresearch).
