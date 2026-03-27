# Research Log — v3 (Hard Counter Enforcement)

## Starting point

Inheriting v2 best: val_bpb 0.953816, 10x640 (85.9M params), batch=262K, 1790 steps.
Config: DEPTH=10, model_dim=640, 5 heads, SSSL window, softcap=12, Muon LR=0.03, warmdown=0.7, WD=0.1, FINAL_LR_FRAC=0.01, x0_lambdas=0.2.

Key v2 lessons: Muon incompatible with per-dim scaling. Per-token VE diversity is load-bearing. ReluSquared > SwiGLU. Phase discipline drifted after hypothesis testing — v3 adds hard counter enforcement (5 eng runs max, then mandatory research).

## Run 1 | val_bpb: N/A | delta from baseline: N/A | CRASH (OOM)

**Diagnosis:** Attention maps in 10-layer model overallocate to irrelevant context; differential attention (noise cancellation) should produce sparser, more focused patterns.

**Domain A** — Differential amplifiers in analog electronics
- Source primitive: Common-mode rejection — subtract two matched signals to cancel shared noise
- Target bottleneck: Attention maps contain noise (uniform activation to irrelevant tokens)
- Mapping (mechanistic): Subtracting two softmax attention maps cancels the common-mode component (uniform baseline attention), amplifying the differential signal (token-specific relevance)
- Validation: Differential Transformer paper shows 35% compute efficiency gain and sparse attention emergence

**Domain B** — Differential Transformer (arXiv 2410.05258)
- Source primitive: DiffAttn = (softmax(Q1K1^T) - λ·softmax(Q2K2^T))V with learned per-layer lambda
- Target bottleneck: Attention noise/overallocation in standard softmax
- Mapping: Two softmax maps with different Q,K projections produce different noise instances; subtraction cancels the common component
- Validation: Outperforms standard Transformer at 65% model size across multiple benchmarks

**Synthesis:** Adapted differential attention for FA3 compatibility by splitting Q,K into halves (64-dim each from 128-dim heads), zero-padding back to 128-dim, running two FA3 calls per layer. QK-norm provides pre-normalized inputs (matched impedance analogy).
**Falsifiability:** "val_bpb will decrease because differential attention cancels noise floor in attention maps, and this would NOT appear if attention is not the bottleneck."
**Code change:** Two FA3 calls with zero-padded half-dim Q,K and shared full-dim V, learned per-layer lambda.
**Outcome:** OOM crash (CUBLAS_STATUS_ALLOC_FAILED). Two FA3 calls per layer at DBS=128 exceeds 80GB VRAM. The zero-padding creates full-size tensors for both calls. The **implementation approach** was wrong — need single-call split-V variant instead, but that changes the mechanism from shared-V differential to split-V differential (different noise cancellation property). The **target bottleneck diagnosis** remains untested.
**Surprise score:** 1 (OOM with doubled attention memory is predictable)

## Run 2 | val_bpb: 0.964137 | delta from baseline: +0.010 | DISCARD

**Diagnosis:** ReluSquared MLP sparsity (~50%+ zeros) means 4× expansion may be below information-theoretic recovery threshold for the sparse intermediate representation.

**Domain A** — Compressive sensing / sparse recovery (Candès, Donoho)
- Source primitive: Restricted Isometry Property — sparse signals need sufficient measurement width for recovery
- Target bottleneck: MLP intermediate dimension may be too narrow given ReluSquared's heavy sparsification
- Mapping: Wider MLP = more "measurements" → better sparse feature recovery
- Validation: Theoretical guarantee for O(s·log(n/s)) measurements

**Domain B** — ReluSquared MLP in transformers
- Source primitive: Binary sparsification of intermediate activations
- Target bottleneck: Effective MLP capacity is ~half nominal due to zero activations
- Mapping: Wider intermediate layer increases non-zero feature count
- Validation: Open question from v2 findings.md

**Synthesis:** 6× MLP expansion (2560→3840 intermediate) to push non-zero activations above recovery threshold.
**Falsifiability:** "val_bpb will decrease because wider MLP increases effective features per token above sparse recovery threshold, and this would NOT appear if attention is the bottleneck."
**Code change:** Changed MLP c_fc from 4×n_embd to 6×n_embd, reduced DBS 128→64 (2 grad_accum) for VRAM.
**Outcome:** +0.010 regression (0.964 vs 0.954). The **mapping was wrong**: the compressive sensing analogy fails because the bottleneck is not MLP representational capacity — it's training steps. With 6× MLP + DBS=64, model got only 1490 steps (vs 1790 baseline), losing 17% of training. The extra 16M params (85.9M→102.2M) couldn't compensate for fewer optimizer updates. Additionally, the DBS reduction forced 2 grad_accum steps per update, adding overhead. The lesson: at this time budget, throughput is king. Any architectural change that reduces step count must provide proportionally more quality per step.
**Surprise score:** 2 (mild surprise that 50% more MLP params couldn't compensate for 17% fewer steps)

## Run 3 | val_bpb: 0.963945 (EMA) / 0.954568 (no EMA) | delta from baseline: +0.010 / +0.001 | DISCARD

**Diagnosis:** Final model snapshot captures a single point on the parameter trajectory. With Muon's rotational updates, averaging should capture the rotation center.

**Domain A** — Polyak-Ruppert averaging (stochastic approximation)
- Source primitive: Iterate averaging achieves optimal O(1/n) convergence
- Target bottleneck: Final snapshot oscillates around optimum
- Mapping: Averaging captures the center of oscillation
- Validation: Schedule-Free (2405.15682) won MLCommons AlgoPerf with this principle

**Domain B** — Schedule-Free Learning (arXiv 2405.15682)
- Source primitive: Unifies iterate averaging and LR scheduling
- Target bottleneck: Cosine warmdown may suboptimally reduce late-training learning
- Mapping: Averaging eliminates need for explicit decay schedule
- Validation: Competitive with cosine schedule on NanoGPT

**Synthesis:** EMA weight averaging (decay=0.995) as lightweight Polyak-Ruppert averaging. The hypothesis was that Muon's norm-preserving (rotational) updates make EMA geometrically meaningful as Fréchet mean approximation.
**Falsifiability:** "val_bpb will decrease because EMA captures center of Muon's rotational oscillations, and this would NOT appear if the model converges to a sharp minimum without oscillation."
**Code change:** Added EMA shadow weights updated after each optimizer step. Evaluated both EMA and non-EMA models.
**Outcome:** EMA val_bpb = 0.964 (worse), no-EMA val_bpb = 0.955 (baseline). The **mapping was wrong**: the cosine warmdown schedule already performs implicit averaging by reducing step size. EMA with decay=0.995 (window ≈ 200 steps) averages over the warmdown phase, blending higher-loss weights from earlier training with the converged final weights. The EMA window spans the LR transition region, producing a mix of under-trained and well-trained weights. A much tighter window (decay=0.9999) or starting EMA only after warmdown might help, but the fundamental issue is that the warmdown schedule already achieves what EMA tries to do. The no-EMA result (0.955) confirms no overhead from the EMA code path.
**Surprise score:** 3 (moderately surprising that EMA actively hurts — expected either improvement or no effect, not degradation. The insight that cosine warmdown IS implicit averaging was not anticipated.)

## Run 4 | val_bpb: 1.747987 | delta from baseline: +0.794 | DISCARD

**Diagnosis:** Each token position receives gradient only from next-token prediction. Richer gradient signal could improve sample efficiency.

**Domain A** — Multi-task learning theory (Caruana 1997; Ruder 2017)
- Source primitive: Auxiliary tasks provide implicit regularization and richer gradients
- Target bottleneck: Single-task gradient signal may be insufficient at this training budget
- Mapping: Auxiliary gradients smooth loss landscape, reducing effective optimization dimensionality
- Validation: Standard result in multi-task learning literature

**Domain B** — Multi-token Prediction (arXiv 2404.19737)
- Source primitive: n independent output heads on shared trunk, each predicting future token
- Target bottleneck: Sample efficiency of next-token prediction
- Mapping: Multiple prediction targets per position = n× gradient signal per forward pass
- Validation: 12% improvement on HumanEval at 13B scale

**Synthesis:** 2-token prediction (n=2) with second lm_head at 0.5× weight. Hypothesis was that Muon's orthogonalization of richer multi-task gradients changes the effective optimization manifold.
**Falsifiability:** "val_bpb will decrease because auxiliary 2nd-token prediction improves shared representations, and this would NOT appear if the trunk already captures sufficient future context."
**Code change:** Added lm_head_2 predicting token t+2. Loss = primary + 0.5 × aux. DBS reduced to 64 for VRAM.
**Outcome:** CATASTROPHIC +0.794 regression (1.748 vs 0.954). The **target bottleneck diagnosis was wrong**: at 85.9M params, the model doesn't have spare capacity for an auxiliary objective. The Meta paper showed benefits at 13B+, not sub-100M. At this scale, the trunk is fully utilized for next-token prediction; adding 0.5× weight to a harder task (2-step-ahead prediction) diverts optimization away from the primary objective. The 2-step-ahead task requires different features (more abstract/anticipatory) than next-token, creating gradient conflict. The 0.5× weight was also too high — with a harder auxiliary task, it should be much lower (0.01-0.1). Additionally, DBS=64 with 2 grad_accum steps gave only 1590 steps vs 1790 baseline.
**Surprise score:** 4 (surprisingly catastrophic — expected mild degradation or marginal improvement, not near-complete training failure. The magnitude reveals that auxiliary objectives can be actively destructive at small scale, not just unhelpful.)

## Run 5 | val_bpb: 0.952837 | delta from baseline: -0.001 | KEEP

**Diagnosis:** x0_lambdas=0.25 was tuned at 10×512 (v2 best). At 10×640 with 5 heads, the optimal input skip strength might be different.
**Rationale:** At 10 layers, stronger input skip (0.30 vs 0.25) preserves more of the initial representation through the deeper stack. The x0_lambdas parameter controls how much of x0 (normalized embedding) is mixed into each layer's input. With 640 model_dim, the residual stream has more capacity, and a stronger skip might help prevent the initial signal from being diluted.
**Prediction:** Small improvement (0.001-0.003) if the bottleneck includes residual stream degradation; no change if 0.25 is already optimal.
**Code change:** Changed x0_lambdas.fill_(0.25) → x0_lambdas.fill_(0.30) in init_weights.
**Outcome:** -0.001 improvement (0.952837 vs 0.953816). At the noise floor (~0.002 seed variance), so this is marginally suggestive but not conclusive. Kept because it's strictly better with zero throughput cost. 1793 steps, same VRAM.
**Surprise score:** 1 (expected marginal change, got marginal change)

## Run 6 | val_bpb: 2.568211 | delta from baseline: +1.615 | DISCARD

**Diagnosis:** Separate wte (5.2M) and lm_head (5.2M) may be redundant — sharing could improve both.

**Domain A** — Information theory: matched encoding/decoding maximizes channel capacity.
**Domain B** — Weight tying in transformers (Press & Wolf 2017).

**Synthesis:** Share wte and lm_head weights. Use embedding LR (0.6×scale) for the shared matrix.
**Code change:** `self.lm_head.weight = self.transformer.wte.weight`, merged optimizer groups, removed separate lm_head init.
**Outcome:** CATASTROPHIC +1.615 regression (2.568 vs 0.953). The **source primitive was wrong for this scale/setup**: weight tying requires compatible initialization (wte uses std=1.0, lm_head needs std=0.001). With the shared std=1.0 init, initial logits have std ≈ 25 (before softcap), causing the model to start with wildly overconfident predictions. The softcap at 12 clips these but the optimization struggles to recover. Additionally, the embedding LR (0.6) is ~150× higher than the original unembedding LR (0.004), which may destabilize the output projection. The model converged to loss ≈ 7.2 (vs baseline ≈ 2.67), indicating it partially recovered but never caught up.
**Surprise score:** 3 (expected modest change, not catastrophic failure. The initialization mismatch is obvious in hindsight but wasn't anticipated — most weight tying implementations use small init for both.)

## Run 7 | val_bpb: 0.953076 | delta from baseline: +0.0002 | DISCARD

**Diagnosis:** Short-window layers (3/4 of SSSL pattern) use 256-token windows. Halving to 128 saves attention FLOPs for more steps.
**Rationale:** Nyquist-inspired: syntactic dependencies span <64 tokens; 128-token window is 2× coverage.
**Prediction:** 5-10% throughput gain → more steps → marginal improvement.
**Code change:** short_window = 256 → 128.
**Outcome:** Neutral (+0.0002, within noise). Throughput gain was minimal: 1807 steps vs 1793 (+14 steps, <1%). FA3 at 256 vs 128 tokens doesn't differ much because the tile-based computation doesn't scale linearly with such small window sizes. The attention context reduction was not compensated by the marginal step gain. The 256→128 change is effectively zero-cost but also zero-benefit.
**Surprise score:** 1 (expected marginal change, got marginal change)
