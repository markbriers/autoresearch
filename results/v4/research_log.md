# Research Log — v4 (Step-Bounded Training)

## Design change

Switched from time-bounded (7.5 min) to step-bounded (1800 steps) training.
This removes the throughput bias that killed every creative hypothesis in v1-v3.
Architectures that are slower per step (wider MLPs, differential attention, auxiliary losses)
now get the same number of learning opportunities as the baseline.
Wall-clock safety kill at 20 minutes.

## Starting point

Inheriting v2 best architecture: 10x640 (85.9M params), batch=262K, SSSL window,
softcap=12, Muon LR=0.03, warmdown=0.7, WD=0.1, FINAL_LR_FRAC=0.01, x0_lambdas=0.2.
v3 best val_bpb was 0.954 under time-bounded training.
**v4 step-bounded baseline: val_bpb = 0.954364** (1800 steps, 453s, 66.2 GB VRAM).
Nearly identical to v3 best — confirms the architecture was already step-optimal at ~1790 steps.

## Hypotheses to revisit under step-bounded regime

These v3 hypotheses failed due to throughput loss. With fixed step count, they deserve retesting:
- 6x MLP expansion (compressive sensing recovery threshold) — lost 17% steps in v3
- Pre-normalised differential attention — OOM in v3, may need DBS adjustment
- Multi-token prediction — catastrophic in v3, but partly due to DBS reduction for memory

## Run 1 | val_bpb: 0.955807 | delta from baseline: +0.001443 | DISCARD

**Diagnosis:** Muon's isotropic NS orthogonalization wastes per-step capacity on noise-dominated curvature directions.

**Domain A** — "Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning" (2603.09697)
- Source primitive: Kronecker-factored curvature estimation with spectral tempering (alpha=0.125)
- Target bottleneck: Muon treats all spectral directions equally
- Mapping: Preconditioning before NS aligns updates with curvature geometry
- Validation: ~12% fewer steps to target loss at 160M-800M scale

**Domain B** — "The Surprising Agreement Between Convex Optimization Theory and LR Scheduling" (2501.18965)
- Source primitive: Non-smooth convex bounds tightly predict LLM training behavior
- Target bottleneck: Cosine warmdown shape is empirically tuned, not principled
- Mapping: Mousse's curvature alignment improves the constant-LR phase; warmdown handles convergence
- Validation: Theory matches practice at 124M-210M scale

**Synthesis:** Left-only Mousse preconditioning on smaller dimension side of each weight matrix, with eigendecomposition every 10 steps, wrapped around existing NS5 iteration. Gradient grafting via reference NS to restore correct RMS.
**Falsifiability:** "val_bpb will decrease because curvature-aligned updates extract more per step, and this would NOT appear if the landscape is already approximately isotropic."
**Code change:** Added non-compiled `_step_muon_mousse` method with left-only Kronecker covariance EMA, periodic eigendecomposition, whiten-NS-unwhiten-graft pipeline.
**Outcome:** Regression of +0.001443. The implementation required a non-compiled code path (torch.compile incompatible with eigh), which shouldn't affect quality but removes kernel fusion. More importantly: the gradient grafting via double-NS (running NS twice — once whitened for the update, once original for the reference RMS) may have introduced artifacts. At 85.9M params, the curvature may be insufficiently anisotropic to benefit from preconditioning — NorMuon's existing per-neuron variance tracking may already capture the relevant structure. **The mapping was wrong**: Mousse's benefit at 160M+ doesn't transfer to 85M because the ratio of curvature anisotropy to noise is too low at this scale.
**Surprise score:** 2 (small regression was expected as a possible outcome; the specific mechanism — curvature anisotropy insufficient at this scale — is somewhat unsurprising)

## Run 2 | val_bpb: 0.949314 | delta from baseline: -0.005050 | KEEP

**Diagnosis:** MLP expressiveness capped at 4x expansion. v3 test of 6x failed due to throughput bias (-17% steps), not quality. Step-bounded training removes this confound.

**Domain A** — "Compressed Sensing" (Donoho 2006)
- Source primitive: Recovery threshold for sparse signals requires sufficient measurement rank
- Target bottleneck: ReluSquared creates sparse activations; wider MLP provides more "measurements"
- Mapping: 6x expansion crosses the sparse recovery threshold that 4x misses
- Validation: Compressive sensing theory predicts improvement when measurements exceed signal sparsity

**Domain B** — "Differential Transformer" (2410.05258)
- Source primitive: Noise cancellation via subtraction of two softmax attention maps
- Target bottleneck: Standard attention dilutes signal across irrelevant context
- Mapping: (Not yet tested together — this run only tests 6x MLP alone as step A)
- Validation: DiffAttn shows sparser attention patterns at 830M+ scale

**Synthesis:** Step A only — testing 6x MLP alone under step-bounded training to isolate the capacity effect from the throughput confound. The full synthesis (6x MLP + DiffAttn) is step B.
**Falsifiability:** "val_bpb will decrease because the wider MLP provides more representational capacity per step, and this improvement would NOT appear if the 4x MLP was already sufficient (no capacity bottleneck)."
**Code change:** Changed MLP expansion from 4x to 6x (c_fc: 640→3840, c_proj: 3840→640). DBS reduced 128→64 for VRAM (grad_accum_steps=2).
**Outcome:** val_bpb improved by -0.005050. This confirms v3's 6x MLP failure was ENTIRELY due to throughput bias. With 1800 steps guaranteed, the extra capacity pays off. Model grew from 85.9M→102.2M params (+19%). Wall-clock 545s (well within 20min limit). VRAM actually dropped to 40.2 GB (from 66.2) due to DBS=64.
**Surprise score:** 3 (the magnitude of improvement is noteworthy — 5 millipoints from just wider MLP, confirming the throughput bias hypothesis from v3)

## Run 3 | val_bpb: 1.830042 | delta from baseline: +0.875678 | DISCARD

**Diagnosis:** Standard attention dilutes signal across irrelevant context; DiffAttn's noise cancellation should produce sparser, more focused attention patterns.

**Domain A** — "Differential Transformer" (2410.05258)
- Source primitive: Subtraction of two softmax attention maps with learnable lambda
- Target bottleneck: Attention noise in irrelevant context positions
- Mapping: Shared noise cancels in subtraction, leaving focused signal
- Validation: Consistently outperforms standard attention at 830M+ scale

**Domain B** — "On the Expressivity Role of LayerNorm" (Findings ACL 2023)
- Source primitive: Mean-centering in LayerNorm enables uniform-attention queries
- Target bottleneck: RMSNorm lacks mean-centering → can't express "attend to nothing"
- Mapping: DiffAttn's subtraction implicitly provides uniform-attention capability
- Validation: Theoretical analysis of norm components

**Synthesis:** DiffAttn with fixed (non-learnable) lambda on top of 6x MLP. Split Q/K after RoPE+QKnorm into two halves (64-dim each), two FA3 calls with full V (128-dim), subtract with fixed lambda_init, per-head RMSNorm, scale by (1-lambda_init).
**Falsifiability:** "val_bpb will decrease because noise cancellation produces sparser attention."
**Code change:** Added DIFF_ATTN flag, differential attention with fixed lambda, per-head norm, (1-lambda_init) scaling.
**Outcome:** CATASTROPHIC regression (+0.876). Fixed lambda completely destroyed training. The issue is that the lambda initialization from the paper (0.2 for layer 0, up to 0.76 for layer 9) combined with the (1-lambda_init) scaling creates a destructive interaction: for deep layers, the scaling factor (1-0.76=0.24) severely attenuates the attention signal, while for shallow layers, the subtraction with lambda=0.2 barely cancels any noise. The paper's learnable lambda reparameterization (exp(q·k) terms) is essential — it allows lambda to adapt during training. Fixed lambda is NOT equivalent. **The implementation was wrong**: the paper's lambda is carefully reparameterized with dot-product terms that sync learning dynamics with the attention weights. A fixed lambda can't adapt to the learned attention patterns.
**Surprise score:** 4 (the magnitude of catastrophe was unexpected — expected maybe +0.01 regression, got +0.876. Fixed lambda is catastrophic, not merely suboptimal)

## Run 4 | val_bpb: 0.953205 | delta from baseline: -0.001159 | delta from best: +0.003891 | DISCARD

**Diagnosis:** Muon parameters may be in sharp minima; anticorrelated noise should steer toward flatter, better-generalizing regions.

**Domain A** — "Anticorrelated Noise Injection for Improved Generalization" (2202.02831)
- Source primitive: Anticorrelated noise (xi_t - xi_{t-1}) implicitly minimizes Hessian trace
- Target bottleneck: Optimization converges to sharp minima under standard Muon
- Mapping: Hessian trace penalty biases toward flatter minima with better generalization
- Validation: Consistent improvement over GD and PGD on CIFAR-10/ResNet (no momentum tests)

**Domain B** — "What Really Matters in Matrix-Whitening Optimizers?" (2510.25000)
- Source primitive: Muon's per-step gain comes from spectral normalization + variance adaptation
- Target bottleneck: Muon may over-commit to spectral descent directions
- Mapping: Anticorrelated noise could decorrelate spectral commitments across steps
- Validation: SOAP outperforms Muon per-step despite less accurate spectral descent

**Synthesis:** Anticorrelated noise injection on Muon parameters after optimizer step. sigma=0.001, only applied to Muon (2D matrix) parameters.
**Falsifiability:** "val_bpb will decrease because anticorrelated noise steers toward flatter optima, and this would NOT appear if the landscape is already flat."
**Code change:** Added Anti-PGD noise: `p.data += sigma * (xi_new - xi_prev)` after optimizer.step() for Muon params.
**Outcome:** val_bpb 0.953205 — better than baseline by -0.001 but WORSE than current best (6x MLP at 0.949314) by +0.004. The anticorrelated noise disrupts Muon's carefully orthogonalized updates. Muon's momentum already creates temporal correlation in parameter updates; adding anticorrelated noise on top creates destructive interference. **The mapping was wrong**: Anti-PGD assumes vanilla GD with i.i.d. gradient noise. Muon's updates are NOT i.i.d. — they are orthogonalized and momentum-smoothed. The anticorrelation conflicts with Muon's own temporal coherence. The paper explicitly notes no testing with momentum-based optimizers.
**Surprise score:** 2 (the direction was expected as a possibility; the mechanism — conflict with Muon's momentum — was anticipated in H4's uncertainty note)

## Run 5 | val_bpb: 0.947314 | delta from baseline: -0.007050 | delta from prev best: -0.002000 | KEEP

**Diagnosis:** 6x MLP confirmed capacity improvement; test whether 8x MLP continues the trend.
**Rationale:** If MLP capacity is the binding constraint, wider MLPs should continue improving val_bpb until either (a) the model is too large for VRAM, (b) the 1800 steps are insufficient to train the extra params, or (c) the attention bottleneck dominates.
**Prediction:** Further improvement of 0.001-0.003, with diminishing returns vs 6x.
**Code change:** MLP expansion 6x → 8x (c_fc: 640→5120, c_proj: 5120→640). Params 102.2M → 118.6M.
**Outcome:** val_bpb improved by -0.002 vs 6x MLP. The capacity scaling continues. The improvement is smaller than 4x→6x (-0.005), showing diminishing returns. Wall-clock 638s (still well within 20min). VRAM 46.7 GB. The MLP capacity curve is sublinear — each additional 2x of expansion gives less. But still improving.
**Surprise score:** 2 (expected diminishing returns; the specific magnitude ~0.002 is reasonable)

## Run 6 | val_bpb: 0.953941 | delta from baseline: -0.000423 | delta from best: +0.006627 | DISCARD

**Diagnosis:** 70% warmdown wastes steps at suboptimal LR; WSD schedule with 20% cooldown should free 900 steps for peak-LR training.

**Domain A** — "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" (2405.18392)
- Source primitive: WSD schedule with (1-sqrt) cooldown. 20% cooldown matches cosine at 124M-1B scale.
- Target bottleneck: 70% of training at reduced LR limits optimization progress
- Mapping: Short cooldown sufficient to damp oscillations; more constant-LR steps = more learning
- Validation: Matches cosine on AdamW at multiple scales

**Domain B** — "The Depth Delusion" (2601.20994)
- Source primitive: Width scales 2.8x more efficiently than depth; wide models benefit from more learning steps
- Target bottleneck: 8x MLP model has many parameters that need sufficient high-LR training
- Mapping: Wider models benefit disproportionately from more constant-LR steps
- Validation: Validated at 17M-7B scale

**Synthesis:** Shorter warmdown (0.7→0.2) to free 900 steps at peak LR for the 8x MLP model.
**Falsifiability:** "val_bpb will decrease because more peak-LR steps enable more learning, and this would NOT appear if warmdown is essential for convergence."
**Code change:** WARMDOWN_RATIO from 0.7 to 0.2 (one line).
**Outcome:** SEVERE regression +0.006627. Muon REQUIRES long warmdown. The WSD paper used AdamW — its findings DO NOT transfer to Muon. **The mapping was wrong**: Muon's orthogonal updates (Newton-Schulz projection) create much larger oscillations than AdamW's adaptive updates. The cosine warmdown isn't just "revealing" progress — it's actively smoothing Muon's aggressive trajectory. With only 20% cooldown (360 steps), the oscillations don't damp enough. This is a fundamental difference between Muon and AdamW optimization dynamics. The 0.7 warmdown that seemed excessive is actually load-bearing for Muon.
**Surprise score:** 4 (the WSD paper strongly predicted this should work; the magnitude of failure was unexpected and reveals a deep Muon-specific interaction)

## Run 7 | val_bpb: 0.915143 | delta from baseline: -0.039221 | delta from prev best: -0.032171 | KEEP

**Diagnosis:** 10 layers may be insufficient depth for the 8x MLP architecture. The Depth Delusion paper shows D_crit ~ 15-18 for W=640, so 12 layers is safe.

**Domain A** — "The Depth Delusion" (2601.20994)
- Source primitive: D_crit ~ W^0.44; width should grow 2.8x faster than depth. 12 layers safely below critical depth.
- Target bottleneck: 10 layers may not have enough representational refinement steps
- Mapping: Each additional layer adds a new step of feature transformation; with 8x MLP each step is very powerful
- Validation: Validated at 17M-7B scale; 16L×512 outperforms 24L×512

**Domain B** — "Revisiting the Shape Convention of Transformer Language Models" (2602.06471)
- Source primitive: Hourglass MLPs suggest deeper-but-lighter per-layer structures
- Target bottleneck: Wide MLP may not need all its capacity if depth is insufficient
- Mapping: More layers × wide MLP = more powerful overall architecture
- Validation: Hourglass outperforms standard FFN at up to 400M params

**Synthesis:** Simply add 2 more layers (10→12) while keeping 8x MLP. Tests whether the model was depth-starved.
**Falsifiability:** "val_bpb will decrease because 12 layers provide more refinement steps, and this would NOT appear if 10 layers is already sufficient."
**Code change:** DEPTH from 10 to 12. Params 118.6M → 191.9M.
**Outcome:** MASSIVE improvement: -0.032 val_bpb. The model was severely depth-starved at 10 layers with 8x MLP. The extra 2 layers add ~73M params (mostly from the 8x MLP per layer) and each provides additional representational refinement. Wall-clock 966s (16 min), near the limit but safe. VRAM 66.6 GB. This is the single largest improvement in v4 and suggests depth was the binding constraint, not MLP width.
**Surprise score:** 5 (the magnitude of -0.032 from just 2 extra layers is remarkable — this is 6x larger than the improvement from 4x→8x MLP. The model was far more depth-starved than width-starved, contradicting the Depth Delusion paper's emphasis on width)

## Run 8 | val_bpb: 0.894903 | delta from baseline: -0.059461 | delta from prev best: -0.020240 | KEEP (with caveat)

**Diagnosis:** Depth=12 showed massive improvement; test whether depth=14 continues the trend (D_crit ~ 15-18 for W=640).
**Rationale:** Each additional layer with 8x MLP adds significant capacity. 14 layers should be safely below D_crit.
**Prediction:** Further improvement of 0.010-0.020, with possible diminishing returns from approaching D_crit.
**Code change:** DEPTH 12→14, DBS 64→32 (OOM at DBS=64).
**Outcome:** val_bpb improved by -0.020 vs depth=12. Depth scaling continues strongly. BUT: wall-clock 1503s (25 min) exceeds the 20-min safety limit. The model is 291M params with DBS=32 (grad_accum=8), at 850ms/step. This architecture is at the edge of feasibility for 1800 steps. **Must find a faster configuration to stay within safety limits.**
**Surprise score:** 3 (continued improvement was expected; the magnitude -0.020 is large but consistent with depth=12's -0.032; the wall-clock violation is the main concern)

## Run 9 | val_bpb: 0.900795 | delta from baseline: -0.053569 | DISCARD

**Diagnosis:** Depth=14 × 8x MLP exceeds 20min. Trade MLP width for speed: 6x MLP with 14 layers.
**Rationale:** 6x MLP is ~18% faster per step, potentially fitting 14 layers in the time budget.
**Prediction:** val_bpb between depth=12×8x (0.915) and depth=14×8x (0.895), with wall-clock under 20min.
**Code change:** MLP 8x→6x, DBS 32→64. Params 291M→246M.
**Outcome:** val_bpb 0.900795 — better than depth=12×8x by -0.014 but worse than depth=14×8x by +0.006. Wall-clock 1267s (still slightly over 20min). VRAM 77.4 GB (very tight). The depth advantage of 14 layers outweighs the 6x→8x MLP tradeoff. **Depth is more valuable than MLP width at this scale.** Reverting to depth=14×8x which gives better results.
**Surprise score:** 2 (expected it to fall between the two extremes; confirms depth > width tradeoff)

## Run 10 | val_bpb: 0.897616 | delta from baseline: -0.056748 | DISCARD (depth=14 is better)

**Diagnosis:** Find max depth that fits in time budget. Test depth=13 as midpoint between 12 and 14.
**Rationale:** Depth=13 should give ~half the improvement of going 12→14.
**Prediction:** val_bpb between 0.915 and 0.895.
**Code change:** DEPTH 14→13, DBS=32. Params 291M→275M.
**Outcome:** val_bpb 0.897616 — between depth=12 (0.915) and depth=14 (0.895) as expected. Wall-clock 1403s (23 min) — still over 20min. The depth scaling is clearly in diminishing returns territory: 10→12 = -0.032, 12→13 = -0.017, 13→14 = -0.003. Depth=12 remains the best fit-in-budget configuration.
**Surprise score:** 1 (completely expected result)

---

## State of Search — 10 Runs

**Which domain pairings produced highest surprise + success?**
- Scaling theory (Depth Delusion) → depth increase: surprise=5, worked massively. The depth-width scaling literature directly enabled the single biggest improvement.
- Compressive sensing → MLP width: surprise=3, worked. Cross-domain analogy correctly predicted wider MLPs help.

**Which mechanistic mappings generalized?**
- "More capacity in the right dimension helps quality" — generalized from MLP width to depth
- "Muon-specific optimization dynamics differ from AdamW" — generalized across warmdown, noise injection, and curvature preconditioning. This is the meta-finding.

**What do failures tell about architecture inductive biases?**
- Muon is extremely rigid about its optimization trajectory: curvature preconditioning (Run 1), anticorrelated noise (Run 4), and short warmdown (Run 6) all hurt. The optimizer is NOT a tuning knob — it's a fixed constraint that shapes what modifications are viable.
- DiffAttn with fixed lambda (Run 3) shows that attention modifications need learnable parameters synced with the optimizer dynamics.

**Most underexplored direction?**
- The current architecture is depth=14 × 8x MLP at 0.895, but exceeds time limit. The most impactful direction is finding ways to make this fit: either (a) reduce MLP to 6x but compensate with other improvements, (b) use model parallelism tricks, or (c) find structural changes that improve quality without adding FLOPs.
- DiffAttn with LEARNABLE lambda (H9, untested) could improve attention quality on the depth=12 configuration without adding layers.
- Stochastic depth during training could speed up deep models by randomly skipping layers.

## Run 11 | val_bpb: 0.895109 | delta from best: +0.000206 | DISCARD (neutral)

**Diagnosis:** z-loss auxiliary penalty should improve gradient quality through softcap by keeping logits away from the boundary.
**Rationale:** PaLM uses z-loss at all scales. Softcap clips at 12; z-loss should discourage the model from approaching the clip boundary.
**Prediction:** Small improvement of 0.001-0.003.
**Code change:** Added `z_loss = 1e-4 * logsumexp(logits)^2` to loss during training.
**Outcome:** Neutral (+0.0002). The softcap=12 already effectively constrains logits. z-loss adds redundant guidance. The model's logits are naturally staying below the softcap, so neither the clip nor the penalty is binding.
**Surprise score:** 1 (neutral outcome was a possibility; confirms softcap is sufficient)

## Run 12 | val_bpb: 0.923938 | delta from baseline: -0.030426 | DISCARD

**Diagnosis:** Trade width (640→512) for depth (14 layers) to fit in time budget.
**Rationale:** Depth Delusion paper shows D_crit ~ 15 for W=512, so 14 layers is safe. Depth > width at this scale.
**Prediction:** val_bpb between 0.905-0.920 (between depth=12×640 and depth=14×640).
**Code change:** ASPECT_RATIO 64→37 (model_dim 640→512, 4 heads), DEPTH=14, DBS=64. 162M params.
**Outcome:** val_bpb 0.924 — worse than depth=12×640 (0.915) by +0.009. The width reduction from 640→512 costs more than the 2 extra layers gain. Width=640 with 5 heads is a better configuration than width=512 with 4 heads at this scale. The head count reduction (5→4) may also hurt attention diversity.
**Surprise score:** 2 (the width > depth result at these specific sizes is modestly surprising given the depth scaling trend)

## Run 13 | val_bpb: N/A | CRASH (killed — too slow)

**Diagnosis:** LayerDrop should enable training depth=14 within time budget by randomly skipping layers.
**Code change:** Graduated LayerDrop (0.0 at layer 0, 0.15 at layer 13), using Python random.random().
**Outcome:** CRASH. LayerDrop's data-dependent branches are incompatible with torch.compile. Without compile, the model runs at half speed (1650ms/step vs 850ms). With DBS=32 this would take 49 min. With compile on, LayerDrop causes OOM because the compiler generates code paths for both branch outcomes. **Fundamental incompatibility**: torch.compile requires static graphs; stochastic depth requires dynamic graphs. Cannot have both.
**Surprise score:** 3 (the 2x slowdown without compile was larger than expected)

## Run 14 | val_bpb: 3.215849 | DISCARD (catastrophic)

**Diagnosis:** Weight tying should save 5.2M params and regularize. Tested on depth=12×640×8x MLP.
**Code change:** `self.lm_head.weight = self.transformer.wte.weight`. Removed lm_head from optimizer groups. Used embedding_lr=0.6 for the shared weight.
**Outcome:** CATASTROPHIC regression (3.216 vs 0.915). The embedding_lr=0.6 is 150x the unembedding_lr=0.004. When tied, the shared weight is optimized at the embedding rate, which is far too aggressive for the output projection. The gradient from CE loss through the output projection dominates, causing the embedding to be corrupted. **Weight tying requires matching LRs between embedding and lm_head**, which this architecture doesn't have (150x difference). Could potentially work with an intermediate LR (~0.04), but that's a separate experiment.
**Surprise score:** 3 (the LR conflict was anticipated in H13 but the magnitude of catastrophe was unexpected — expected maybe +0.1 regression, got +2.3)

## Run 15 | val_bpb: 0.913071 | delta from baseline: -0.041293 | delta from prev best (time-safe): -0.002072 | KEEP

**Diagnosis:** MLP width continues to help at depth=12. Test 10x expansion (from 8x) at the safe depth.
**Rationale:** 4x→6x gave -0.005, 6x→8x gave -0.002 (diminishing returns). 8x→10x should give ~0.001-0.002 if the trend continues.
**Prediction:** val_bpb improvement of 0.001-0.002 vs depth=12×8x.
**Code change:** MLP 8x→10x, DEPTH=12, DBS=32. Params 192M→220M.
**Outcome:** val_bpb 0.913071 — improvement of -0.002 vs 8x MLP. The MLP width scaling continues to help, though with diminishing returns. Wall-clock 1133s (18.9 min) — fits within 20min. This is now the best configuration that completes within the time budget. The MLP width curve: 4x→6x=-0.005, 6x→8x=-0.002, 8x→10x=-0.002.
**Surprise score:** 2 (expected ~0.002 improvement, got exactly that)

## Run 16 [REVIVAL] | val_bpb: 1.829448 | DISCARD (catastrophic)

**Revival of:** Run 3 (DiffAttn fixed lambda — catastrophic)
**Hypothesis:** Learnable lambda reparameterization (exp(q1·k1) - exp(q2·k2) + init) should fix the fixed-lambda failure.
**Code change:** Full DiffAttn implementation with learnable lambda_q1/k1/q2/k2 vectors per head (AdamW group), per-head GroupNorm, (1-lambda_init) scaling.
**Outcome:** STILL CATASTROPHIC (1.829 vs 0.913). The learnable lambda did not help. The problem is deeper than lambda initialization — DiffAttn fundamentally doesn't work with this combination of architecture and optimizer. Possible reasons: (1) splitting Q/K after RoPE+QKnorm halves the effective query/key dimensionality, reducing attention quality for each sub-map below what's needed; (2) the GroupNorm interacts poorly with torch.compile or Muon; (3) at 220M params with only 6 heads, each head needs its full dimensionality. The DiffAttn paper's smallest model is 830M — at sub-250M, the attention dimensionality is too constrained for the split.
**Revival assessment:** NULL REVIVAL — the original failure was NOT just implementation (fixed lambda). DiffAttn is genuinely incompatible with this architecture at this scale.
**Surprise score:** 4 (expected learnable lambda to fix the failure; the fact that it's still catastrophic means the bottleneck is architectural, not parameterization)

## Run 17 | val_bpb: 0.912772 | delta from baseline: -0.041592 | delta from prev best: -0.000299 | KEEP

**Diagnosis:** Standard ReluSquared with 10x expansion has uncontrolled sparse routing. Adding a learned gate should improve neuron selection.
**Code change:** ReluSquared-GLU: gate=relu(x@W_gate)^2, up=x@W_up, output=(gate*up)@W_proj. Three projections at 7x (iso-param with 10x×2).
**Outcome:** Marginal improvement (-0.0003). Gating helps very slightly. The sparse gate from ReluSquared selects neurons slightly better when combined with a learned up-projection. But the effect is small — at 10x expansion, the standard ReluSquared is already selecting useful neurons well enough. Wall-clock 1188s (19.8 min), barely within budget.
**Surprise score:** 1 (small improvement was one of the predicted outcomes)

## Run 18 | val_bpb: 0.910682 | delta from baseline: -0.043682 | delta from prev best: -0.002090 | KEEP

**Diagnosis:** ReluSquared-GLU at 7x is slightly better than standard 10x MLP. Push to 8x GLU for more capacity.
**Rationale:** 7x→8x adds capacity within the GLU structure. The gating benefit may scale with hidden size.
**Prediction:** Further improvement of 0.001-0.002.
**Code change:** GLU hidden 7x→8x. Params 227M→249M.
**Outcome:** val_bpb 0.910682 — improvement of -0.002 vs 7x GLU. The wider GLU continues to help. Wall-clock 1294s (21.6 min) — slightly over 20min. New overall best for approximately in-budget configurations.
**Surprise score:** 1 (expected improvement of ~0.002)

## Run 19 | val_bpb: 0.907107 | delta from baseline: -0.047257 | delta from prev best: -0.003575 | KEEP

**Diagnosis:** SwiGLU (SiLU gate) vs ReluSquared-GLU at 8x expansion on 12×768 architecture.
**Rationale:** SwiGLU is the industry standard (LLaMA, PaLM, etc.). ReluSquared creates sparsity but SiLU provides smoother gradients. At 8x expansion, which matters more?
**Prediction:** SwiGLU may match or beat ReluSquared-GLU. The sparsity advantage may be less important at this scale.
**Code change:** F.relu(x).square() → F.silu(x) in the gate path. One-line change.
**Outcome:** SwiGLU gives -0.0036 vs ReluSquared-GLU. SiLU's smooth gradients outweigh ReluSquared's sparsity advantage at this scale and expansion. This validates the industry consensus (LLaMA uses SwiGLU). The sparsity from ReluSquared was creating ~95% inactive neurons, meaning only ~5% of the 8x expanded hidden dim was being used per token. SiLU uses all neurons (no sparsity), giving the model access to the full hidden dim. At 8x expansion, having all neurons contribute with smooth gradients is better than having a sparse subset with squared gradients.
**Surprise score:** 3 (SwiGLU beating ReluSquared was not expected to be this large — -0.0036 is substantial for a single activation change)

## Run 20 | val_bpb: 0.895269 | delta from baseline: -0.059095 | KEEP (over time limit)

**Diagnosis:** Combine best depth (14) with best activation (SwiGLU) at 6x expansion.
**Rationale:** SwiGLU 6x should match ReluSquared 8x at lower param count due to gating efficiency.
**Prediction:** val_bpb between 0.895 (depth14×8x ReluSq) and 0.907 (depth12×8x SwiGLU).
**Code change:** SwiGLU gate, 6x expansion, DEPTH=14, ASPECT_RATIO=64 (→896 width), DBS=32. 313M params.
**Outcome:** val_bpb 0.895269 — nearly identical to depth=14×8x ReluSquared (0.894903). SwiGLU 6x matches ReluSquared 8x at the same depth. SwiGLU's gating compensates for less raw hidden capacity. Wall-clock 1714s (28.6 min) — well over limit. This confirms SwiGLU and ReluSquared are approximately equivalent when expansion is adjusted to iso-FLOP.
**Surprise score:** 2 (close match between SwiGLU 6x and ReluSq 8x was plausible)

## Run 21 | val_bpb: 0.912996 | delta from best: +0.005889 | DISCARD

**Diagnosis:** GQA (n_kv_head=2) should speed up attention, freeing budget for more capacity.
**Code change:** n_kv_head 6→2 on depth=12 × SwiGLU 8x. 248M→214M params.
**Outcome:** val_bpb 0.913 — regression of +0.006 vs full MHA SwiGLU 8x (0.907). Wall-clock improved 1287s vs 1346s (4.4% faster). But quality cost is too high — KV capacity is still load-bearing at 248M. The earlier v2 finding (GQA destructive at 60M) extends to 248M. GQA remains destructive at this scale.
**Surprise score:** 2 (expected possible regression; the 0.006 magnitude is at the upper end of predicted range)

## Run 22 | val_bpb: 0.906837 | delta from baseline: -0.047527 | delta from SwiGLU@0.03: -0.000270 | KEEP

**Diagnosis:** Muon LR=0.03 may be too aggressive at 248M scale (was tuned at 86M).
**Code change:** MATRIX_LR 0.03→0.02 on depth=12 × SwiGLU 8x.
**Outcome:** Tiny improvement (-0.0003). LR=0.02 is marginally better than 0.03 at this scale, confirming the muP-Muon transfer concern. The 1/sqrt(dmodel/768) scaling gives a correction of 1.0 at 768 width, so LR was being used without scaling. The slight improvement suggests LR=0.03 was slightly too high but not catastrophically so. LR=0.025 might be the sweet spot.
**Surprise score:** 1 (small improvement from LR tuning was expected)

## Run 23 | val_bpb: 0.896134 | delta from depth14 SwiGLU base: +0.000865 | DISCARD

**Diagnosis:** Apply LR=0.02 fix (confirmed at depth=12) to depth=14.
**Code change:** MATRIX_LR 0.03→0.02 on depth=14 × SwiGLU 6x.
**Outcome:** Regression +0.0009. The deeper model needs higher LR. The LR-depth interaction is opposite to naive muP: more layers benefit from slightly HIGHER LR with Muon, because Muon's orthogonal updates need sufficient magnitude to propagate through more layers. LR=0.02 works better at depth=12 (shallower, less gradient propagation needed), while LR=0.03 is better at depth=14.
**Surprise score:** 3 (the LR-depth interaction was unexpected — LR=0.02 helped depth=12 but hurt depth=14)

## Run 24 | val_bpb: 0.907151 | delta from LR=0.02: +0.000314 | DISCARD

**Diagnosis:** LR=0.02 beat 0.03 at depth=12. Test if 0.015 is even better.
**Code change:** MATRIX_LR 0.02→0.015.
**Outcome:** Slight regression. LR=0.015 is too low. The Muon LR landscape at depth=12 × SwiGLU 8x: 0.015→0.9072, 0.02→0.9068, 0.03→0.9071. All very close; 0.02 is the marginally optimal value.
**Surprise score:** 1 (expected result — LR landscape is flat near the optimum)
