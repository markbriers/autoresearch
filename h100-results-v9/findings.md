# Findings

## Confirmed Mechanisms

### SwiGLU Activation with Wider Hidden Dim (H6) -- CONFIRMED, delta=-0.004964
**What:** Replaced ReluSquared MLP activation with SwiGLU (gated SiLU), using hidden_dim=1792 (a ~5% parameter increase over baseline). This is a follow-up to H1 (hidden_dim=1664, delta=-0.00246, INCONCLUSIVE).

**Why it works:** SwiGLU's learned multiplicative gating provides better feature selection than ReluSquared's fixed sparsity pattern. The gate (sigmoid of a linear projection) learns which features to suppress, whereas ReluSquared's squaring operation indiscriminately zeros out all negative features. The critical finding is that width was the binding constraint in H1: going from 1664 to 1792 (7.7% wider) nearly doubled the delta (from -0.0025 to -0.005). This superlinear response indicates that H1 was operating in a width-starved regime where the gating quality was throttled by insufficient hidden capacity.

**Relation to prior work:** This is the strongest single-intervention result in the programme (delta=-0.005), surpassing H4's PD residual scaling (delta=-0.004). It modifies only the MLP activation and hidden dimension, leaving the residual stream untouched. This orthogonality with H4 makes stacking highly plausible.

**Stacking potential:** High. SwiGLU modifies the MLP interior (activation function and hidden dim); H4 modifies the residual stream dynamics (derivative scaling). The two interventions touch non-overlapping code paths. VRAM: H6 used 67.5 GB, H4 used 69.3 GB. Stacking both would likely use ~69-70 GB (the VRAM cost is dominated by the derivative term's extra tensor, and SwiGLU at 1792 is actually lighter than baseline ReluSquared at 2560 for activations). A stacking experiment should be prioritised.

---

### PD Residual Scaling (H4) -- CONFIRMED, delta=-0.004197
**What:** Adding a learnable derivative term to the residual stream, inspired by PD controllers from control theory. Each layer computes velocity = x - x_prev and adds deriv_lambda * velocity to the residual stream before the block. deriv_lambdas are per-layer scalars initialized at 0, trained with scalar_lr * 0.01.

**Why it works:** The derivative term provides anticipatory correction -- it biases the residual stream in the direction of its recent trajectory, acting as a form of layer-wise momentum. The very low learning rate (scalar_lr * 0.01) constrains the derivative coefficients to remain small, avoiding noise amplification. This means the correction is gentle and stabilizing rather than destabilizing.

**Relation to prior work:** This is the first confirmed result in the residuals subsystem. It demonstrates that the residual stream has exploitable layer-wise temporal structure even at depth=10. The closely related H3 (predictor-corrector momentum) was INCONCLUSIVE (delta=-0.000554), which is interesting: both add inter-layer coupling, but the PD formulation (velocity before the block) outperformed the predictor-corrector formulation (previous delta after the block). The key difference is that PD modifies the input to each block (anticipatory), while predictor-corrector modifies the output (retrospective).

**Stacking potential:** The intervention adds only 10 scalar parameters and 69.3 GB VRAM. It modifies only the residual stream path, leaving attention, MLP, and activation functions untouched. It should stack with activation function changes (e.g., SwiGLU if a follow-up confirms it). The VRAM headroom (76 - 69.3 = 6.7 GB) is tight but sufficient for one additional lightweight intervention.

### Factored Embeddings with 256-dim Bottleneck (H14) -- CONFIRMED, delta=-0.003215
**What:** Replaced the full embedding table (vocab_size x 640 = 32.2M params) with a factored embedding: (vocab_size x 256 = 12.9M) followed by a linear projection (256 x 640 = 0.16M, optimized by Muon). Total embedding params reduced from 32.2M to 13.1M (60% reduction). Overall model params reduced from ~84M to ~65M.

**Why it works:** Two mechanisms likely contribute: (1) Regularization through parameter reduction. At 1800 training steps, the full 32.2M embedding table is under-trained -- many tokens appear rarely, and their 640-dim embeddings are poorly learned. The 256-dim bottleneck forces all tokens through a lower-dimensional space, acting as implicit regularization that prevents overfitting to rare token embeddings. (2) Muon-optimized projection. The wte_proj matrix (256x640) is optimized by Muon (Newton-Schulz orthogonalization), which may produce a better linear map than AdamW could learn for the full embedding. Muon's near-orthogonal weight constraint on the projection means the 256-dim embedding is mapped to 640 dims via a well-conditioned transformation that preserves distances.

**Why both agents were wrong:** Both agents dismissed the regularization mechanism. The Evaluator wrote "this is not a regularization-limited regime at 1800 steps" -- this was incorrect. With 32.2M embedding params and only 1800 steps, most of the vocab (8192 tokens) gets limited gradient updates. The bottleneck pools information across embedding dimensions, reducing the effective parameter count that needs to be learned. The Researcher predicted delta=-0.001, underestimating the benefit by 3.2x.

**Relation to prior work:** This is the first confirmed result in the embeddings subsystem and the third confirmed improvement in the programme (after H6/SwiGLU and H4/PD residual). It is the first parameter-reducing intervention to succeed -- both prior confirmations added parameters.

**Stacking potential:** High. The factored embedding modifies only the input embedding layer. It is fully orthogonal to SwiGLU (MLP activation), PD residual (residual stream dynamics), and Peri-LN (normalisation). The 19M parameter reduction may actually help stacking by providing regularization headroom. Priority: test stacking with H10 (SwiGLU + PD, the current best configuration at val_bpb=0.953648).

---

### Stacking Ceiling: H10 (SwiGLU+PD) remains the best configuration

Cycle 5 attempted to extend the best known configuration (H10, delta=-0.005692) by adding factored embeddings (H14). The result was catastrophic for the stacking thesis:
- H15 (SwiGLU+PD+factored): val_bpb=0.956253 -- WORSE than H10 (0.953648) by +0.002605
- H19 (SwiGLU+factored): val_bpb=0.956265 -- WORSE than H6 (0.954376) by +0.001889

Adding factored embeddings to any SwiGLU configuration degrades performance. The mechanism: SwiGLU's c_gate learns to select features from the input representation. Factored embeddings compress the input through a 256-dim bottleneck, reducing feature diversity. With fewer distinct input features, the gate becomes less effective -- it cannot discriminate what it cannot see. Both H15 and H19 achieve ~0.956, which is approximately H14 alone (0.956125), suggesting SwiGLU contributes nearly nothing when factored embeddings constrain the input rank.

H10 (SwiGLU+PD, val_bpb=0.953648) remains the best configuration. The stacking frontier has been mapped: SwiGLU + PD compose subadditively (37% discount) but positively. Factored embeddings are incompatible with SwiGLU and should not be stacked with gated activations.

---

## Dead Ends

### Divisive Normalization after ReluSquared (H2) -- REFUTED, delta=+0.004629
**What failed:** Dividing ReluSquared activations by the sqrt of their local channel-pooled mean (groups of 64 channels).

**Why it failed:** ReluSquared produces highly sparse activations (many exact zeros). Divisive normalization divides by a pooled mean that is near-zero in sparse regions, amplifying noise. In dense regions, it suppresses the dominant active channels, destroying the sparse feature selection that ReluSquared provides. The biological analogy (cortical divisive normalization) breaks down because cortical neurons have dense firing rates, not sparse binary-like activations.

**Lesson:** Do not apply normalization-style operations to sparse intermediate activations. Any intervention in the MLP hidden layer must respect the sparsity structure that ReluSquared creates. This blocks: group normalization, local response normalization, and similar pooling-then-dividing schemes applied after ReluSquared.

### Per-Head Attention Temperature (H8) -- REFUTED, delta=+0.004604
**What failed:** Adding a learnable per-head temperature scalar (init=1.0) that scales Q vectors after QK-norm and before the attention matmul. 50 scalar parameters total (5 heads x 10 layers).

**Why it failed:** The +0.005 regression from such a minimal intervention is severe. The most likely explanation is that per-head temperature creates a degenerate optimisation landscape when combined with QK-norm + softcap. QK-norm constrains Q and K to unit norm, and the softcap clips logits. The temperature scalar sits between these two constraints, creating a narrow corridor where the only effect is to either (a) push logits toward the softcap ceiling (high temperature, causing uniform attention via saturation) or (b) flatten attention patterns (low temperature). Neither direction helps. The optimizer may have pushed temperatures in harmful directions before the learning rate could correct. Implementation risk: if head_temps were accidentally grouped with Muon (shape (5,) is 1D and Newton-Schulz requires 2D), the parameters could have been corrupted. This should be verified.

**Lesson:** Do not add scalar temperature controls to the attention path when QK-norm and softcap are both active. The existing logit conditioning already determines an effective temperature, and a learnable scalar has no productive direction to move. More broadly, multiplicative interventions on Q/K that are absorbable into the QK projection weights may be inherently redundant in this architecture.

### LAWA/EMA Weight Averaging (H12) -- REFUTED, delta=+0.003935 (last-iterate), EMA val_bpb=1.142475
**What failed:** Maintaining an EMA shadow copy of all model weights (beta=0.999) and evaluating with the EMA weights. The EMA val_bpb was catastrophically bad (1.142, +19% regression). The last-iterate also regressed by +0.004 despite the training procedure being supposedly identical to baseline.

**Why it failed:** beta=0.999 creates an averaging window of ~1000 steps, covering more than half the 1800-step training run. The EMA incorporates weights from the high-LR warmup phase (steps 0-540) where weights are far from converged. With a 70% warmdown schedule (steps 540-1800), the last ~1260 steps already provide implicit averaging through decaying learning rate. EMA on top of warmdown is redundant at best and destructive when the beta is too wide. The last-iterate regression (+0.004) is unexplained: it may be seed variance, torch.compile graph differences from the lerp_ calls, or bf16 numerical perturbation from the shadow param updates.

**Lesson:** Weight averaging with broad beta is incompatible with warmdown-dominated short training schedules. The warmdown already serves the same purpose (reducing noise in late updates). If EMA is to be attempted again, beta must be much narrower (0.99 or 0.995, ~100-200 step window) and applied only during the warmdown phase.

### QK-Norm Linf Relaxation (H13) -- REFUTED, delta=+0.007378
**What failed:** Replacing L2 QK-normalization with Linf normalization (x / max|x|) plus 1/sqrt(head_dim) scaling on Q. This changes the constraint geometry from hypersphere (L2) to hypercube (Linf).

**Why it failed:** The +0.007 regression is the worst in the programme. Three mechanisms likely contribute: (1) Linf norm has sparse gradients (only the argmax dimension receives gradient), creating training instability that degrades attention pattern quality over 1800 steps. (2) The 1/sqrt(128) = 0.0884 scaling on Q heavily compresses the logit range, reducing the effective resolution of attention scores within the softcap=15 window. (3) The hypercube constraint surface has sharp corners at the vertices, creating a non-smooth optimization landscape that Muon's Newton-Schulz (designed for smooth weight manifolds) handles poorly.

**Lesson:** L2 QK-norm is well-suited to this architecture. The smooth, rotationally symmetric constraint surface of L2 provides better gradient flow than Linf's sparse-gradient hypercube. The attention logit regime (QK-norm + softcap) is rigid: both adding temperature (H8) and changing the norm function (H13) produce large regressions. Future attention modifications must avoid changing the logit distribution -- only structural changes (head count, attention pattern) remain viable.

### Peri-LN Output Normalization (H11) -- INCONCLUSIVE, delta=-0.000522
**What did not work well enough:** Adding RMSNorm after each sublayer output (attention and MLP) before adding to the residual stream, bounding the variance of sublayer contributions.

**Why it was weak:** At depth=10, the variance cascade that Peri-LN addresses is not severe. The paper demonstrating Peri-LN's benefits operates at 1.5B+ scale with 24+ layers, where exponential variance growth in Pre-LN is problematic. At 10 layers, the cumulative variance growth is modest and Pre-LN alone is sufficient. The delta (-0.000522) is nearly identical in magnitude to H3 (-0.000554) and H7 (-0.000701) -- all three are "technically directional improvements that are practically zero."

**Lesson:** Output normalization on sublayer contributions is not beneficial at depth=10. The normalisation subsystem is not a bottleneck at this scale. Additionally, the intervention costs 4.8 GB VRAM (72.6 vs 67.8 baseline) despite adding zero parameters, due to torch.compile activation memory for the extra norm calls. This unfavorable cost-benefit ratio makes Peri-LN unattractive even as a marginal addition.

### Factored Embeddings + SwiGLU: Negative Interaction (H15, H19) -- INCONCLUSIVE
**What failed:** Stacking factored embeddings (H14) with SwiGLU (H6) in two configurations: H15 (SwiGLU+PD+factored, delta=-0.003087) and H19 (SwiGLU+factored, delta=-0.003075). Both are worse than their SwiGLU-only counterparts (H10: -0.005692, H6: -0.004964).

**Why it failed:** SwiGLU's gating mechanism (c_gate) learns to select features based on the diversity of input representations. The 256-dim factored embedding bottleneck reduces the rank of the first-layer input, limiting the feature diversity available to the gate. With constrained input rank, the c_gate projection cannot learn discriminative gating patterns. The result: SwiGLU degenerates to near-linear behavior, losing its gating advantage over ReluSquared.

**Lesson:** Gated activations (SwiGLU and likely other GLU variants) require high-rank input features to function effectively. Do not combine input compression (factored embeddings) with gated activations. This is a hard constraint: the interaction is not merely subadditive but actively destructive. Factored embeddings remain viable with non-gated activations (e.g., ReluSquared + factored embeddings = H14, delta=-0.003215, CONFIRMED).

### Tail-Phase EMA (H17) -- REFUTED (null effect, baseline contaminated)
**What failed:** EMA weight averaging during warmdown only (beta=0.99, steps 540-1800). The EMA weights (0.956294) were worse than the last iterate (0.956133). The experiment was run on a contaminated baseline (H14 factored embeddings already present).

**Why it failed:** Same root cause as H12: the warmdown schedule already provides optimal implicit averaging. At beta=0.99 (100-step window), the EMA captures a local neighborhood of the weight trajectory, but the declining learning rate already makes consecutive weight updates near-identical. EMA of nearly-identical weights is the same as the last iterate. The Muon off-manifold concern (EMA of near-orthogonal matrices is not itself near-orthogonal) may also contribute, though the narrow window minimizes this effect.

**Lesson:** Weight averaging is definitively unproductive in this training regime. Two attempts (H12: broad window, H17: narrow window + warmdown only) both failed. The warmdown schedule is the sufficient averaging mechanism. The training-loop/weight-averaging subsystem is BLOCKED.

### 192-dim Factored Embeddings (H16) -- INCONCLUSIVE, delta=-0.000563
**What did not work well enough:** Narrowing the factored embedding bottleneck from 256-dim (H14, CONFIRMED) to 192-dim.

**Why it was weak:** The 192-dim bottleneck crosses the rate-distortion knee. At 256-dim, the bottleneck removes noise (poorly-learned embedding dimensions) without losing discriminative token information. At 192-dim, the bottleneck begins to lose token-distinguishing information, degrading downstream task performance. The sharp cliff between 256-dim (-0.003215) and 192-dim (-0.000563) -- a 0.002652 difference from just 64 fewer dimensions -- indicates the knee is steep.

**Lesson:** The optimal bottleneck for token embeddings at 1800 training steps is in the 224-256 range. Going below 192 dims will be counterproductive. Further compression experiments in this regime have low expected value.

### Leaky Integral Residual Accumulator (H7) -- INCONCLUSIVE, delta=-0.000701
**What did not work well enough:** Adding a leaky integral term (decay=0.9) to the residual stream that accumulates all prior block output deltas and adds integ_lambda * integral at each layer.

**Why it was weak:** The integral accumulates stale information from early layers. At depth=10 with decay=0.9, the integral at layer 9 carries 39% of layer 0's delta, 43% of layer 1's delta, etc. These early-layer signals are contextually irrelevant to late-layer processing. The integ_lambdas (init=0) likely stayed near zero because the gradient signal for learning useful integral coefficients was too weak. This mirrors H3's failure (retrospective momentum, delta=-0.0006): both are retrospective modifications that accumulate past information, and both produced negligible results.

**Lesson:** Retrospective residual modifications (accumulating history) do not work at depth=10. Only anticipatory (forward-looking) modifications work. See the pattern analysis under "Architecture Inductive Biases" below.

---

## Architecture Inductive Biases

### Residual stream has exploitable layer-wise structure at depth=10
H4's success shows that consecutive layers produce correlated updates to the residual stream -- the "velocity" (difference between consecutive states) contains predictive signal. This is surprising at depth=10 because the ODE-integration framing of residual networks is typically motivated by depth-50+ networks. The implication is that even shallow transformers have smooth enough layer-wise dynamics to benefit from derivative-based correction.

### Sparse activations are fragile to mid-layer normalization
H2's failure shows that ReluSquared's sparsity pattern is load-bearing. Interventions that modify the magnitude distribution of hidden activations (without respecting sparsity) actively degrade performance. This constrains the space of viable MLP modifications.

### Anticipatory vs retrospective residual modifications: only anticipatory works
Three experiments now test this distinction:
- H4 (derivative/velocity, anticipatory): CONFIRMED, delta=-0.004. Modifies residual BEFORE the block based on rate of change.
- H3 (predictor-corrector momentum, retrospective): INCONCLUSIVE, delta=-0.0006. Modifies residual AFTER the block based on previous single delta.
- H7 (leaky integral, retrospective): INCONCLUSIVE, delta=-0.0007. Modifies residual AFTER the block based on accumulated history.

The pattern is unambiguous. Retrospective modifications (H3, H7) produce near-zero deltas despite different mechanisms (1-step lookback vs exponential accumulation). Anticipatory modification (H4) produces a clear improvement. The explanation: at depth=10, each block's output is a locally optimal update given its input. Modifying the block's input (anticipatory) allows the block to compute a better update. Modifying the block's output (retrospective) fights the block's already-computed answer. This constrains all future residual-stream hypotheses to anticipatory formulations.

### Attention temperature is tightly constrained by QK-norm + softcap
H8's regression (+0.005) from 50 learnable temperature scalars reveals that the attention logit distribution is already tightly controlled by QK-norm (constraining input magnitude) and softcap (constraining output magnitude). Inserting a learnable temperature between these two constraints has no productive direction. Future attention modifications should either change the attention pattern structure (not just its temperature) or remove one of the existing constraints.

### Attention logit regime is globally rigid (H8 + H13)
Two mechanistically different attention modifications -- scalar temperature (H8, +0.005) and norm function change (H13, +0.007) -- both produced large regressions. The common thread: both alter the distribution of attention logits. H8 scales logits multiplicatively; H13 changes the geometric constraint that determines logit range. Both fail. The attention logit distribution is tightly controlled by QK-norm (constraining input magnitude) and softcap (constraining output magnitude). Any intervention that changes the logit distribution disrupts the careful balance between these constraints. The regression magnitude escalates with intervention severity: H8 (50 scalars, +0.005) < H13 (norm function replacement, +0.007). This suggests the attention subsystem will resist modifications with increasing severity as they become more fundamental. Only structural changes that preserve the logit distribution (e.g., different head count, window pattern) remain viable.

### Depth=10 is too shallow for normalisation interventions (H11)
Peri-LN output normalization (H11, delta=-0.000522) tested whether bounding sublayer output variance improves training. The trivial improvement confirms that the normalisation subsystem is not a bottleneck at this scale. The Friis cascade analogy (noise accumulates across stages) is correct in principle, but at 10 stages the cumulative effect is too small to matter. This joins the pattern of "shallow network is robust to variance/noise problems": H3 (layer-wise truncation error, delta=-0.0006), H7 (accumulated residual drift, delta=-0.0007), and H11 (variance cascade, delta=-0.0005) all address problems that are severe at 50+ layers but negligible at 10.

### Confirmed improvements can interact NEGATIVELY -- stacking is not monotone (H15, H19)
The most important finding of cycle 5. The assumption underlying all stacking experiments -- that confirmed improvements compose positively (at worst subadditively) -- is violated. Adding factored embeddings (H14, CONFIRMED, delta=-0.003215) to SwiGLU (H6, CONFIRMED, delta=-0.004964) produces a result WORSE than SwiGLU alone:
- H6 alone: 0.954376
- H19 (H6+H14): 0.956265 -- worse by +0.001889
- H15 (H6+H4+H14): 0.956253 -- worse than H10 (H6+H4, 0.953648) by +0.002605

**Cycle 6 update (H22):** The hypothesis that the negative interaction was "specific to the gated-activation + input-compression combination" has been FALSIFIED. H22 (PD+factored, no SwiGLU, val_bpb=0.956004) is worse than H4 alone (PD only, val_bpb=0.955143) by +0.000861. Factored embeddings degrade PD performance even though PD does not use input-dependent gating and ReluSquared is a non-gated activation. The degradation magnitude correlates with the stacking partner's sensitivity to residual stream quality:
- PD alone: +0.000861 degradation from adding factored (lowest -- PD's velocity signal is partially insensitive to input rank)
- SwiGLU alone: +0.001889 degradation from adding factored (medium -- SwiGLU's gate depends on input feature diversity)
- SwiGLU+PD: +0.002605 degradation from adding factored (highest -- both components degraded simultaneously)

The revised mechanism: the 256-dim embedding bottleneck constrains the rank of the ENTIRE residual stream trajectory, not just the first-layer input. Every component that exploits the residual stream's full-rank structure is degraded. PD's velocity signal (x - x_prev) is computed between consecutive layer states; when the initial state is lower-rank, the velocity vectors span a lower-dimensional subspace, providing less informative anticipatory corrections. SwiGLU's gate is additionally degraded because it explicitly selects features by rank.

Factored embeddings (H14) should be understood as a STANDALONE regularizer only. It works in isolation because the baseline model's embedding layer is over-parameterized. But it is fundamentally incompatible with any stacking configuration because the bottleneck constrains the information available to all downstream improvements. The stacking/with-factored-embeddings subcategory is now tested with 3 experiments (H15, H19, H22) across 3 different stacking partners, with 0 positive interactions. This subcategory is permanently BLOCKED.

### Confirmed improvements are subadditive at ~37% discount (H10)
H6 (SwiGLU, delta=-0.005) and H4 (PD residual, delta=-0.004) individually confirmed. Stacked as H10, they produced delta=-0.005692, which is 63% of the sum of individual deltas (-0.009). The 37% discount implies the two subsystems are not fully orthogonal despite modifying different code paths. The likely mechanism: SwiGLU changes the gradient magnitude profile flowing through the MLP, which affects the gradient signal that deriv_lambdas receive for learning velocity coefficients. The gradient coupling occurs at the residual stream, where both interventions meet. This has implications for all future stacking: expect ~60% of the additive sum, not 100%.

### Embedding layer is over-parameterized at this scale (H14)
The full embedding table (32.2M params, 38% of total model params) can be compressed 60% to a 256-dim bottleneck without loss -- in fact, with a gain. This reveals that the effective rank of the learned embedding is well below 256 at 1800 training steps. The embedding is the largest single parameter block in the model and it is undertrained. Factored embeddings with Muon-optimized projection provide implicit regularization that helps. This is the first evidence that the model is regularization-limited in at least one subsystem.

### Width bottleneck for gated activations is severe at small scale
H1 (SwiGLU, hidden_dim=1664) vs H6 (SwiGLU, hidden_dim=1792) shows a superlinear response: 7.7% more width produced 100% more delta improvement (-0.0025 to -0.005). This implies that gated activations like SwiGLU have a minimum viable width below which the gating mechanism cannibalises compute without providing enough benefit. At dim=640, this threshold is somewhere between 1664 and 1792.

## Cross-Domain Transfer Patterns

### Control theory -> residual connections: PRODUCTIVE
H4 transferred the derivative term from PD controllers to residual stream scaling. This is the first confirmed cross-domain transfer. The key adaptation was using a very low learning rate for the derivative coefficients, analogous to derivative gain tuning in control systems (where excessive derivative gain causes oscillation).

### Neuroscience -> activation functions: UNPRODUCTIVE (so far)
H2 attempted to transfer divisive normalization from cortical computation. The transfer failed because the analogy between cortical firing rates (dense, continuous) and transformer hidden activations (sparse, after ReluSquared) is too weak. Future neuroscience-inspired transfers should account for the sparsity regime of the target activation.

### Control theory -> residual connections: PRODUCTIVE (strengthened)
H7 (leaky integral) failed alongside H4's success. This sharpens the control theory analogy: the derivative term is productive, but the integral term is not. In classical PID control, the integral term eliminates steady-state error in systems with persistent bias. Transformers at depth=10 apparently do not have persistent steady-state error in the residual stream -- each block fully processes its input. The derivative term works because it captures trajectory smoothness, not error correction.

### Statistical mechanics -> attention temperature: UNPRODUCTIVE
H8 attempted to transfer the Boltzmann temperature concept (heterogeneous temperatures in non-equilibrium ensembles) to per-head attention. The transfer failed because the attention logit distribution is already tightly controlled by QK-norm + softcap, leaving no room for temperature variation. The analogy to statistical mechanics assumes the system has a free energy landscape to explore; the constrained logit space does not.

### Information theory (rate-distortion) -> factored embeddings: PRODUCTIVE
H14 transferred the rate-distortion concept (optimal compression rate for redundant sources) to token embeddings. The bottleneck_dim=256 forces the embedding through a lower-rank representation, and the result improved val_bpb. This confirms the rate-distortion intuition: the embedding's effective information content at 1800 steps is well below its nominal dimensionality (640), and compressing to the bottleneck eliminates noise without losing task-relevant signal. This is the second productive cross-domain transfer (after H4's PD controller).

### Compressed sensing (Lp norm relaxation) -> attention normalization: UNPRODUCTIVE
H13 transferred the Lp norm relaxation concept (different norms favor different solution geometries) to QK-normalization. The transfer failed because the attention mechanism's requirement for smooth gradient flow is incompatible with Linf's sparse gradients. The compressed sensing analogy assumes the norm choice affects only the constraint geometry; in practice, it also affects gradient flow, which is critical for learning.

### Telecommunications (Friis cascade noise) -> output normalization: UNPRODUCTIVE
H11 transferred the Friis noise figure formula (cascaded amplifier stages accumulate noise from early stages) to sublayer output normalization. The analogy is correct for deep networks but inapplicable at depth=10, where the cascade is too short for variance accumulation to matter. Cross-domain transfers from cascaded-system theory should account for the number of stages (layers) before predicting benefit.

### Bias-variance tradeoff (James-Stein estimator) -> weight averaging: UNPRODUCTIVE (confirmed by H17)
H12 transferred the bias-variance tradeoff concept (temporal shrinkage reduces variance) to weight averaging via EMA. The transfer failed because the warmdown schedule already provides the variance-reduction benefit that EMA would offer, and the broad EMA window (beta=0.999) introduces excessive bias by averaging over unconverged weights. H17 retested with a narrow window (beta=0.99, warmdown only) following the Evaluator's recommendations. Result: still null effect. The warmdown schedule is the dominant averaging mechanism and explicit EMA is redundant regardless of beta or phase restriction. This cross-domain transfer is definitively closed.

## Open Questions

1. **What values did the deriv_lambdas converge to in H4?** If they are layer-dependent (e.g., larger in early layers, smaller in late layers), this would reveal where in the network the derivative correction matters most.

2. **Does SwiGLU (H6) + PD residual scaling (H4) stack?** H6 achieved delta=-0.005 (activation/MLP) and H4 achieved delta=-0.004 (residual stream). They modify orthogonal subsystems. If they stack additively, the combined delta could reach -0.009. If they stack subadditively, the combined delta would still likely exceed -0.006. This is the highest-priority experiment for cycle 3.

3. ~~Is the predictor-corrector framing (H3) salvageable?~~ CLOSED. H7's failure alongside H3's failure confirms that retrospective residual modifications do not work at depth=10. Only anticipatory (derivative) modifications are productive.

4. ~~Why did SwiGLU underperform expectations (H1)?~~ ANSWERED by H6. Width was the bottleneck. hidden_dim=1792 (vs 1664) crossed the threshold.

5. ~~Was H8's regression caused by an implementation bug or a genuine architectural incompatibility?~~ LARGELY RESOLVED. H13's independent regression (+0.007) from a completely different attention modification confirms the attention subsystem is genuinely rigid. Even if H8 had an implementation bug, the pattern holds: attention logit modifications regress.

6. **What is the minimum viable hidden dim for SwiGLU at this scale?** The threshold is between 1664 and 1792. This is relevant for future parameter-matched designs.

7. ~~**Does the H14 factored embedding benefit stack with H10 (SwiGLU + PD)?**~~ ANSWERED: NO. H15 (triple stack) performed worse than H10 (double stack). Factored embeddings negatively interact with SwiGLU. See cycle 5 findings.

8. ~~**What is the optimal bottleneck_dim for factored embeddings?**~~ PARTIALLY ANSWERED by H16: the rate-distortion knee is between 192 and 256 dims. 256-dim is near-optimal. Going lower is counterproductive.

9. ~~**Why are confirmed improvements subadditive at 37% discount?**~~ PARTIALLY ANSWERED: the interaction is worse than subadditive -- it can be negative. The mechanism for the SwiGLU-factored interaction is input rank starvation: gated activations need high-rank input. The 37% subadditive discount for H6+H4 (H10) remains the best stacking result. Adding a third component (H14) does not continue the 37% pattern -- it makes things worse.

10. **Does H14 (factored embeddings) stack with H4 (PD residual) WITHOUT SwiGLU?** The negative interaction was between factored embeddings and SwiGLU's gating. PD residual does not depend on input rank. If PD+factored works, the negative interaction is SwiGLU-specific. If it also fails, factored embeddings are incompatible with stacking in general. This is the key disambiguation experiment.

11. **Are there other improvements that stack positively with H10 (SwiGLU+PD)?** The stacking frontier is H10 at val_bpb=0.953648. Factored embeddings failed. What other interventions could extend this? Possibilities: learning rate schedule modifications, different head counts, or positional encoding changes -- all in untested subsystems.
