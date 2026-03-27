# Hypotheses

## Engineering Run Counter: 4/5 | Phase: ENGINEERING

## Hypothesis 1 | Status: REFUTED | Priority: 1

**Diagnosis:** Muon's Newton-Schulz orthogonalization assumes isotropic curvature, wasting per-step learning on noise-dominated high-curvature directions. This is the optimizer-level bottleneck — each of our 1800 steps could extract more information with curvature-aware updates.

**Domain A:** "Mousse: Rectifying the Geometry of Muon with Curvature-Aware Preconditioning" (2603.09697) — source primitive: Kronecker-factored curvature estimation (Shampoo-style L/R covariance matrices) with spectral tempering (alpha=0.125) wrapped around Muon's NS iteration.

**Domain B:** "The Surprising Agreement Between Convex Optimization Theory and LR Scheduling" (2501.18965) — source primitive: non-smooth convex optimization bounds tightly predict real LLM training behavior; the cooldown phase eliminates logarithmic terms in the bound.

**Synthesis:** Mousse's curvature preconditioning aligns Muon's orthogonal updates with the loss landscape geometry, while the existing cosine warmdown provides the convergence guarantee. The non-obvious combination: Mousse's spectral tempering at alpha=0.125 (NOT the standard Shampoo 0.25) prevents curvature overcorrection that would fight the warmdown schedule — the mild tempering respects the warmdown's role in final convergence while the curvature alignment improves the constant phase. Neither paper proposes this specific interaction.

**Falsifiability:** "val_bpb will decrease because curvature-aligned updates extract more information per step during the constant-LR phase, and this improvement would NOT appear if the loss landscape were already approximately isotropic (in which case standard Muon would be optimal)."

**Prediction:** Loss should decrease faster during the constant-LR phase (steps 0-540), with the gap maintained through warmdown. The final val_bpb improvement should be 0.003-0.010 based on Mousse's reported 12% step reduction at 160M-800M scale. Eigendecomposition every 10 steps adds ~3% wall-clock overhead, well within the 20-min safety limit.

**Implementation sketch:** In `MuonAdamW._step_muon()`: (1) maintain L/R covariance EMA buffers per parameter group, (2) every 10 steps compute eigendecomposition with trace normalization, (3) whiten gradients before NS iteration via `G_tilde = Q_L^T @ G @ Q_R` scaled by `Lambda^{-0.125}`, (4) unwhiten after NS, (5) apply gradient grafting to restore correct RMS norm. Approximately 40 lines of code change, all within the optimizer.

## Hypothesis 2 | Status: CONFIRMED (step A) | Priority: 2

**Diagnosis:** MLP expressiveness is capped at 4x expansion. In v3, 6x MLP failed due to throughput bias (-17% steps), NOT due to quality. With step-bounded training (1800 steps guaranteed), the extra capacity should now help.

**Domain A:** "Compressed Sensing" (Donoho 2006) — source primitive: the recovery threshold for sparse signals requires measurement matrices with sufficient rank relative to signal sparsity. ReluSquared creates sparse activations; wider MLPs provide more "measurements" of the sparse representation.

**Domain B:** "Differential Transformer" (2410.05258) — source primitive: noise cancellation via subtraction of two softmax attention maps produces sparser, more focused attention patterns.

**Synthesis:** 6x MLP alone was tested in v3 and failed due to throughput. The retesting under step-bounded is NOT the synthesis — that's just removing a confound. The synthesis is: if DiffAttn produces sparser attention output (fewer but more relevant tokens contributing), then the MLP receives a cleaner signal with lower effective rank. A 6x MLP can then separate this cleaner sparse signal more effectively than a 4x MLP, because the compressive sensing recovery threshold is easier to meet when the input signal is sparser. Neither paper proposes this attention-MLP interaction.

**Falsifiability:** "val_bpb will decrease because the wider MLP can better recover the sparse signal from attention, and this improvement would NOT appear if the attention output sparsity were the same as standard attention (i.e., 6x MLP without DiffAttn should show smaller gains than 6x MLP with DiffAttn)."

**Prediction:** 6x MLP alone should improve val_bpb by 0.002-0.005 (the quality component that was hidden by throughput loss in v3). Combined with DiffAttn, the improvement should be larger (0.005-0.010) due to the sparsity interaction. Wall-clock will increase to ~12 min (from ~7.5 min) due to more params, but well within 20-min limit.

**Implementation sketch:** Two-stage test: (A) 6x MLP only — change `self.c_fc = nn.Linear(config.n_embd, 6 * config.n_embd)` and `self.c_proj = nn.Linear(6 * config.n_embd, config.n_embd)`. May need DBS reduction to 64 for VRAM. (B) If A succeeds, add DiffAttn: split Q/K into two halves, compute two FA3 calls, subtract with learnable lambda. GroupNorm per head on the output.

## Hypothesis 3 | Status: REFUTED (fixed lambda catastrophic; needs learnable reparameterization) | Priority: 3

**Diagnosis:** Attention heads use full softmax which allocates probability mass to irrelevant context (noise tokens). This is wasted representational capacity, especially critical at 85.9M params where each attention head needs to be maximally informative.

**Domain A:** "Differential Transformer" (2410.05258) — source primitive: differential attention computes `softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T)`, cancelling shared noise across the two maps. Produces sparser attention with reduced activation outliers.

**Domain B:** "On the Expressivity Role of LayerNorm in Transformers' Attention" (Findings ACL 2023) — source primitive: LayerNorm's mean-centering component (which RMSNorm lacks) enables attention to express "attend-to-all-equally" queries via projection to the orthogonal complement of the all-ones vector.

**Synthesis:** DiffAttn's noise cancellation and RMSNorm's missing mean-centering address the SAME bottleneck from opposite directions — attention can't produce clean sparse patterns because (a) both softmax maps attend to noise, and (b) RMSNorm can't express uniform-attention queries that would allow the model to explicitly "not attend." DiffAttn fixes (a) directly. But applying DiffAttn with the existing QK-norm (which uses RMSNorm) may underperform because the QK-norm still can't express uniform queries. However: the subtraction in DiffAttn *implicitly* creates the uniform-attention capability — if both maps converge to the same distribution, the subtraction yields zero (uniform non-attention). This means DiffAttn may partially compensate for RMSNorm's expressivity gap without needing LayerNorm.

**Falsifiability:** "val_bpb will decrease because DiffAttn's noise cancellation produces sparser attention, and this would NOT appear if attention sparsity were already optimal (measurable: if the two softmax maps converge to identical distributions, the cancellation has no effect and lambda→0)."

**Prediction:** val_bpb improvement of 0.003-0.008. The learned lambda should stabilize above 0.5 in most layers (indicating the noise cancellation is active). DiffAttn adds ~9-12% wall-clock per step (two FA3 calls per head), increasing total time from ~7.5 to ~8.5 min — safe under 20-min limit. Parameter count stays nearly identical (half the heads, but each has 2x Q/K projections).

**Implementation sketch:** Replace CausalSelfAttention with DiffAttn: (1) split Q projection into Q1, Q2 (each head_dim/2); (2) split K projection into K1, K2; (3) V stays at full head_dim; (4) call FA3 twice: y1=FA3(Q1,K1,V), y2=FA3(Q2,K2,V); (5) y = y1 - lambda*y2 with learnable lambda per head (init via 0.8-0.6*exp(-0.3*(l-1))); (6) GroupNorm per head; (7) scale by (1-lambda_init). Halve n_head to keep parameter count constant.

## Hypothesis 4 | Status: REFUTED | Priority: 4 [EXPLORATORY]

**Diagnosis:** The Muon optimizer uses Nesterov momentum with fixed beta=0.95, creating temporally correlated updates. But the gradient noise from mini-batch sampling is i.i.d. across steps. Replacing this with anticorrelated noise could steer toward flatter minima, improving generalization.

**Domain A:** "Anticorrelated Noise Injection for Improved Generalization" (2202.02831) — source primitive: adding `(xi_t - xi_{t-1})` (difference of consecutive i.i.d. noise samples) to gradient updates implicitly minimizes the Hessian trace, biasing toward flatter minima. Provably converges where standard PGD diverges.

**Domain B:** "What Really Matters in Matrix-Whitening Optimizers?" (2510.25000) — source primitive: Muon's per-step gains come from two effects — spectral normalization AND variance adaptation. The variance adaptation component is what makes Muon's updates well-scaled.

**Synthesis:** Anticorrelated noise was designed for vanilla GD and never tested with momentum or matrix-whitening optimizers. The non-obvious question: does Muon's Newton-Schulz orthogonalization interact constructively or destructively with anticorrelated perturbations? The NS iteration projects gradients onto the Stiefel manifold — anticorrelated noise in parameter space would create anticorrelated perturbations in the gradient's spectral structure, potentially amplifying the Hessian trace minimization effect along the directions Muon already identifies as important. This is genuinely uncertain — the interaction could also be destructive if the noise disrupts Muon's momentum coherence.

**Falsifiability:** "val_bpb will decrease because anticorrelated noise steers Muon toward flatter optima with lower Hessian trace, and this improvement would NOT appear if the loss landscape near Muon's optimum were already flat (in which case the noise would just add variance without regularization benefit)."

**Prediction:** Uncertain — this is exploratory. If constructive: val_bpb improvement of 0.001-0.005 with negligible compute overhead. If destructive: val_bpb regression of 0.005+ due to noise disrupting momentum coherence. Either outcome is informative about Muon's optimization geometry. Noise scale sigma should be small (~0.001) to avoid overwhelming the optimizer.

**Implementation sketch:** After `optimizer.step()` in the training loop: (1) sample `xi_new` for each Muon parameter, (2) add `sigma * (xi_new - xi_prev)` to parameters, (3) store `xi_new` as `xi_prev`. Approximately 10 lines of code. Memory overhead: 1x Muon parameters (~49M floats, ~200MB). Tune sigma in {0.0001, 0.001, 0.01}.

## Hypothesis 5 | Status: SKIPPED (H1 refuted curvature preconditioning at this scale) | Priority: 2

**Diagnosis:** Each of our 1800 optimizer steps extracts a fixed amount of information from the gradient. Muon's isotropic treatment of curvature directions wastes capacity on noise-dominated directions. Mousse (H1) addresses this with full eigendecomposition, but may be over-engineered for a 10-layer model where the curvature structure is simpler.

**Domain A:** "Mousse" (2603.09697) — source primitive: left-only (single-sided) Kronecker preconditioning achieves nearly the same benefit as full two-sided at lower cost. Key insight: the row covariance L = G @ G^T captures per-neuron gradient statistics.

**Domain B:** "NorMuon variance reduction" (already in train.py) — source primitive: the existing `second_momentum_buffer` in our Muon implementation already tracks per-neuron (or per-dimension) variance statistics for NorMuon-style normalization.

**Synthesis:** The existing NorMuon variance tracking in our code provides a *partial* curvature signal (variance ≈ diagonal of the curvature). Mousse's left-only preconditioning provides the *full* row covariance (off-diagonal correlations between neurons). The synthesis: augment the existing NorMuon variance buffer with off-diagonal covariance information via a left-only Mousse preconditioning step, reusing the existing EMA infrastructure. This is lighter than full Mousse (no eigendecomposition of R) and leverages code structure already present.

**Falsifiability:** "val_bpb will decrease because left-only curvature preconditioning captures inter-neuron gradient correlations that NorMuon's per-neuron variance misses, and this would NOT appear if the gradient covariance were approximately diagonal (in which case NorMuon already captures the relevant structure)."

**Prediction:** Improvement of 0.002-0.006 val_bpb. Left-only variant should add <5% wall-clock (eigendecomp of m×m matrix every 10 steps, where m=640 for most layers). Memory overhead: one m×m matrix per parameter group (~400K floats per group).

**Implementation sketch:** In `muon_step_fused` or a wrapper: (1) maintain L = EMA(G @ G^T) per group, (2) every 10 steps: trace-normalize L, eigendecompose, compute S_L = Lambda^{-0.125}, (3) before NS iteration: G_tilde = Q_L^T @ G (rotate to eigenbasis), scale rows by S_L, (4) after NS: rotate back via Q_L @ result, (5) gradient graft to restore RMS. Changes concentrated in `_step_muon`.

---

# Round 2 Hypotheses

## Hypothesis 6 | Status: REFUTED | Priority: 1

**Diagnosis:** The current LR schedule spends 70% of training (1260/1800 steps) in cosine warmdown with progressively decaying LR. WSD research shows real optimization progress happens during high-LR phases; the decay only "reveals" progress by damping oscillations. We're wasting ~50% of steps at suboptimal learning rates.

**Domain A:** "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations" (2405.18392) — source primitive: WSD (Warmup-Stable-Decay) schedule with (1-sqrt) cooldown shape. 20% cooldown matches or beats full cosine decay at 124M-1B scale. The optimal constant LR is ~half the cosine peak.

**Domain B:** "The Depth Delusion: Why Transformers Should Be Wider, Not Deeper" (2601.20994) — source primitive: Width scaling gives 2.8x more benefit than depth scaling per parameter. Our 8x MLP is already exploiting this — but each additional parameter benefits more from more learning steps at peak LR than from more cooldown steps.

**Synthesis:** The WSD paper uses AdamW, not Muon. The non-obvious interaction: Muon's orthogonal updates are scale-invariant (NS projection removes magnitude information), so Muon should be LESS affected by LR changes than AdamW — the LR multiplier only scales the step size, not the direction. This means Muon may be even more tolerant of short cooldowns than AdamW, because the cosine warmdown's primary benefit (smoothing optimization trajectory) is less important when the optimizer already produces orthogonal updates. A shorter cooldown with Muon could work even better than the WSD paper predicts for AdamW.

**Falsifiability:** "val_bpb will decrease because more steps at peak LR enable more optimization progress, and this improvement would NOT appear if the warmdown is essential for convergence (i.e., if removing warmdown causes divergence or loss spike)."

**Prediction:** Switching from WARMDOWN_RATIO=0.7 to 0.2 should improve val_bpb by 0.002-0.005. The constant-LR phase extends from 30%→80% of training (540→1440 steps at peak LR). The model gets 900 more steps at full learning rate. If the WSD paper's finding transfers to Muon: the (1-sqrt) cooldown shape should outperform cosine over the same 20% window.

**Implementation sketch:** Change WARMDOWN_RATIO from 0.7 to 0.2. Optionally change the cooldown shape from cosine to (1-sqrt): `cooldown = 1 - sqrt((progress - 0.8) / 0.2)`. The WSD paper also suggests MATRIX_LR might need adjustment (constant LR should be ~half of cosine peak), but since our current schedule already has 0% warmup, we should try the simpler change first.

## Hypothesis 7 | Status: UNTESTED | Priority: 2

**Diagnosis:** ReluSquared at 8x expansion creates extreme activation sparsity (~95%), meaning only ~5% of MLP neurons fire per token. This is effectively an uncontrolled sparse routing — which neurons fire is determined solely by the input magnitude, not by a learned gating mechanism. Adding a learned gate could improve which neurons activate without destroying the sparsity.

**Domain A:** "GLU Variants Improve Transformer" (Shazeer, 2002.05202) — source primitive: Gated Linear Units (GLU) use the component-wise product of two projections, one passed through an activation. SwiGLU consistently outperforms ReLU/GELU at iso-parameter counts.

**Domain B:** "ReLU Strikes Back: Exploiting Activation Sparsity" (2310.04564) — source primitive: ReLU-family activations produce ~90% sparsity vs ~0% for SwiGLU/GELU. Sparsity enables inference speedups but is lost with gated activations.

**Synthesis:** The obvious application of GLU would be SwiGLU (replacing ReluSquared), but this kills the sparsity that makes 8x expansion efficient. The non-obvious synthesis: use ReluSquared AS the gate activation in a GLU structure: `output = (ReluSquared(x @ W_gate)) * (x @ W_up)`. This preserves the ~95% sparsity (the gate is sparse) while the up-projection provides a learnable signal that the sparse gate selects. Neither paper proposes this specific combination because GLU papers default to SiLU/GELU gates, and sparsity papers don't consider gated architectures.

However: this requires 3 projections (gate, up, down) instead of 2, so at iso-params the expansion would reduce from 8x to ~5.3x. We need to test whether the gating benefit outweighs the capacity reduction.

**Falsifiability:** "val_bpb will decrease because learned gating improves neuron selection beyond ReluSquared's magnitude-based selection, and this would NOT appear if the gating signal provides no information beyond what ReluSquared's thresholding already captures."

**Prediction:** Uncertain — this is a structural change with competing effects (better gating vs reduced expansion). If the gating helps: -0.002 to -0.005. If capacity loss dominates: +0.002. The sparsity pattern should remain high (~90%+).

**Implementation sketch:** MLP change: `gate = F.relu(self.c_gate(x)).square(); up = self.c_up(x); x = gate * up; x = self.c_down(x)`. Three linear layers: c_gate and c_up project to 5x n_embd (iso-param with current 8x using 2 projections: 2*8=16 vs 3*5.3≈16), c_down projects back to n_embd.

## Hypothesis 8 | Status: UNTESTED | Priority: 3

**Diagnosis:** The model currently has 10 layers × 640 width. The Depth Delusion paper shows width should scale 2.8x faster than depth. With 8x MLP, each layer is very compute-heavy. Reducing depth from 10 to 8 and increasing width to compensate could improve val_bpb if width matters more than depth at this scale.

**Domain A:** "The Depth Delusion" (2601.20994) — source primitive: D_crit ~ W^0.44; width should grow 2.8x faster than depth. Validated at 17M-7B scale.

**Domain B:** "Primer: Searching for Efficient Transformers" (2109.08668) — source primitive: ReluSquared was discovered via NAS as optimal activation. The NAS also found depthwise convolution after Q/K/V projections, suggesting per-layer compute can be traded for quality.

**Synthesis:** Reducing depth from 10→8 frees ~20% of attention+MLP parameters. Redistributing these to width (640→~700, or equivalently keeping 640 but going to 10x MLP expansion) could be more efficient. The Depth Delusion paper shows this is the correct direction, but their calibration assumes 4x MLP. With 8x MLP, the "effective depth" of each layer is higher (more compute per layer), so the optimal depth might be even shallower than their formula suggests.

**Falsifiability:** "val_bpb will decrease because redistributing parameters from depth to width is more efficient, and this would NOT appear if 10 layers is already below the optimal depth for this width (i.e., the model is depth-starved, not width-starved)."

**Prediction:** 8 layers × 768 width (or 8 layers × 640 with 10x MLP) should give a small improvement (0.001-0.003). Risk: 8 layers may be too shallow for the attention mechanism to build sufficient representations.

**Implementation sketch:** Change DEPTH from 10 to 8 and increase ASPECT_RATIO from 64 to 96 (giving 768 width, 6 heads at HEAD_DIM=128). Or keep width at 640 and increase MLP to 10x.

## Hypothesis 9 | Status: UNTESTED | Priority: 2

**Diagnosis:** DiffAttn with fixed lambda failed catastrophically (Run 3). But the core mechanism (noise cancellation via subtraction of two attention maps) is sound — the paper validates it at 830M+ scale. The failure was in the implementation (fixed lambda), not the concept. With proper learnable lambda, DiffAttn should work.

**Domain A:** "Differential Transformer" (2410.05258) — source primitive: The paper's lambda reparameterization uses exp(lambda_q1 · lambda_k1) - exp(lambda_q2 · lambda_k2) + lambda_init, where the dot-product terms synchronize lambda learning dynamics with attention weights. This is essential for stability.

**Domain B:** Run 3 failure analysis — the (1-lambda_init) scaling at deep layers (0.24 for layer 9) severely attenuated the signal, while the fixed lambda couldn't adapt to learned attention patterns.

**Synthesis:** Implement DiffAttn with the full learnable lambda reparameterization from the paper. The lambda parameters (small per-head vectors) should be treated as scalar parameters in the optimizer (Adam, not Muon) to avoid conflicts with Muon's orthogonal updates. The key difference from Run 3: lambda adapts during training, so the initial (1-lambda_init) scaling is temporary — lambda will learn to produce the right subtraction strength.

**Falsifiability:** "val_bpb will decrease because learnable lambda adapts the noise cancellation to the learned attention patterns, and this would NOT appear if the two Q/K subspaces learn identical attention distributions (lambda→0, reducing to standard attention)."

**Prediction:** With learnable lambda on the 8x MLP base, val_bpb improvement of 0.002-0.005. The implementation is more complex but addresses the exact failure mode from Run 3.

**Implementation sketch:** (1) Add learnable lambda_q1, lambda_k1, lambda_q2, lambda_k2 vectors per head (shape [sub_dim]) initialized to zeros. (2) Compute lambda = exp(q1·k1) - exp(q2·k2) + lambda_init. (3) In setup_optimizer, collect these as scalar params (Adam group). (4) Same FA3 two-call approach as Run 3 but with learnable lambda.

## Hypothesis 10 | Status: CONFIRMED | Priority: 1 [EXPLORATORY]

**Diagnosis:** The current 8x MLP with 10 layers uses DEPTH=10, ASPECT_RATIO=64 (640 width). But the MLP expansion is set as a constant multiplier (8x). What if we increase depth to 12 while keeping 8x MLP? The Depth Delusion paper shows D_crit ~ 18 for W=640, so 12 layers is safely below. More layers means more steps of feature refinement, and each step has a very powerful 8x MLP.

**Domain A:** "The Depth Delusion" (2601.20994) — D_crit for W=640 is ~15-18 layers. We're well below at 10 layers.

**Domain B:** "Revisiting the Shape Convention of Transformer Language Models" (2602.06471) — Hourglass MLPs suggest reallocating MLP params to increase hidden dim/depth.

**Synthesis:** Simply add 2 more layers (10→12) with 8x MLP. This adds ~24M params (2 * (640*5120 + 5120*640 + attention) ≈ 24M). Total becomes ~143M params. The extra depth allows more representational refinement while staying safely below D_crit. Wall-clock increases proportionally but stays within 20min. DBS may need to be reduced to 48 or 32 if OOM.

**Falsifiability:** "val_bpb will decrease because 12 layers provide more feature refinement steps, and this would NOT appear if 10 layers is already sufficient to represent the data (no underfitting from depth)."

**Prediction:** val_bpb improvement of 0.003-0.008 from the additional capacity and depth. Risk: VRAM constraints may force DBS reduction, adding grad_accum overhead.

**Implementation sketch:** Change DEPTH from 10 to 12. May need to reduce DBS to 48 or 32. The model should auto-configure correctly via build_model_config.

---

# Round 3 Hypotheses

## Hypothesis 11 | Status: INCONCLUSIVE (crashes: LayerDrop incompatible with torch.compile, 2x slowdown) | Priority: 1

**Diagnosis:** Depth=14 × 8x MLP achieves val_bpb 0.895 but exceeds 20min (25min actual). The time bottleneck prevents using our best architecture. LayerDrop can reduce effective forward cost by randomly skipping layers during training.

**Domain A:** "Reducing Transformer Depth on Demand with Structured Dropout" (1909.11556) — source primitive: LayerDrop drops each transformer layer with fixed probability p during training. Model learns to be robust to missing layers. Effective forward cost per step = n_layers * (1-p).

**Domain B:** "Deep Networks with Stochastic Depth" (1603.09382) — source primitive: Linearly increasing drop rates (early layers safe, deeper layers dropped more). Reduces training time ~25% while improving test error via regularization.

**Synthesis:** Train depth=14 with graduated LayerDrop: drop rates from 0.0 (layer 0) to 0.2 (layer 13). Expected layers per forward pass ≈ 14 * (1 - 0.1) ≈ 12.6. This should reduce per-step time from 850ms to ~765ms (14→12.6 effective layers), total from 1530s to ~1377s ≈ 23min. Still over 20min, but the regularization effect may compensate by improving quality per step. Combined with 8x MLP, each layer that DOES execute has full capacity.

The non-obvious interaction: Muon's residual lambda parameters (which scale residual connections per layer) interact with stochastic depth. When a layer is dropped, its contribution to the residual stream is zero, but the residual_lambdas still apply to the preceding accumulation. This could create training signal for the lambdas to learn which layers are most critical.

**Falsifiability:** "val_bpb will improve (or match depth=14 without LayerDrop) because regularization compensates for fewer effective layers, and this would NOT appear if the depth=14 model were not overfitting."

**Prediction:** val_bpb 0.900-0.910 (between depth=12 dense and depth=14 dense). Wall-clock ~23min (still tight). The regularization may give better generalization than a dense depth=12.

**Implementation sketch:** In Block.forward(), wrap the attention+MLP in a stochastic skip during training: `if self.training and random.random() < self.drop_rate: return x`. Set drop_rate = layer_idx * 0.2 / (n_layer - 1) for each layer. At eval, scale by (1 - drop_rate). ~5 lines of code.

## Hypothesis 12 | Status: INCONCLUSIVE (neutral: +0.000206, softcap already suffices) | Priority: 1

**Diagnosis:** The depth=12 × 8x MLP configuration (0.915, fits in time) could benefit from z-loss auxiliary penalty. At 192M params, the logits may drift to large magnitudes during training, causing softmax saturation and suboptimal gradients.

**Domain A:** "PaLM: Scaling Language Modeling with Pathways" (2204.02311) — source primitive: z-loss adds `1e-4 * log(sum(exp(z)))^2` to the loss, penalizing logit magnitude drift. Zero extra params, near-zero FLOPs.

**Domain B:** Current architecture uses softcap=12 (tanh(logits/12)*12), which already constrains logit range. But softcap is a hard constraint that clips, while z-loss is a soft penalty that guides the model to naturally produce moderate logits.

**Synthesis:** The non-obvious question: does z-loss help when softcap already constrains logits? Softcap clips logits above 12 but doesn't penalize the model for producing logits near 12. z-loss would encourage logits to stay centered, potentially reducing the information loss from softcap clipping. The two mechanisms are complementary, not redundant: softcap prevents overflow, z-loss prevents the model from relying on overflow. This should improve gradient quality through the softcap.

**Falsifiability:** "val_bpb will decrease because z-loss improves gradient quality by keeping logits away from the softcap boundary, and this would NOT appear if logits are already well-centered (i.e., softcap never clips)."

**Prediction:** Small improvement of 0.001-0.003 val_bpb. Zero overhead.

**Implementation sketch:** After computing loss in the forward method, add: `z_loss = 1e-4 * torch.logsumexp(logits, dim=-1).square().mean()`. Add to the total loss. ~2 lines.

## Hypothesis 13 | Status: REFUTED (LR conflict: embedding_lr=0.6 is 150x too high for output projection) | Priority: 2

**Diagnosis:** The model has separate wte (5.2M params) and lm_head (5.2M params). Tying them saves 5.2M parameters and provides implicit regularization (the input representation IS the output prediction).

**Domain A:** "Using the Output Embedding to Improve Language Models" (Press & Wolf, 1608.05859) — source primitive: Weight tying between input/output embeddings reduces perplexity and provides regularization. The tied embedding evolves more like the output embedding.

**Domain B:** "MobileLLM: Optimizing Sub-billion Parameter Language Models" (2402.14905) — source primitive: Weight tying is identified as critical for sub-billion parameter LMs. The saved params can be reallocated to increase depth or width.

**Synthesis:** Tie wte and lm_head, saving 5.2M params. On a 192M model, this is a 2.7% reduction. The regularization benefit should outweigh the reduced capacity. The saved params aren't reallocated (iso-compute test). The non-obvious interaction with Muon: the wte is trained with AdamW while lm_head is also trained with AdamW (but at different LRs — embedding_lr=0.6 vs unembedding_lr=0.004). Tying them forces a single LR, which might conflict. Need to use the embedding LR since the lm_head parameters are shared.

**Falsifiability:** "val_bpb will decrease because tying provides regularization at minimal capacity cost, and this would NOT appear if the model needs separate input/output representations."

**Prediction:** Small improvement of 0.001-0.002 or slight regression of 0.001. The LR conflict is the main risk.

**Implementation sketch:** `model.lm_head.weight = model.transformer.wte.weight`. Remove lm_head from parameter groups, ensure embedding group covers both. Adjust init_weights to not reinit lm_head separately.

## Hypothesis 14 | Status: REFUTED (width=512 too narrow, loses more than 2 extra layers gain) | Priority: 3 [EXPLORATORY]

**Diagnosis:** The current depth=14 × 8x MLP at 291M params takes 25min. What if we increase depth to 14 and REDUCE model_dim from 640 to 512 (4 heads instead of 5), keeping 8x MLP? This reduces per-layer FLOPs while keeping the depth advantage. Params ≈ 14 * (512 * 4096 * 2 + 512 * 512 * 4 + ...) ≈ 190M. Should fit in ~17min.

**Domain A:** "The Depth Delusion" (2601.20994) — D_crit ~ W^0.44. For W=512, D_crit ~ 15. So 14 layers at W=512 is still below critical depth.

**Domain B:** v4 depth scaling results — depth is far more valuable than width in this architecture.

**Synthesis:** Trade width for depth to stay within time budget. 14 × 512 × 8x MLP should be ~190M params, fitting in ~17min. This tests whether the depth advantage holds at narrower width.

**Falsifiability:** "val_bpb will decrease because 14 layers at W=512 outperforms 12 layers at W=640 (depth > width), and this would NOT appear if width is more important than depth at this scale."

**Prediction:** val_bpb between 0.905-0.920. If better than depth=12×640 (0.915), depth > width confirmed at this scale.

**Implementation sketch:** DEPTH=14, ASPECT_RATIO=36 or 37 (gives 504 or 518, round to 512 for head_dim=128 compatibility → 4 heads). DBS=64 should fit. Adjust HEAD_DIM if needed.

---

# Round 4 Hypotheses (includes Island Model Revival)

## Hypothesis 15 [REVIVAL] | Status: REFUTED (still catastrophic even with learnable lambda) | Priority: 1

**Revival of:** Run 3 (DiffAttn with fixed lambda — catastrophic failure)
**Why it may succeed now:** (1) The original failure was due to fixed lambda, not the DiffAttn concept. With learnable lambda reparameterization (dot-product form from the paper), lambda can adapt during training. (2) The model is now 220M params (vs 86M), where attention improvements should matter more.

**Implementation:** DiffAttn with learnable lambda per head, on the current depth=12 × 10x MLP base. Lambda reparameterized as `exp(lambda_q1 · lambda_k1) - exp(lambda_q2 · lambda_k2) + lambda_init`. Lambda vectors treated as scalar params (AdamW, not Muon). Split Q/K after RoPE+QKnorm into halves, two FA3 calls, per-head GroupNorm, (1-lambda_init) scaling.

**Prediction:** val_bpb improvement of 0.002-0.005 vs the 10x MLP base (0.913).

## Hypothesis 16 | Status: UNTESTED | Priority: 2

**Diagnosis:** The 10x MLP expansion at depth=12 uses ~220M params. The MLP expansion is at 10x with ReluSquared. H7 proposed ReluSquared-GLU (gated variant) but was never tested. With 10x expansion, iso-param GLU would use ~6.7x (10*2/3) expansion. The gating could improve quality enough to offset the reduced expansion.

**Implementation:** MLP change: `gate = F.relu(self.c_gate(x)).square(); up = self.c_up(x); x = gate * up; x = self.c_down(x)`. Three linear layers: c_gate and c_up project to 7*n_embd (≈iso-param with 10x*2 projections), c_down projects back.

**Prediction:** Uncertain. If gating helps: -0.002 to -0.005. If capacity loss dominates: +0.002.

---

# Round 5 Hypotheses

## Hypothesis 17 | Status: UNTESTED | Priority: 1

**Diagnosis:** The time budget is the binding constraint. Attention is ~30% of per-step cost. GQA (fewer KV heads) reduces attention memory bandwidth and compute without significant quality loss. This could free time budget for more depth.

**Domain A:** "GQA: Training Generalized Multi-Query Transformer Models" (EMNLP 2023) — GQA with 1-2 KV heads matches full MHA quality while approaching MQA speed. Already wired into the architecture (n_kv_head is configurable).

**Domain B:** v4 depth scaling results — each added layer gives massive improvement. If attention is 15% faster, depth=14 × SwiGLU could fit in 20-22 min.

**Synthesis:** Reduce n_kv_head from full (5-7, matching n_head) to 1-2. FA3 natively supports GQA. This speeds up attention by reducing KV projection FLOPs and memory bandwidth. On the current depth=12 × SwiGLU 8x architecture, this may free enough time to either (a) add more depth, or (b) widen the MLP. The earlier finding that "GQA is destructive at 60M params" (v2 Run 69) was at a much smaller scale; at 248M+ params, the quality cost of fewer KV heads should be much smaller.

**Falsifiability:** "val_bpb will stay within 0.002 of the full-MHA baseline while wall-clock decreases by 5-10%, and this speedup would NOT appear if attention compute were already a small fraction of total cost."

**Prediction:** val_bpb regression of 0.001-0.003, wall-clock improvement of 5-10%. Net win if the saved time enables more depth/width.

**Implementation sketch:** In GPTConfig, set n_kv_head=2 (or 1) while keeping n_head=5-7. The architecture already supports this. One-line config change.

## Hypothesis 18 | Status: UNTESTED | Priority: 2

**Diagnosis:** The Muon LR (0.03) was tuned at the 86M param scale (10×640×4x MLP). The model is now 248M+ params with 12-14 layers. muP-style LR transfer may not hold for Muon (paper "Controlled LLM Training on Spectral Sphere" suggests Muon is only "half-aligned" with muP). A quick LR probe could unlock significant gains.

**Implementation sketch:** Test MATRIX_LR=0.02 and MATRIX_LR=0.04 on the current depth=12 × SwiGLU 8x architecture. Two quick probes.

**Prediction:** If LR is sub-optimal: one of these will improve val_bpb by 0.002-0.005. If LR is already optimal: both will regress.
