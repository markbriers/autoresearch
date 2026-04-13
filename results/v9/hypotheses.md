# Hypotheses

## Engineering Run Counter: 0/5 | Phase: RESEARCH

---

## Hypothesis 1: SwiGLU Activation | Status: INCONCLUSIVE

### Sprint Contract

**Intervention:** Replace ReluSquared MLP activation with SwiGLU (gated SiLU), using 8/3 expansion ratio to match parameter count.
**Subsystem:** activation, MLP
**Papers:** [Domain A] Shazeer 2020, "GLU Variants Improve Transformer" (arXiv:2002.05202) x [Domain B] Carandini & Heeger 2012, "Normalization as a canonical neural computation" (Nature Reviews Neuroscience) -- divisive normalization in cortex uses multiplicative gating of neural responses by pooled activity, directly analogous to GLU's element-wise product of a linear projection and a sigmoidal gate.
**Closest prior art:** SwiGLU is standard in LLaMA/Gemma. Prior knowledge says "SwiGLU beats ReluSquared per-step (v4)." However, no run has been done at this exact config (DEPTH=10, dim=640, Muon). The v4 result was at different scale. Re-testing at current frozen config is warranted because per-step vs per-FLOP tradeoffs change with model scale.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.005
**Predicted confidence:** medium
**Case for failure:** SwiGLU uses 3 matrix projections (gate, up, down) vs 2 (up, down) for ReluSquared. At matched parameter count (8/3 ratio), the hidden dimension shrinks from 4x to ~2.67x. The reduced hidden dimension may hurt expressivity at this small scale. Also, Muon's Newton-Schulz orthogonalization operates on the weight matrices, and the interaction with gated activations is untested. SiLU may also produce different gradient magnitude profiles than ReluSquared, potentially destabilizing Muon momentum.
**Feasibility pre-flight:** 3 weight matrices per MLP layer instead of 2, but total params roughly matched. Hidden dim goes from 2560 to ~1706 (round to 1706 = 640*8/3, nearest even). VRAM similar. torch.compile should handle SiLU fine -- it's a standard op.
**Implementation sketch:** Replace MLP class. Add c_gate linear layer. Forward: gate = silu(c_gate(x)); up = c_fc(x); x = gate * up; x = c_proj(x). Adjust hidden_dim to 8*n_embd//3 (round to nearest multiple of 128 for efficiency).

**Information gain analysis:**
- P(success): 0.55
- If CONFIRMED, I learn: SwiGLU's gating mechanism provides better feature selection than ReluSquared's implicit sparsity at this scale/optimizer combination.
- If REFUTED, I learn: ReluSquared's extreme sparsity (squaring zeros out more neurons) is actually preferable at small scale with Muon, suggesting the optimizer benefits from sparser gradients.
- Expected information gain: high -- both outcomes update beliefs about activation-optimizer interaction.

**Evaluator P(success):** 0.45
**Belief divergence:** 0.10

**Evaluator notes:** Implementation is straightforward. Hidden dim should round to nearest multiple of 128 (1664, not 1706). Ensure init_weights covers c_gate. The Muon grouping-by-shape will naturally pick up the new c_gate weights since they share shape with c_fc. setup_optimizer assertion will hold since c_gate is inside transformer.h. Main risk: hidden dim shrinks from 2560 to 1664 at matched params, which is a 35% reduction in MLP width. At this small scale, width may matter more than gating quality.

---

## Hypothesis 2: Divisive Normalization in MLP (SquaredReLU + Channel Pooling) | Status: REFUTED

### Sprint Contract

**Intervention:** Add a divisive normalization step after ReluSquared activation: divide hidden activations by a local channel-pooled mean (with small epsilon), inspired by cortical divisive normalization.
**Subsystem:** activation, MLP
**Papers:** [Domain A] Shazeer 2020, "GLU Variants Improve Transformer" (arXiv:2002.05202) -- gating as learned normalization x [Domain B] Carandini & Heeger 2012, "Normalization as a canonical neural computation" (Nature Reviews Neuroscience) -- divisive normalization where neural response r_i is divided by sigma^2 + sum(w_j * r_j^2), creating competition between channels.
**Closest prior art:** BatchNorm/LayerNorm normalize across channels but use additive centering + multiplicative scaling. Divisive normalization is fundamentally different: it divides by the pooled activity of nearby neurons, creating competitive dynamics without learnable affine parameters. SqueezeNet's SE blocks are related but operate at a different granularity.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003
**Predicted confidence:** low
**Case for failure:** Divisive normalization reduces the dynamic range of hidden activations, which may conflict with ReluSquared's intentional sparsification. The operation is essentially x / (eps + pool(x)), which could dampen the signal that ReluSquared provides. Also, adding a new non-learnable operation in the middle of the MLP may interact poorly with Muon -- the Newton-Schulz step assumes clean gradient flow through the weight matrices, and injecting a nonlinear normalization between them could distort gradient geometry. torch.compile should handle this since it's just division and pooling, no new parameters.
**Feasibility pre-flight:** Zero new parameters. Small VRAM increase from the intermediate pooling tensor. torch.compile compatible (avg_pool1d or reshape+mean).
**Implementation sketch:** After ReluSquared in MLP forward: pool the squared activations over groups of channels (e.g., groups of 64), take mean, add eps=1e-6, divide hidden activations by sqrt of this pool. This is: h = relu(x).square(); pool = h.reshape(B,T,G,-1).mean(-1,keepdim=True).expand_as(h.reshape(B,T,G,-1)).reshape_as(h); h = h / (pool.sqrt() + 1e-6).

**Information gain analysis:**
- P(success): 0.25
- If CONFIRMED, I learn: Inter-channel competition (a la cortical lateral inhibition) improves feature selectivity even in small transformers, suggesting the model is bottlenecked on feature competition rather than raw capacity.
- If REFUTED, I learn: The gradient flow disruption from mid-MLP normalization outweighs any selectivity benefit, confirming Muon's sensitivity to gradient pathway changes.
- Expected information gain: medium -- low P(success) means failure is expected and somewhat less informative.

**Evaluator P(success):** 0.15
**Belief divergence:** 0.10

**Evaluator notes:** Zero new parameters, no optimizer changes needed, torch.compile compatible. Numerical stability concern: ReluSquared produces many exact zeros, so pooled mean over groups could be very small in sparse regions. The eps=1e-6 may be insufficient. Recommend the implementer verify the division does not produce NaN/Inf in early steps. The implementation sketch is clear enough. Low-risk experiment (no new params, identity-like with appropriate epsilon), but low expected benefit.

---

## Hypothesis 3: Predictor-Corrector Residual Connections | Status: INCONCLUSIVE

### Sprint Contract

**Intervention:** Replace the simple residual connection x = lambda*x + f(x) with a second-order predictor-corrector step inspired by Heun's method: predict = x + f(norm(x)), then correct = x + 0.5*(f(norm(x)) + f(norm(predict))). Simplified version: use a momentum-like term that blends the current block output with the previous block's output, approximating a higher-order ODE integrator.
**Subsystem:** residuals
**Papers:** [Domain A] "Predictor-Corrector Enhanced Transformers with EMA Coefficient Learning" (arXiv:2411.03042) -- shows higher-order ODE integration improves transformer performance x [Domain B] Classical numerical analysis: Heun's method / improved Euler method for ODEs -- the key insight is that Euler's method (standard residual) has O(h^2) local truncation error while Heun's method achieves O(h^3), reducing accumulated error across layers.
**Closest prior art:** The 2411.03042 paper does this for translation/summarization but uses a complex multi-step framework with learnable EMA coefficients. We simplify to a single-step correction using the previous block's output as a predictor, keeping the implementation minimal and Muon-compatible.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.004
**Predicted confidence:** low
**Case for failure:** The ODE analogy to transformers is imperfect -- transformer layers are not truly discretizing a continuous dynamical system, and the "step size" is not well-defined. A second evaluation of the block function doubles the compute per layer, which at 10 layers would be catastrophic for wall-clock time. The simplified version (blending with previous block output) avoids the double-evaluation cost but may not capture the corrector benefit. Also, storing previous block outputs increases VRAM. With only 10 layers, the accumulated truncation error savings may be negligible.
**Feasibility pre-flight:** Simplified version adds zero parameters and minimal VRAM (one extra residual stream tensor). Full Heun's method would double compute and is infeasible within time budget. The simplified momentum blend needs only a per-layer scalar, compatible with existing scalar_lr optimizer group.
**Implementation sketch:** In GPT.forward, maintain prev_delta (previous block's output delta). For each block: delta = block(x, ve, cos_sin, window_size); x = resid_lambda * x + x0_lambda * x0 + delta + beta * prev_delta; prev_delta = delta. Beta is a small learnable scalar (init 0.0) per layer, optimized with scalar_lr group. First layer uses prev_delta=0.

**Information gain analysis:**
- P(success): 0.30
- If CONFIRMED, I learn: The residual stream benefits from higher-order integration, meaning inter-layer dynamics are smooth enough to exploit temporal (layer-wise) coherence. This would open a family of multi-step methods.
- If REFUTED, I learn: Layer-wise dynamics are too discontinuous (or 10 layers too few) for higher-order integration benefits to manifest, constraining future residual-stream hypotheses.
- Expected information gain: high -- this is a clean test of whether the ODE perspective on residual streams holds at small depth.

**Evaluator P(success):** 0.20
**Belief divergence:** 0.10

**Evaluator notes:** Implementation sketch has an ambiguity: Block.forward applies residual connections internally (x = x + attn(...); x = x + mlp(...)), so "delta" must be computed as x_after_block - x_before_block in GPT.forward, not returned by block. The implementer must: (1) save x_before = x.clone() before block call, (2) compute delta = x - x_before after block call, (3) add beta * prev_delta. This clone adds VRAM but is small (B*T*dim). Beta init=0 makes this safe at startup. New beta params (10 scalars) must be added to setup_optimizer -- update the assertion on line 247 and add a param group (scalar_lr). With only 10 layers the momentum signal is weak. Feasible but marginal.

---

## Hypothesis 4: Proportional-Derivative Residual Scaling | Status: CONFIRMED

### Sprint Contract

**Intervention:** Replace the static resid_lambda/x0_lambda scaling with a PD (proportional-derivative) controller-inspired mechanism: the residual scaling adapts based on both the current residual magnitude (proportional) and its rate of change across layers (derivative). Specifically, compute a per-layer "velocity" term as the difference between consecutive residual stream states, and use it to modulate the scaling.
**Subsystem:** residuals
**Papers:** [Domain A] "Predictor-Corrector Enhanced Transformers with EMA Coefficient Learning" (arXiv:2411.03042) -- adaptive coefficient learning for residual connections x [Domain B] Classical control theory: PID controllers -- the derivative term in PID control provides anticipatory correction by responding to the rate of change of the error signal, reducing overshoot and oscillation.
**Closest prior art:** The existing resid_lambdas are a proportional-only controller (fixed gain per layer). Adding a derivative term is novel in this context. DeepNet's alpha scaling is also proportional-only.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003
**Predicted confidence:** low
**Case for failure:** The derivative term requires storing the previous layer's residual stream state and computing a difference, which adds computation. More critically, the derivative of a discrete sequence (layer outputs) is noisy and may not provide useful signal -- in control theory, derivative control is notorious for amplifying noise. Per-dimension scaling conflicts with Muon (known dead end), but this is per-LAYER scaling which uses the scalar_lr group and should be safe. However, any input-dependent gating could create gradient flow issues.
**Feasibility pre-flight:** Adds 10 scalar parameters (one per layer) for the derivative coefficient. VRAM: one extra tensor the size of the residual stream. torch.compile: should be fine since it's just subtraction and multiplication with a scalar.
**Implementation sketch:** In GPT.forward, add self.deriv_lambdas = nn.Parameter(torch.zeros(n_layer)). In forward loop: velocity = x - x_prev (where x_prev is x from previous layer iteration); x = resid_lambda * x + x0_lambda * x0 + deriv_lambda * velocity; then proceed with block. x_prev updated each iteration. First layer uses velocity=0.

**Information gain analysis:**
- P(success): 0.20
- If CONFIRMED, I learn: The residual stream exhibits layer-wise momentum patterns that a derivative term can exploit, suggesting the model benefits from smoothing its layer-wise trajectory.
- If REFUTED, I learn: Layer-wise velocity is too noisy or uninformative at this depth, confirming that simple fixed-gain residual connections are near-optimal for shallow transformers.
- Expected information gain: medium -- low P(success) but the derivative/velocity concept is novel enough that either outcome is informative.

**Evaluator P(success):** 0.15
**Belief divergence:** 0.05

**Evaluator notes:** Very similar in spirit to H3 -- both add a term based on inter-layer dynamics to the residual stream. deriv_lambda init=0 is safe. Same implementation concern as H3: new scalar params must be added to setup_optimizer with updated assertion. The derivative of a 10-step discrete sequence is inherently noisy. Both the Researcher and I agree this is unlikely to work (low divergence), which means this experiment has lower expected information gain than H3. Feasible to implement.

---

## Hypothesis 5: SwiGLU + Predictor-Corrector Combo | Status: REJECTED

### Sprint Contract

**Intervention:** Combine H1 (SwiGLU activation) with H3 (predictor-corrector residual momentum) in a single run, testing whether the two orthogonal improvements stack.
**Subsystem:** activation, MLP, residuals
**Papers:** Combination of H1 and H3 sources.
**Closest prior art:** No prior art on this specific combination. Standard practice in architecture search is to test promising modifications independently then stack.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.006 (3x seed variance, must beat both individual improvements)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.006 and -0.002
**REFUTED if:** delta > -0.002 or CRASH

**Predicted delta:** -0.008
**Predicted confidence:** low
**Case for failure:** Stacking two modifications increases the risk of unexpected interactions. The SwiGLU gating changes the gradient distribution through the MLP, which the predictor-corrector residual momentum depends on for its smoothness assumptions. The combined modifications may also push VRAM past 76 GB. Should only be run if H1 or H3 shows promise individually.
**Feasibility pre-flight:** Combined parameter overhead is small (SwiGLU matches params, predictor-corrector adds ~10 scalars). VRAM is dominated by activation memory which may increase slightly with SwiGLU's gate tensor + the momentum residual tensor. Estimate ~70 GB total.
**Implementation sketch:** Apply both H1 and H3 modifications to train.py simultaneously.

**Information gain analysis:**
- P(success): 0.20
- If CONFIRMED, I learn: Activation improvements and residual stream improvements are orthogonal and stack, opening the door for combinatorial architecture search.
- If REFUTED, I learn: There are hidden interactions between activation function choice and residual stream dynamics, suggesting these subsystems are more coupled than they appear.
- Expected information gain: medium -- most useful as a follow-up to individual confirmations, less informative as a standalone test.

**Note:** This hypothesis should be deprioritized in favor of H1 and H3 individually. Only run if both H1 and H3 are CONFIRMED or if engineering budget allows after testing the others.

**REJECTION REASON:** Combo hypothesis depends on individual results from H1 and H3 that have not yet been obtained. The contract itself acknowledges this: "Should only be run if H1 or H3 shows promise individually." Running a combo before components are validated wastes an engineering run and produces uninterpretable results (if it fails, you cannot attribute failure to H1, H3, or their interaction). Re-propose after H1 and H3 verdicts are known.

---

# Cycle 2 Hypotheses

---

## Hypothesis 6: SwiGLU Activation Follow-Up (wider hidden dim) | Status: CONFIRMED

### Sprint Contract

**Intervention:** Replace ReluSquared MLP activation with SwiGLU (gated SiLU), using hidden_dim=1792 (nearest multiple of 128 above 8/3 * 640 = 1706.7), accepting a modest ~3-5% parameter increase over baseline to avoid the severe width bottleneck that limited H1.
**Subsystem:** activation/gating, MLP/activation-fn
**Papers:** [Domain A] Shazeer 2020, "GLU Variants Improve Transformer" (arXiv:2002.05202) -- SwiGLU uses element-wise product of a sigmoid-gated linear projection and an ungated projection, providing learned feature selection x [Domain B] Carandini & Heeger 2012, "Normalization as a canonical neural computation" (Nature Reviews Neuroscience) -- multiplicative gating in cortical circuits implements gain control, a canonical computation that selectively amplifies or suppresses signals based on context. SwiGLU's gate is a learned analogue of this neural gain control.
**Closest prior art:** H1 tested SwiGLU at hidden_dim=1664 (exact 8/3 ratio) and got INCONCLUSIVE delta=-0.00246. The 35% width reduction from 2560 to 1664 was likely the binding constraint. This follow-up tests hidden_dim=1792 (+128 over H1), which is a 7.7% increase in hidden capacity. SwiGLU is standard in LLaMA/Gemma but has not been tested at this exact config with Muon.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003
**Predicted confidence:** medium
**Case for failure:** Even at hidden_dim=1792, the MLP hidden layer is still 30% narrower than the baseline ReluSquared hidden dim of 2560. The gating mechanism requires learning good gate values, which takes representational capacity away from direct computation. Muon's Newton-Schulz may interact differently with the three-projection SwiGLU structure compared to the two-projection ReluSquared MLP. Additionally, H1 already showed that the benefit of SwiGLU at this scale is marginal -- adding 128 hidden dimensions may not be enough to cross the threshold.
**Feasibility pre-flight:** 3 weight matrices per MLP layer (c_fc, c_gate, c_proj) with shapes (640, 1792), (640, 1792), (1792, 640). Total MLP params per layer: 3 * 640 * 1792 = 3,440,640 vs baseline 2 * 640 * 2560 = 3,276,800 (~5% increase). VRAM: H1 used 65.6 GB, this will be slightly higher due to wider hidden dim, estimate ~66-67 GB. torch.compile handles SiLU natively.
**Implementation sketch:** Replace MLP class: add c_gate = nn.Linear(n_embd, 1792, bias=False). Forward: gate = silu(c_gate(x)); up = c_fc(x); x = gate * up; x = c_proj(x). Update c_fc to (n_embd, 1792), c_proj to (1792, n_embd). Update init_weights for c_gate (uniform like c_fc). Muon will auto-group c_gate by shape.

**Information gain analysis:**
- P(success): 0.45
- If CONFIRMED, I learn: SwiGLU's gating quality exceeds ReluSquared's sparsity benefit at this scale, and the H1 shortfall was purely a width bottleneck. This validates width as the binding constraint for gated activations in small models.
- If REFUTED, I learn: SwiGLU's benefit at this scale is genuinely marginal even with adequate width. ReluSquared's extreme sparsity is a better inductive bias for Muon at small scale. This closes the SwiGLU avenue definitively.
- Expected information gain: high -- this is a clean isolation of the width variable from H1, and both outcomes are decision-relevant (CONFIRMED opens stacking with H4, REFUTED closes the activation avenue).

**Evaluator P(success):** 0.35
**Belief divergence:** 0.10

**Evaluator notes:** This is an authorised follow-up. The implementation is straightforward and well-specified. Key concern: H1 achieved delta=-0.00246 at hidden_dim=1664. The gap to the -0.003 threshold is 0.00054. Going from 1664 to 1792 is a 7.7% width increase, but the width-to-performance relationship is sublinear, so the expected gain is perhaps 0.0003-0.0005 -- right at the edge. The 5% parameter increase is acceptable. Implementation guidance: (1) round hidden_dim to 1792 (multiple of 128, confirmed); (2) init c_gate with same uniform distribution as c_fc; (3) c_gate will auto-group with c_fc in Muon since they share shape (640, 1792); (4) no param assertion change needed since c_gate is inside transformer.h. This is a well-designed follow-up but the margin for success is thin.

---

## Hypothesis 7: Leaky Integral Residual Accumulator | Status: INCONCLUSIVE

### Sprint Contract

**Intervention:** Add a leaky integral term to the residual stream, extending the confirmed PD mechanism (H4) toward a full PID controller. Each layer maintains a running leaky integral I_i = decay * I_{i-1} + block_output_delta, and adds integ_lambda * I_i to the residual before the next block. The "leaky" (exponential decay) prevents integral windup, a classic failure mode in PID control where the integral term grows without bound.
**Subsystem:** residuals/integral-scaling
**Papers:** [Domain A] "Predictor-Corrector Enhanced Transformers with EMA Coefficient Learning" (arXiv:2411.03042) -- shows that inter-layer coupling via exponential moving averages of block outputs can improve transformer performance. The EMA coefficient is analogous to our integral decay rate. x [Domain B] Classical PID control theory: Astrom & Murray, "Feedback Systems: An Introduction for Scientists and Engineers" (Princeton UP, 2008) -- the integral term in PID controllers eliminates steady-state error by accumulating past error signals. Anti-windup (leaky integration) prevents the integral from saturating when the system is constrained, maintaining responsiveness. In our context, the "steady-state error" analogue is the residual stream's drift from an ideal trajectory across layers.
**Closest prior art:** H4 (PD residual scaling) added a derivative term and was CONFIRMED at delta=-0.004. H3 (predictor-corrector momentum) added a retrospective blend of previous block outputs and was INCONCLUSIVE at delta=-0.0006. The integral term differs from both: the derivative looks at rate of change (velocity), momentum looks at the previous single delta, but the integral accumulates ALL prior deltas with exponential decay. This captures longer-range layer-wise structure that neither H3 nor H4 addressed.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**Case for failure:** The integral term accumulates information from ALL prior layers, which at depth=10 means it carries signal from up to 9 layers back. This accumulated signal may be stale and noisy. The leaky decay helps but may either be too aggressive (losing useful signal) or too mild (accumulating noise). Unlike the derivative term (H4) which looks at local layer-wise structure, the integral looks at global structure, and 10 layers may not have enough global structure to exploit. Additionally, the integral introduces a new pathway for gradient flow that could interact poorly with Muon. The integ_lambdas being learned may not converge to useful values in 1800 steps. H3's failure (retrospective momentum, delta=-0.0006) is a warning that backward-looking residual modifications are weaker than forward-looking ones (H4's derivative).
**Feasibility pre-flight:** Adds 10 scalar parameters (integ_lambdas, init=0) and 1 fixed decay constant (0.9, not learned -- learning it would add torch.compile complexity). VRAM: one extra tensor the size of the residual stream for the running integral. At 69.3 GB baseline (with H4-style changes), this adds maybe 0.2 GB, estimate ~69.5 GB. torch.compile: the leaky accumulation is just lerp + multiply, standard ops.
**Implementation sketch:** In GPT.__init__: self.integ_lambdas = nn.Parameter(torch.zeros(config.n_layer)). In forward: integral = torch.zeros_like(x). For each layer i: (1) save x_before = x; (2) x = block(x, ve, cos_sin, window_size); (3) delta = x - x_before; (4) integral = 0.9 * integral + delta; (5) x = x + integ_lambdas[i] * integral. Add integ_lambdas to scalar_lr optimizer group (same LR as deriv_lambdas: scalar_lr * 0.01). Update setup_optimizer assertion.

**Information gain analysis:**
- P(success): 0.20
- If CONFIRMED, I learn: The residual stream has exploitable long-range layer-wise structure beyond just local velocity. The accumulated error signal across layers contains useful information for correcting the trajectory. This would validate the full PID framework and suggest that a combined PD+I (or full PID) controller on the residual stream could be even better.
- If REFUTED, I learn: Layer-wise residual structure at depth=10 is purely local (velocity/derivative is useful, but cumulative history is not). The PD result (H4) was capturing local smoothness, not global trajectory correction. This constrains future residual hypotheses to local (1-2 layer lookback) mechanisms only.
- Expected information gain: high -- this directly tests whether H4's success was about local smoothness (derivative) or whether broader temporal structure exists. Both outcomes sharply update the model.

**Evaluator P(success):** 0.12
**Belief divergence:** 0.08

**Evaluator notes:** This hypothesis is well-aligned with the exploration directive (HIGH SURPRISAL on residuals/derivative-scaling -- explore further with control-theory residual modifications). However, I am more sceptical than the Researcher. The key concern is the analogy with H3: H3 was retrospective (blending previous block delta) and got only -0.0006. H7 is also retrospective (accumulating all past block deltas). The integral accumulates stale information from layers 0-8 when computing the correction at layer 9 -- and with only 10 layers and decay=0.9, the effective memory is about 7 layers, meaning the integral carries substantial signal from early layers whose outputs may be irrelevant to late-layer dynamics. H4 worked because it was anticipatory (velocity before the block). H7 adds its correction after computing delta, making it retrospective like H3. Nevertheless, the mechanism is genuinely different from H3 (full exponential history vs 1-step lookback), and the information gain is high: if this works, it validates the full PID framework; if it fails alongside H3's near-failure, it confirms that only anticipatory (derivative/velocity) mechanisms work in the residual stream. Implementation guidance: (1) integ_lambdas must be added to setup_optimizer; update the assertion on line 247 to include them; (2) use scalar_lr * 0.01 learning rate, same as deriv_lambdas; (3) the integral tensor is the same size as x, adding ~0.2 GB VRAM, well within budget; (4) delta must be computed as x_after_block - x_before_block in GPT.forward, same pattern as H4's velocity.

---

## Hypothesis 8: Per-Head Learnable Attention Temperature | Status: REFUTED

### Sprint Contract

**Intervention:** Add a learnable per-head temperature scalar that scales the QK dot products before softmax (equivalently, scales the post-norm Q vectors). Each attention head gets a scalar tau_h initialized at 1.0, so attention logits become (q * tau_h) @ k^T. This allows different heads to specialize in sharp (high tau) vs diffuse (low tau) attention patterns without changing the QK-norm or the softcap.
**Subsystem:** attention
**Papers:** [Domain A] Ye et al. 2024, "Differential Transformer" (arXiv:2410.05258) -- while differential attention itself failed 3 times, the paper demonstrates that per-head attention modulation improves language modeling by allowing heads to specialize. Our approach is simpler: scalar temperature instead of differential subtraction. x [Domain B] Statistical mechanics: the Boltzmann distribution P(state) = exp(-E/kT) / Z, where temperature T controls the sharpness of the distribution over energy states. In attention, the QK logits are "energies" and softmax is the Boltzmann distribution. Allowing per-head temperature is analogous to allowing different subsystems in a thermodynamic ensemble to equilibrate at different temperatures -- a concept from non-equilibrium statistical mechanics where heterogeneous temperature distributions arise naturally in driven systems.
**Closest prior art:** Standard transformers use a fixed temperature of 1/sqrt(d_k). QK-norm (used in our baseline) replaces this with unit-norm Q and K, making the effective temperature controlled by the softcap. Differential attention (failed 3x in v3-v5) tried per-head modulation via subtraction of two attention patterns. Our approach is fundamentally different: multiplicative scaling (temperature) rather than additive subtraction (differential). The scaling preserves the attention pattern's shape while adjusting its sharpness, whereas differential attention creates entirely new patterns. Per-head temperature has been used in some KV-cache compression papers but not as a learned architectural parameter in standard transformer training.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**Case for failure:** With only 5 heads, there are only 5 temperature scalars to learn -- the model may not have enough heads for meaningful specialization. The softcap at 12 already constrains the attention logit range, so temperature scaling may have limited dynamic range. QK-norm already normalizes Q and K to unit norm, so the temperature scalar is effectively competing with (or redundant to) the existing projection weights. Muon optimizes the QK projection matrices and may already implicitly tune the attention sharpness through the weight magnitudes. The scalar may also interact with torch.compile -- however, since it's a simple multiply on Q before the matmul, it should be fine. Differential attention's 3x failure is a cautionary note about per-head modulation, though temperature is a much lighter intervention.
**Feasibility pre-flight:** Adds n_head * n_layer = 5 * 10 = 50 scalar parameters. These are tiny and add negligible VRAM. The multiply happens on Q after norm, before the attention matmul. torch.compile: multiplying Q by a scalar per head is a standard broadcast operation, should not break the graph. Must be careful to implement as q = q * tau[None, None, :, None] (broadcast over B, T, head, dim). Add tau params to scalar_lr optimizer group.
**Implementation sketch:** In CausalSelfAttention.__init__: self.head_temps = nn.Parameter(torch.ones(self.n_head)). In forward, after QK-norm: q = q * self.head_temps[None, None, :, None]. That's it. In GPT.setup_optimizer: collect head_temps from all layers into a new scalar_lr group. Update the parameter count assertion. Init at 1.0 means identity at start (safe initialization, follows v6 winning pattern).

**Information gain analysis:**
- P(success): 0.25
- If CONFIRMED, I learn: Attention heads benefit from heterogeneous sharpness at this scale. The 5 heads are currently forced to operate at the same effective temperature, and allowing specialization improves modeling capacity. This opens further per-head parameterization (per-head value scaling, per-head window sizes).
- If REFUTED, I learn: With only 5 heads and QK-norm + softcap already constraining the logit range, per-head temperature provides no additional benefit. The existing projection weights already encode optimal temperature implicitly. This would suggest that attention-level modifications at this scale require changing the structure (more heads, different patterns) rather than adding per-head knobs.
- Expected information gain: high -- this is the first test of the attention subsystem (currently OPEN with zero tests). Either outcome provides a strong signal about whether attention-level modifications are a productive direction at all for this architecture. High information value from an untested subsystem.

**Evaluator P(success):** 0.20
**Belief divergence:** 0.05

**Evaluator notes:** Low belief divergence -- both agents agree this is a long shot. The attention subsystem is completely untested (OPEN, 0 tests), giving this hypothesis high information value regardless of outcome. The implementation is minimal (50 scalars, one multiply on Q). CRITICAL implementation concern: head_temps parameters live inside CausalSelfAttention (inside transformer.h). The current setup_optimizer uses `matrix_params = list(self.transformer.h.parameters())` and groups them by shape for Muon. head_temps has shape (5,) which is 1D. The Muon grouping loop would create a group for shape (5,) and muon_step_fused would crash attempting Newton-Schulz on a 1D tensor (X.mT @ X requires 2D). FIX REQUIRED: either (a) move head_temps to GPT level (like resid_lambdas) and pass to blocks, or (b) filter 1D params out of matrix_params in setup_optimizer and add them to the scalar_lr AdamW group. Option (a) is simpler and matches the pattern used by resid_lambdas. The Researcher must implement this correctly or the run will crash. Softcap at 15 (line 284 of train.py, not 12 as the contract states -- verify) may further limit the temperature's effective range. With QK-norm producing unit-norm Q and K, the dot product range is [-1, 1] per dimension, and scaling by tau just scales this range. The softcap then clips at 15, so temperature has limited dynamic range to exploit.

---

## Hypothesis 9: Block Output Gating (Attention + MLP) | Status: REJECTED

### Sprint Contract

**Intervention:** Add a learnable scalar gate to each sub-layer output (attention and MLP separately) that scales the sub-layer's contribution before adding it to the residual stream. Each block computes: x = x + alpha_attn * attn(norm(x)) + alpha_mlp is NOT how it works -- rather, the two residual additions in Block.forward become: x = x + alpha_attn_i * self.attn(norm(x), ...); x = x + alpha_mlp_i * self.mlp(norm(x)). The alphas are per-layer scalars initialized at 1.0 (identity, following the v6 winning pattern).
**Subsystem:** attention, residuals/sub-layer-scaling
**Papers:** [Domain A] Wang et al. 2022, "DeepNet: Scaling Transformers to 1,000 Layers" (arXiv:2203.00555) -- introduces per-sublayer alpha scaling for deep transformers, showing that appropriate scaling enables stable training at extreme depth. Our version makes the alphas learnable rather than fixed, allowing the model to discover optimal sub-layer contribution ratios. x [Domain B] Telecommunications: automatic gain control (AGC) in radio receivers -- AGC dynamically adjusts the gain of amplifier stages to maintain optimal signal level despite varying input strength. Each amplifier stage has an independent gain control, analogous to our per-sublayer alpha. The key insight from AGC is that heterogeneous gain across processing stages is essential for optimal signal-to-noise ratio in cascaded systems.
**Closest prior art:** DeepNet uses fixed alpha = (2 * n_layer)^{-0.25} for all sub-layers. Our alphas are learnable and per-sublayer (separate for attention and MLP). The existing resid_lambdas scale the entire residual before the block, but do not distinguish between attention and MLP contributions. This hypothesis tests whether attention and MLP sub-layers should have different contribution weights, which resid_lambdas cannot express. FixUp and similar zero-init schemes are related but target initialization rather than learned scaling.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**Case for failure:** The existing resid_lambdas already provide per-layer scaling of the residual stream, and the projection layers (c_proj in attention and MLP) already have learned output magnitudes. Adding another multiplicative scalar on top may be redundant -- the model can achieve the same effect by scaling the c_proj weights. With init=1.0, the alphas may simply stay near 1.0 throughout training if the baseline scaling is already near-optimal. Additionally, 20 new scalar parameters (2 per layer) in the optimizer may interact with Muon's gradient processing in unexpected ways. The scalar_lr group uses AdamW, not Muon, so this should be safe, but the gradients flowing through the alphas could affect the gradient magnitudes seen by the Muon-optimized projection weights.
**Feasibility pre-flight:** Adds 2 * n_layer = 20 scalar parameters. Negligible VRAM. The modification is inside Block.forward, changing `x = x + self.attn(...)` to `x = x + alpha_attn * self.attn(...)`. torch.compile: scalar multiply is trivial, no graph break. Must modify Block to accept the alpha params (pass from GPT.forward or store in Block). Add alpha params to scalar_lr optimizer group. Update assertion.
**Implementation sketch:** In GPT.__init__: self.alpha_attn = nn.Parameter(torch.ones(config.n_layer)); self.alpha_mlp = nn.Parameter(torch.ones(config.n_layer)). In Block.forward, accept alpha_attn and alpha_mlp as arguments: x = x + alpha_attn * self.attn(norm(x), ve, cos_sin, window_size); x = x + alpha_mlp * self.mlp(norm(x)). In GPT.forward loop: pass self.alpha_attn[i] and self.alpha_mlp[i] to block. In setup_optimizer: add [self.alpha_attn, self.alpha_mlp] to scalar_lr group (same LR as resid_lambdas: scalar_lr * 0.01 to be conservative). Update assertion.

**Information gain analysis:**
- P(success): 0.25
- If CONFIRMED, I learn: Attention and MLP sub-layers benefit from different contribution weights, meaning the current architecture's implicit equal weighting is suboptimal. This reveals that the model wants to up-weight or down-weight specific processing stages, which could guide further architectural changes (e.g., if MLP alphas converge high and attention alphas converge low, we might benefit from wider MLPs and narrower attention).
- If REFUTED, I learn: The existing resid_lambdas and c_proj weight magnitudes already encode optimal sub-layer scaling. Adding another degree of freedom at the sub-layer level is redundant. This constrains future residual modifications to mechanisms that cannot be absorbed by weight scaling (like the derivative term in H4).
- Expected information gain: medium-high -- tests whether sub-layer heterogeneity matters, which is relevant for both the attention and residuals subsystems. Partial overlap with the resid_lambdas mechanism limits novelty slightly.

**Evaluator P(success):** 0.10
**Belief divergence:** 0.15

**REJECTION REASON:** Redundancy with existing mechanisms. The intervention adds alpha_attn * attn_output and alpha_mlp * mlp_output, but the model already has three mechanisms that subsume this: (1) resid_lambdas scale the residual before the block; (2) c_proj weights in both attention and MLP control output magnitude; (3) the init_weights already zeros c_proj (line 163-165 of train.py), meaning output magnitude is entirely learned through c_proj. A scalar alpha multiplying the output of a projection layer with zero-init is mathematically equivalent to scaling the c_proj weight matrix by alpha -- the model can absorb the alpha into c_proj at any point during training. This makes the alpha a redundant degree of freedom that Muon (operating on c_proj) can already express. The contract itself acknowledges this: "the model can achieve the same effect by scaling the c_proj weights." Unlike H4's derivative term (which captures inter-layer velocity, a quantity c_proj cannot express), sub-layer scaling is absorbable into existing weights. The expected information gain is low because the most likely outcome (alphas stay near 1.0, trivial interaction with c_proj) tells us nothing new. Re-propose only if combined with a mechanism that breaks the alpha-c_proj equivalence (e.g., input-dependent gating rather than fixed scalars).

---

# Cycle 3 Hypotheses

---

## Hypothesis 10: SwiGLU + PD Residual Stacking (H6+H4) | Status: INCONCLUSIVE

### Sprint Contract

**Intervention:** Combine the two confirmed improvements: (1) SwiGLU activation with hidden_dim=1792 (H6, delta=-0.005) and (2) PD residual derivative scaling (H4, delta=-0.004) in a single training run. Both modifications are applied simultaneously to the baseline architecture.
**Subsystem:** activation/gating + residuals/derivative-scaling (stacking)
**Papers:** [Domain A] Shazeer 2020, "GLU Variants Improve Transformer" (arXiv:2002.05202) x [Domain B] Classical PID control theory -- PD controller derivative term for residual stream. These are the same paper sources as H6 and H4; this is a stacking experiment, not a novel hypothesis.
**Closest prior art:** H6 (SwiGLU, delta=-0.005) and H4 (PD residual, delta=-0.004) individually confirmed. H5 (proposed combo of H1+H3) was rejected because components were unvalidated. Now both components are confirmed and explicitly authorised for stacking by the Evaluator.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.006 (must exceed best individual result by at least 0.001)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.006 and -0.003
**REFUTED if:** delta > -0.003 or CRASH

**Predicted delta:** -0.007
**Predicted confidence:** medium
**P(regression):** 0.05 (very low -- both components are individually confirmed and modify orthogonal subsystems)
**Case for failure:** Even though the subsystems are orthogonal (MLP activation vs residual stream), there could be a subtle interaction: SwiGLU changes the gradient magnitude profile flowing through the MLP, which affects the gradient signal that the deriv_lambdas receive for learning the velocity coefficients. If SwiGLU's smoother gradients (vs ReluSquared's sparse gradients) change the optimal deriv_lambda learning rate, the stacked result could be subadditive. The VRAM budget is tight: H6 used 67.5 GB, H4 used 69.3 GB. The stacked version needs both the wider SwiGLU matrices (3 x 1792 projections) and the derivative velocity tensor. Estimated ~69-70 GB, leaving ~6 GB headroom for torch.compile overhead.
**Feasibility pre-flight:** SwiGLU replaces MLP class (3 projections, hidden_dim=1792). PD adds deriv_lambdas (10 scalars) and velocity computation in forward loop. Total parameter increase ~5% over baseline (dominated by SwiGLU). VRAM ~69-70 GB. torch.compile: both components individually compile; no new interaction expected.
**Implementation sketch:** Apply both H6 and H4 modifications simultaneously: (1) Replace MLP with SwiGLU (c_fc, c_gate, c_proj at hidden_dim=1792); (2) Add deriv_lambdas = nn.Parameter(torch.zeros(n_layer)); (3) In forward loop: compute velocity = x - x_prev before each block, apply deriv_lambda * velocity; (4) Update setup_optimizer with deriv_lambdas in scalar_lr * 0.01 group. The implementation is the union of the two confirmed implementations.

**Information gain analysis:**
- P(success): 0.60
- If CONFIRMED, I learn: Orthogonal architectural improvements stack at this scale. This validates the "subsystem independence" assumption and opens combinatorial search across confirmed mechanisms. The magnitude of stacking (additive vs subadditive) reveals the degree of interaction between activation function choice and residual stream dynamics.
- If REFUTED, I learn: There are hidden interactions between activation function and residual stream modifications that prevent stacking. This is highly informative because it would challenge the assumption that modifications to different subsystems are independent, and would require a mechanistic explanation (gradient flow coupling, capacity competition, etc.).
- Expected information gain: high -- the stacking assumption is load-bearing for the entire research programme. Either outcome dramatically updates the strategy.

**Rollback plan:** If regression or crash, separate into two diagnostic runs: (a) re-run H6 alone to verify it still works on current codebase, (b) re-run H4 alone. If both still work individually, the interaction is the cause and this subsystem combination is blocked.

**Evaluator P(success):** 0.55
**Belief divergence:** 0.05

**Evaluator notes:** Explicitly authorised stacking experiment. Both components are confirmed and modify orthogonal subsystems (MLP activation vs residual stream dynamics). Implementation is the union of two known-good implementations. VRAM estimate of 69-70 GB is credible (SwiGLU at 1792 used 67.5 GB; PD adds one residual-sized tensor). The success threshold of delta < -0.006 is appropriate -- it requires the stack to beat the best individual result by 0.001, which is a fair test of genuine stacking. Low belief divergence: both agents agree this is likely to work. The main risk is subadditive stacking (both modifications succeed individually but compete for the same gradient signal at the residual stream interface). Note to implementer: the current train.py still has H8's head_temps code baked in -- make sure deriv_lambdas and SwiGLU are added cleanly on top of the current codebase without removing head_temps (which is used in the forward loop).

---

## Hypothesis 11: Peri-LN Output Normalization on Sublayer Outputs | Status: INCONCLUSIVE

### Sprint Contract

**Intervention:** Add an RMSNorm after each sublayer output (attention and MLP) before adding to the residual stream. The current architecture uses Pre-LN only (norm before attention and before MLP). Peri-LN adds a second normalization on the sublayer output: x = x + norm(self.attn(norm(x), ...)) and x = x + norm(self.mlp(norm(x))). This bounds the variance of sublayer contributions to the residual stream, preventing activation spikes that Pre-LN allows.
**Subsystem:** normalisation
**Papers:** [Domain A] Kim et al. 2025, "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture" (arXiv:2502.02732, ICML 2025) -- shows Peri-LN (norm on both input and output of sublayers) achieves 2.7-2.8% lower eval loss than Pre-LN at 1.5B-3.2B scale, with more balanced variance growth and steadier gradient flow. Adopted silently by Gemma and OLMo model families. x [Domain B] Friis formula for cascaded noise figure in telecommunications receiver chains -- F_total = F1 + (F2-1)/G1 + (F3-1)/(G1*G2) + ..., showing that in cascaded amplifier stages, the noise contribution of each stage is divided by the cumulative gain of all preceding stages. The first stage dominates total noise. Analogously, in a transformer, early layers' variance spikes propagate through all subsequent layers. Peri-LN's output normalization is analogous to placing a limiter/AGC after each amplifier stage in a receiver chain, preventing any single stage from injecting excessive variance that dominates the cascade. The Friis formula predicts that normalizing early-layer outputs has disproportionate impact -- exactly what Peri-LN achieves.
**Closest prior art:** Peri-LN is used in Gemma 2 and OLMo 2, but has not been tested at our scale (dim=640, depth=10) with Muon optimizer. The paper evaluates at 1.5B+ scale; small-scale behavior is unknown. Our architecture already uses RMSNorm (the `norm()` function calls F.rms_norm). Adding output normalization is zero-parameter (RMSNorm has no learnable affine parameters in our implementation).

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**P(regression):** 0.15 (the output norm changes gradient flow through the sublayer, which could interact with Muon's Newton-Schulz; also, bounding sublayer outputs may prevent the model from learning necessary magnitude variations across layers)
**Case for failure:** At depth=10, the variance accumulation problem that Peri-LN addresses may not be severe enough to matter. Peri-LN's benefits are demonstrated at 1.5B+ scale with 24+ layers, where exponential variance growth in Pre-LN becomes problematic. At 10 layers, Pre-LN's variance growth may be mild enough that the output norm adds overhead without benefit. Additionally, the output norm changes the gradient magnitude flowing back through c_proj, which Muon optimizes. Muon's Newton-Schulz orthogonalization assumes a particular gradient geometry; normalizing the forward pass output could distort this geometry. The output norm also eliminates magnitude information from sublayer outputs, which the model may use to signal confidence or salience.
**Feasibility pre-flight:** Zero new parameters (RMSNorm is parameterless in our implementation). Negligible VRAM increase (norm is in-place on existing tensors). torch.compile: F.rms_norm is already compiled in the forward pass; adding two more calls should not break the graph. Wall-clock: two extra norm calls per block per forward pass; at 10 layers this adds 20 norm calls. Each is very fast (dim=640). Estimate <5% wall-clock overhead.
**Implementation sketch:** In Block.forward, change: `x = x + self.attn(norm(x), ve, cos_sin, window_size, head_temps=head_temps)` to `x = x + norm(self.attn(norm(x), ve, cos_sin, window_size, head_temps=head_temps))`. Same for MLP: `x = x + norm(self.mlp(norm(x)))`. No optimizer changes needed. No new parameters.

**Information gain analysis:**
- P(success): 0.25
- If CONFIRMED, I learn: Even at shallow depth (10 layers), variance regulation on sublayer outputs improves training. This suggests the residual stream variance is suboptimally distributed even in shallow networks, and the Friis cascade analogy holds -- early-layer variance spikes propagate and degrade later processing.
- If REFUTED, I learn: At depth=10, Pre-LN's variance growth is not problematic, and the benefits of Peri-LN require deeper networks to manifest. This confirms that normalisation is not a bottleneck at this scale.
- Expected information gain: medium -- the normalisation subsystem is completely untested, so any result provides new information. However, P(success) is low enough that failure is the expected outcome, making it somewhat less informative.

**Rollback plan:** Remove the two output norm() calls in Block.forward. Zero-parameter intervention, trivial to revert.

**Evaluator P(success):** 0.20
**Belief divergence:** 0.05

**Evaluator notes:** Clean, zero-parameter intervention targeting an untested subsystem (normalisation). Critically, this normalizes sublayer outputs (dense activations flowing into the residual stream), NOT sparse post-activation hidden states -- so it does NOT violate the BLOCKED activation/normalization directive. The Friis cascade analogy is apt. My main concern is the same as the contract's: at depth=10, variance accumulation may simply not be severe enough for Peri-LN to matter. The paper's results are at 1.5B+ scale with 24+ layers. At 10 layers, the multiplicative variance growth may be too small to create problems. Also, the output norm changes gradient magnitudes flowing through c_proj, which could interact with Muon. Low belief divergence: both agents agree this is a long shot. Implementation is trivial (two extra norm() calls in Block.forward). Note to implementer: use the existing norm() function (which calls F.rms_norm), not a new normalization. Verify that the norm is applied to the sublayer output BEFORE adding to the residual, not to x after the residual addition.

---

## Hypothesis 12: Latest Weight Averaging (LAWA) in Training Loop | Status: REFUTED

### Sprint Contract

**Intervention:** Implement Latest Weight Averaging (LAWA) during training: maintain an exponential moving average (EMA) of model weights alongside the training weights, and use the EMA weights for evaluation. Specifically, after each optimizer step, update EMA weights: ema_param = beta * ema_param + (1 - beta) * param, with beta=0.999. At the end of training, evaluate with EMA weights instead of the last-iterate weights. This is a training-loop-only modification: the model architecture is unchanged.
**Subsystem:** training-loop
**Papers:** [Domain A] Sanyal et al. 2024, "Early Weight Averaging meets High Learning Rates for LLM Pre-training" (arXiv:2306.03241, COLM 2024) -- shows that weight averaging (EMA/LAWA) improves GPT-2 small validation loss from 2.963 to 2.917 (LAWA) on OpenWebText with 70K steps. The benefit is larger with high learning rates and short training schedules, both of which apply to our setting (1800 steps). x [Domain B] Information theory: bias-variance decomposition and the James-Stein estimator. The James-Stein estimator (Stein 1956) shows that shrinking parameter estimates toward a target (the trajectory mean) reduces expected squared error in dimensions >= 3. EMA weight averaging is a form of temporal shrinkage: the EMA estimate is biased toward earlier parameters but has lower variance than the last iterate. In our 1800-step regime with high learning rates (matrix_lr=0.03 with Muon), the last iterate has high variance; EMA averaging exploits the bias-variance tradeoff by accepting mild bias for substantial variance reduction.
**Closest prior art:** EMA is standard in image generation (diffusion models use EMA universally) but uncommon in short-schedule LLM pretraining. The COLM 2024 paper demonstrates it works for GPT-2 pretraining. Our setting differs: we use Muon optimizer (not Adam) and train for only 1800 steps. The interaction of EMA with Muon's Newton-Schulz weight updates is untested. Note: EMA modifies only the evaluation procedure, not the training dynamics -- the training weights and gradients are unchanged.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**P(regression):** 0.05 (EMA cannot regress -- the training run is identical to baseline; only the evaluation uses EMA weights. Worst case, EMA weights are slightly worse than last-iterate, but since EMA averaging reduces noise, this is unlikely. The 0.05 accounts for edge cases where EMA with wrong beta produces stale weights.)
**Case for failure:** With only 1800 steps, the learning rate schedule already includes a 70% warmdown phase (steps 540-1800). During warmdown, the learning rate decays to 0.01x, which already provides implicit averaging by making late updates very small. EMA on top of warmdown may be redundant -- the last iterate is already "averaged" by the slow final updates. Additionally, Muon's Newton-Schulz updates produce weight matrices on a specific manifold (near-orthogonal); EMA of near-orthogonal matrices is not itself near-orthogonal, so the EMA weights may be off-manifold and suboptimal. The beta=0.999 means the EMA window is ~1000 steps, which covers more than half the training run -- this may be too wide, incorporating weights from the early training phase that are far from converged.
**Feasibility pre-flight:** EMA doubles the parameter storage (shadow copy of all parameters), adding ~50M params * 2 bytes (bf16) = ~100 MB. At baseline ~68 GB, this is negligible. Wall-clock: one extra parameter copy per step (lerp operation), ~1-2% overhead. torch.compile: EMA update happens outside the compiled forward/backward; no graph changes. Implementation is in the training loop only. No model architecture changes.
**Implementation sketch:** (1) After model initialization, create EMA shadow params: `ema_params = {name: p.clone() for name, p in model.named_parameters()}`. (2) After each optimizer.step(): `for name, p in model.named_parameters(): ema_params[name].lerp_(p, 1 - beta)` where beta=0.999. (3) Before final evaluation: swap model params with EMA params. (4) Run validation. (5) Report EMA val_bpb. Implementation note: the lerp_ must handle both bf16 and fp32 params correctly. The EMA update should NOT be inside torch.compile.

**Information gain analysis:**
- P(success): 0.30
- If CONFIRMED, I learn: The last-iterate weights from our training run are suboptimal due to noise, and temporal averaging improves generalization even at 1800 steps with warmdown. This opens further training-loop optimizations (different EMA schedules, multiple EMA rates, EMA-only during warmdown).
- If REFUTED, I learn: The warmdown schedule already provides sufficient implicit averaging, making explicit EMA redundant. This confirms the training loop is well-optimized and further gains must come from architecture, not training procedure.
- Expected information gain: medium-high -- the training-loop subsystem is completely untested, and EMA is low-risk (cannot regress), so failure is informative without being costly. The interaction between EMA and Muon is genuinely unknown and both outcomes update beliefs.

**Rollback plan:** Simply evaluate with the original (non-EMA) weights. The training run itself is unchanged.

**Evaluator P(success):** 0.15
**Belief divergence:** 0.15

**Evaluator notes:** Novel subsystem (training-loop, completely untested). The claim that "EMA cannot regress" is overstated -- EMA weights could be worse than last-iterate if beta is poorly chosen, producing stale averaged weights. However, the training dynamics are genuinely unchanged, so the risk of a catastrophic failure is near zero. My main scepticism is about redundancy with warmdown: the existing schedule decays LR to 0.01x over 70% of training, which already makes late updates small and produces implicit averaging. EMA on top of warmdown is likely redundant. Additionally, beta=0.999 means the EMA window is ~1000 steps, covering more than half the 1800-step run. This incorporates weights from early training that are far from converged, which is counterproductive. A better beta might be 0.995 or 0.99 (window of 200-100 steps, covering just the warmdown phase). Belief divergence is moderate (0.15): the Researcher is more optimistic than I am. This makes it a useful experiment -- if it works despite my scepticism, it reveals that EMA adds value beyond warmdown, which would be genuinely surprising. Note to implementer: (1) the EMA update must be outside torch.compile; (2) use torch.lerp_ for the update; (3) ensure both bf16 and fp32 params are handled correctly; (4) consider logging both last-iterate and EMA val_bpb so we can see the delta attributable to averaging vs architecture.

---

## Hypothesis 13: QK-Norm L2 to Linf Relaxation for Attention Diversity | Status: REFUTED

### Sprint Contract

**Intervention:** Replace the L2 QK-norm (which normalizes Q and K to unit L2 norm per head) with a softer normalization: divide Q and K by their Linf norm (max absolute value) plus a small epsilon. This relaxes the constraint from "all Q/K vectors live on a hypersphere" to "all Q/K vectors live in a hypercube [-1, 1]^d". This preserves directional information while bounding magnitude, allowing different dimensions to have different scales. The attention logits then depend on both direction AND relative magnitude of Q/K dimensions, giving richer attention patterns.
**Subsystem:** attention/normalization-type (not attention/temperature -- this is structural, not scalar)
**Papers:** [Domain A] Karagodin et al. 2025, "Normalization in Attention Dynamics" (MIT, 2025) -- analyzes how different normalization choices in attention affect training dynamics and convergence, showing that the choice of norm function materially affects what attention patterns can be learned. x [Domain B] Compressed sensing and signal processing: Lp norm relaxation. In compressed sensing (Candes & Tao 2005, Donoho 2006), the L1 norm is used as a convex relaxation of L0 for sparse recovery. More generally, the choice of Lp norm determines the geometry of the constraint set and thus which solutions are favored. L2 favors isotropic (uniform magnitude across dimensions) solutions; Linf favors solutions where all dimensions are bounded but can vary freely. In attention, L2 norm forces Q/K to distribute their magnitude uniformly across the 128 head dimensions. Linf allows concentration of magnitude in specific dimensions, enabling sharper feature-selective attention.
**Closest prior art:** Standard QK-norm uses L2 (our baseline). The recent paper "Enhanced QKNorm normalization for neural transformers with the Lp norm" (arXiv:2602.05006) explores Lp norms for QK normalization, confirming that L2 is not necessarily optimal. However, their experiments are on different architectures and scales. Linf normalization for QK has not been tested in our configuration. This is fundamentally different from H8 (per-head temperature): H8 added a scalar multiplier on Q that was absorbable/redundant with QK-norm. This hypothesis changes the norm function itself, altering the geometry of the attention computation.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.001
**Predicted confidence:** low
**P(regression):** 0.30 (HIGH -- changing the norm function alters the entire attention logit distribution; Linf norm produces values in [-1, 1] per dimension rather than on a unit sphere, which changes the effective scale of dot products. The softcap at 15 was tuned for L2-normed Q/K where dot products range roughly [-1, 1] per head. With Linf norm, dot products could be much larger (up to 128 for head_dim=128), potentially saturating the softcap. This requires careful analysis.)
**Case for failure:** The softcap at 15 was designed for L2-normed Q and K, where the dot product per head is bounded by [-1, 1] (since both Q and K have unit L2 norm, and the dot product of unit vectors is in [-1, 1]). With Linf normalization, Q and K have max absolute value 1 per dimension, but the dot product can range up to [-head_dim, head_dim] = [-128, 128]. This would instantly saturate the softcap, producing uniform attention patterns. CRITICAL: The Linf norm must be accompanied by a scaling factor of 1/sqrt(head_dim) on the dot product (or equivalently, divide Q by sqrt(head_dim) after Linf norm) to maintain similar logit scales. Even with this fix, the attention pattern geometry changes fundamentally, and Muon's gradient updates through the QK projections may not adapt correctly within 1800 steps. The attention subsystem is flagged HIGH RISK after H8's regression.
**Feasibility pre-flight:** Zero new parameters. The change is in the `norm()` calls for Q and K only (line 93 of train.py): replace `q, k = norm(q), norm(k)` with Linf normalization. Must NOT change the global norm() function (used for residual stream normalization). Implement as a separate function: `def linf_norm(x): return x / (x.abs().amax(dim=-1, keepdim=True) + 1e-6)`. Apply a 1/sqrt(head_dim) scaling to maintain logit scale. VRAM: identical. torch.compile: amax is a standard op, should compile.
**Implementation sketch:** (1) Add function: `def linf_norm(x): return x / (x.abs().amax(dim=-1, keepdim=True) + 1e-6)`. (2) In CausalSelfAttention.forward, replace `q, k = norm(q), norm(k)` with `q, k = linf_norm(q) * (self.head_dim ** -0.5), linf_norm(k)`. The 1/sqrt(128) = 0.0884 scaling on Q brings dot product magnitudes back to [-1, 1] range, compatible with softcap=15. (3) No optimizer changes.

**Information gain analysis:**
- P(success): 0.15
- If CONFIRMED, I learn: L2 QK-norm is suboptimal and the attention mechanism benefits from non-isotropic normalization. This reveals that different head dimensions carry different amounts of useful information for attention computation, and forcing uniform magnitude across dimensions (L2) wastes capacity. Opens exploration of other Lp norms (L1, L4, etc.).
- If REFUTED, I learn: L2 QK-norm is well-suited to this architecture, and the attention logit geometry does not benefit from relaxation. Combined with H8's failure, this would suggest the attention subsystem is tightly optimized and further modifications require structural changes (more heads, different head dimensions) rather than normalization tweaks.
- Expected information gain: high -- the attention subsystem is flagged as HIGH SURPRISAL by the Evaluator. This hypothesis tests a fundamentally different aspect of attention (norm geometry) than H8 (scalar temperature). Regardless of outcome, it disambiguates whether the attention subsystem is globally rigid (both norm changes and scalar changes fail) or whether H8 failed for a narrow reason (scalar redundancy with existing constraints).

**Rollback plan:** Revert the two lines in CausalSelfAttention.forward back to `q, k = norm(q), norm(k)`.

**Evaluator P(success):** 0.10
**Belief divergence:** 0.05

**Evaluator notes:** This is a HIGH RISK experiment on an already-flagged subsystem. The attention subsystem produced a +0.005 regression with H8, and this hypothesis makes a more fundamental change (replacing the norm function entirely). However, it is genuinely mechanistically different from H8: H8 added a redundant scalar temperature on top of the existing L2 norm; H13 changes the geometric constraint itself. The information gain is high regardless of outcome. If it fails, combined with H8's failure, it strongly suggests the attention subsystem is globally rigid at this scale. If it succeeds, it reveals that L2 QK-norm was a bottleneck. My concerns: (1) flash_attn_func (fa3) may have internal assumptions about Q/K scale or norm. Verify the fa3 API does not internally normalize or assume unit-norm inputs. (2) The 1/sqrt(head_dim) scaling on Q shifts the dot product range to approximately [-11.3, 11.3], which is within the softcap=15 range but uses most of it. This is a tighter operating range than L2-normed attention. (3) The Linf norm's gradient is sparse (gradient only flows through the argmax dimension), which could cause training instability. (4) P(regression)=0.30 is the highest in this batch. Despite these concerns, the experiment is worth running because the attention subsystem is the highest-surprisal subsystem in the programme and needs disambiguation. Note to implementer: define linf_norm as a SEPARATE function, do NOT modify the global norm() function. Apply the 1/sqrt(head_dim) scaling as specified.

---

# Cycle 5 Hypotheses (Sprint Contracts)

## Hypothesis 15: Triple Stack H6+H4+H14 (SwiGLU + PD Residual + Factored Embeddings) | Status: INCONCLUSIVE

### Sprint Contract

**Intervention:** Combine all three confirmed improvements into a single configuration: (1) Replace ReluSquared MLP with SwiGLU activation using hidden_dim=1792 (H6), (2) Add PD-controller derivative term to residual stream with per-layer deriv_lambdas init=0, lr=scalar_lr*0.01 (H4), (3) Factored embeddings with 256-dim bottleneck and Muon-optimized projection (H14). This also removes the Peri-LN output normalization (H11, INCONCLUSIVE) from the current codebase. The baseline for comparison is the original val_bpb=0.959340.
**Subsystem:** stacking
**Papers:** [Domain A] Shazeer 2020, "GLU Variants Improve Transformer" (arXiv:2002.05202) combined with Lan et al. 2020, "ALBERT" (arXiv:1909.11942) x [Domain B] Control theory: PD controllers (Astrom & Murray 2008, "Feedback Systems") -- superposition principle from linear systems theory predicts that independent modifications to orthogonal subsystems should combine additively; the 37% discount observed in H10 (H6+H4 stacking) quantifies the nonlinear interaction. Adding H14 (a regularizing intervention that reduces parameters) may reduce or increase the discount rate.
**Closest prior art:** H10 stacked H6+H4 and achieved delta=-0.005692 (INCONCLUSIVE, 37% subadditive). This adds H14 to the stack. The question is whether the triple-stack discount exceeds, matches, or falls below the 37% pairwise discount. The additive sum of all three would be -0.012 (H6:-0.005 + H4:-0.004 + H14:-0.003). At 37% discount, the expected triple-stack delta is ~0.012 * 0.63 = -0.0076. At 50% discount (if interactions worsen with more components), delta would be ~-0.006.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.006 from original baseline (i.e., val_bpb < 0.953340)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.006 and -0.003
**REFUTED if:** delta > -0.003 or CRASH

**Predicted delta:** -0.006
**Predicted confidence:** medium
**P(regression):** 0.05 (all three components individually confirmed; regression would require destructive interaction)
**Case for failure:** The 37% stacking discount from H10 may worsen with a third component. The three interventions meet at the residual stream: SwiGLU changes gradient magnitudes through the MLP, the PD derivative term adds a velocity correction to the residual input, and factored embeddings change the initial representation quality. These gradient-level interactions could compound, pushing the discount beyond 50%. Additionally, SwiGLU at hidden_dim=1792 changes the MLP parameter shapes, which affects the Muon grouping -- the PD deriv_lambdas learn from gradients flowing through these new shapes, and their optimal values may differ. VRAM is a concern: H10 used 72.3 GB. Adding factored embeddings saves ~38 MB (negligible), but the SwiGLU + PD + factored embedding combination has never been tested together and torch.compile may produce a larger graph.
**Feasibility pre-flight:** Expected VRAM ~72-73 GB (H10 was 72.3 GB; factored embeddings add negligible memory). SwiGLU hidden_dim=1792 requires 3 weight matrices per MLP layer (c_fc, c_gate, c_proj), each optimized by Muon. PD derivative adds 10 scalar deriv_lambdas (negligible). Factored embeddings: wte is (vocab_size x 256) with wte_proj (256 x 640) optimized by Muon. torch.compile: all three components have been individually compiled successfully. The combination should compile without new graph breaks.
**Implementation sketch:** Starting from baseline train.py: (1) Replace MLP with SwiGLU: add c_gate linear (n_embd -> 1792), change c_fc to (n_embd -> 1792), c_proj to (1792 -> n_embd). Forward: gate = silu(c_gate(x)); up = c_fc(x); x = gate * up; x = c_proj(x). (2) Add PD derivative: in GPT.__init__ add deriv_lambdas = nn.Parameter(torch.zeros(n_layer)). In forward loop, before block call: if i > 0, compute velocity = x - x_prev; x = x + deriv_lambdas[i] * velocity. Store x_prev = x.clone() (or detach). Add deriv_lambdas to optimizer with lr=scalar_lr*0.01. (3) Keep factored embeddings (already in codebase): embed_dim=256, wte_proj. (4) Remove Peri-LN: change Block.forward from `x = x + norm(self.attn(...))` back to `x = x + self.attn(norm(x), ...)` and `x = x + self.mlp(norm(x))`. (5) Update parameter assertion in setup_optimizer.

**Information gain analysis:**
- P(success): 0.45
- If CONFIRMED, I learn: The three confirmed improvements stack to a meaningful combined delta, and the stacking discount does not worsen catastrophically with additional components. The best possible configuration uses all three. This sets a new baseline for future exploration.
- If REFUTED, I learn: Stacking more than two improvements produces diminishing or negative returns, suggesting the architecture has a "capacity ceiling" where multiple modifications compete for the same optimization degrees of freedom.
- Expected information gain: high -- the stacking discount rate with 3 components is genuinely unknown and both outcomes update beliefs about the fundamental combinability of architectural improvements.

**Evaluator P(success):** 0.30
**Belief divergence:** 0.15

**Evaluator notes:** This is the highest-priority experiment per standing authorisations from cycle 3. All three components are individually CONFIRMED. The key unknown is the triple-stack discount rate. The Researcher's predicted delta of -0.006 assumes a ~50% discount from the additive sum (-0.012), which is more pessimistic than the observed 37% pairwise discount from H10. I am more sceptical: adding a third component introduces two new pairwise interactions (H4-H14, H6-H14) plus a three-way interaction. The gradient coupling at the residual stream -- where all three interventions meet -- may produce a discount well above 50%. My P(success)=0.30 reflects that the -0.006 threshold requires the discount not to worsen beyond ~50%, which is uncertain. The VRAM estimate of ~72-73 GB is reasonable (H10 was 72.3 GB, factored embeddings are memory-neutral). Implementation is clear -- all three components have been individually compiled. The cross-domain paper citation is weak (the "superposition principle from linear systems" is a loose analogy at best; confirmed improvements are empirically subadditive, not additive), but both parent hypotheses had proper citations. Approved. **Testing priority: 1 (highest)** -- this directly answers the standing authorisation and has the highest belief divergence in this batch.

---

## Hypothesis 16: Factored Embeddings Bottleneck Dim Sweep (192-dim) | Status: INCONCLUSIVE

### Sprint Contract

**Intervention:** Change the factored embedding bottleneck dimension from 256 (H14, confirmed) to 192. This increases the compression ratio from 60% to 70% (embedding params: vocab_size x 192 = 6.3M + projection 192x640 = 0.12M = 6.4M total, vs 13.1M at 256-dim, vs 32.2M at full). The projection matrix (192 x 640) is still optimized by Muon. Everything else matches the original baseline (ReluSquared MLP, standard residuals, no PD derivative).
**Subsystem:** embeddings
**Papers:** [Domain A] Lan et al. 2020, "ALBERT" (arXiv:1909.11942) -- factored embeddings with variable bottleneck dimension x [Domain B] Information theory: rate-distortion theory (Shannon 1959). The rate-distortion function R(D) defines the minimum bit rate needed to represent a source with distortion at most D. H14 at 256-dim is one point on this curve. Moving to 192-dim tests whether we are above or below the "knee" of the rate-distortion curve for token embeddings at 1800 training steps. If 192-dim improves over 256-dim, the effective rank of the embedding is below 192 and we have additional regularization headroom. If 192-dim regresses, we have crossed the knee and the bottleneck is losing task-relevant information.
**Closest prior art:** H14 tested bottleneck_dim=256 and achieved delta=-0.003215 (CONFIRMED). The Evaluator's exploration directive specifically requests bottleneck_dim variations (128, 192, 384). We choose 192 as the most informative single test: it is 25% narrower than 256, enough to distinguish genuine improvement from noise, while not so extreme (128) that information loss dominates.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 from original baseline (i.e., val_bpb < 0.956340)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003
**Predicted confidence:** low
**P(regression):** 0.20 (192-dim may cross the rate-distortion knee, losing important token distinctions that 256-dim preserves)
**Case for failure:** 192-dim is a more aggressive bottleneck. The effective rank of the embedding at 1800 steps may be between 192 and 256 -- in which case 192-dim loses marginal but important information. The Muon-optimized projection (192 x 640) has a more extreme aspect ratio than (256 x 640), which may interact differently with Newton-Schulz orthogonalization. Additionally, with fewer embedding dimensions, each dimension must carry more information, making the AdamW optimization of the embedding table harder at 1800 steps.
**Feasibility pre-flight:** Reduces model params further: from ~65M (H14 at 256-dim) to ~58M (at 192-dim). VRAM essentially unchanged from baseline (~68 GB). The projection matrix shape (192, 640) will create a new Muon group (different shape from any existing matrix). torch.compile: no new ops, should compile identically.
**Implementation sketch:** In GPT.__init__, change self.embed_dim from 256 to 192. This propagates to wte = nn.Embedding(vocab_size, 192) and wte_proj = nn.Linear(192, n_embd, bias=False). Update init_weights for wte_proj. No other changes needed -- the rest of the forward pass receives the same 640-dim representation after projection.

**Information gain analysis:**
- P(success): 0.30
- If CONFIRMED, I learn: The rate-distortion knee for token embeddings at 1800 steps is below 192 dims, meaning the embedding is even more over-parameterized than H14 revealed. Further compression (128-dim) may also work. The regularization benefit increases with compression.
- If REFUTED, I learn: The rate-distortion knee is between 192 and 256 dims, precisely locating the information bottleneck for this training regime. 256-dim is near-optimal and further compression is counterproductive.
- Expected information gain: high -- regardless of outcome, this precisely locates one point on the rate-distortion curve. Combined with H14 (256-dim), we get two data points to interpolate. Both outcomes are informative and neither is strongly expected.

**Evaluator P(success):** 0.20
**Belief divergence:** 0.10

**Evaluator notes:** This was explicitly authorised in cycle 3 as a bottleneck variation follow-up. The embeddings subsystem is HIGH SURPRISAL, so further exploration is warranted. My scepticism: H14 at 256-dim was borderline CONFIRMED (delta=-0.003215, clearing the threshold by only 0.000215). Going 25% narrower to 192-dim risks crossing the rate-distortion knee where information loss outweighs regularization gain. The Researcher's predicted delta of -0.003 is exactly at the threshold -- this is the pattern of overestimating weak interventions that calibration notes identify (weak interventions are overestimated by 2-4x). If I apply the calibration correction, the actual delta may be closer to -0.001 to -0.002, which would be INCONCLUSIVE. However, the information gain is genuinely high: combined with H14, this gives two points on the rate-distortion curve. Implementation is trivial (change one constant). Approved as LOW PRIORITY per cycle 3 authorisation. **Testing priority: 4** -- run after stacking experiments and EMA.

---

## Hypothesis 17: Tail-Phase EMA with Narrow Window (beta=0.99, warmdown only) | Status: REFUTED

### Sprint Contract

**Intervention:** Apply EMA weight averaging ONLY during the warmdown phase (steps 540-1800, the last 70% of training), with a narrow beta=0.99 (effective window ~100 steps). This addresses both failure modes of H12: (1) the broad window (beta=0.999, ~1000 steps) incorporated unconverged early weights -- now EMA starts only after warmup+plateau phase; (2) the wide window averaged across the entire learning rate lifecycle -- now the 100-step window captures only local weight neighborhoods within the warmdown. At the end of training, evaluate with EMA weights. The training dynamics are unchanged (EMA is a shadow copy).
**Subsystem:** training-loop
**Papers:** [Domain A] "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits" (arXiv:2411.18704, Nov 2024) -- shows EMA acts as implicit regularizer, with optimal decay rates 0.968-0.998; recommends that EMA requires less learning rate decay since averaging naturally reduces noise. Also: "When, Where and Why to Average Weights?" (arXiv:2502.06761, Feb 2025) -- finds optimal averaging window is ~1% of total training budget. For 1800 steps, 1% = 18 steps. Our 100-step window is conservative but in the right order of magnitude. x [Domain B] Stochastic approximation theory: Polyak-Ruppert tail averaging (Polyak 1991, Ruppert 1988). Tail averaging retains the advantages of iterate averaging while ensuring the initial error is forgotten exponentially fast. The optimal rate O(d*sigma^2/n) is achieved by averaging only the "tail" of the iterate sequence (the last portion after burn-in). In our context, the warmup+plateau phase (steps 0-540) is the burn-in, and the warmdown phase (steps 540-1800) is the tail. EMA during warmdown only is a soft version of Polyak-Ruppert tail averaging, where exponential weighting replaces uniform averaging.
**Closest prior art:** H12 tested EMA with beta=0.999 over the full training run and was REFUTED (EMA val_bpb=1.142, last-iterate +0.004). The failure was cleanly attributed to the wide window incorporating early unconverged weights. The Evaluator's notes specifically state: "If EMA is to be attempted again, beta must be much narrower (0.99 or 0.995, ~100-200 step window) and applied only during the warmdown phase." This hypothesis directly implements that recommendation.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 from original baseline (i.e., val_bpb < 0.956340)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.001
**Predicted confidence:** low
**P(regression):** 0.15 (the narrow window and late start avoid the catastrophic failure mode of H12, but the EMA shadow update lerp_ calls may still cause torch.compile graph perturbation or bf16 numerical effects on the training params, as possibly observed in H12's last-iterate regression)
**Case for failure:** The warmdown schedule already decays the learning rate from 1.0x to 0.01x over 1260 steps. During warmdown, the late updates are already very small, making the last-iterate weights a de facto average of the recent trajectory. EMA on top of this is redundant -- the "noise reduction" that EMA provides is already achieved by the decaying LR. Additionally, Muon's Newton-Schulz produces weights on a near-orthogonal manifold; EMA of near-orthogonal matrices is NOT near-orthogonal, potentially pushing EMA weights off-manifold. With only 100-step effective window, this off-manifold drift is small but may still degrade quality. The 2502.06761 paper found averaging windows of ~1% optimal (18 steps for us), suggesting even beta=0.99 (100 steps) may be too wide.
**Feasibility pre-flight:** EMA shadow params add ~100 MB memory (negligible at 68 GB baseline). The lerp_ updates happen outside torch.compile (in the training loop after optimizer.step()). Wall-clock: one lerp_ pass per step during warmdown only (1260 of 1800 steps). Estimate <2% overhead. No model architecture changes.
**Implementation sketch:** (1) After model init, create ema_params dict (clone all params). (2) Compute warmdown start step: warmdown_start = int(STEP_BUDGET * (1.0 - WARMDOWN_RATIO)). (3) After optimizer.step(), if step >= warmdown_start: for name, p in model.named_parameters(): ema_params[name].lerp_(p, 1 - 0.99). (4) Before final eval, swap model params with ema_params. (5) Run validation. (6) Also log last-iterate val_bpb for comparison (swap back, eval again).

**Information gain analysis:**
- P(success): 0.15
- If CONFIRMED, I learn: Even with warmdown, the last-iterate has residual noise that tail averaging can remove. The warmdown and EMA are not fully redundant -- EMA provides additional variance reduction that warmdown alone does not. This opens further tuning (beta=0.995, 0.98) and combining EMA with other confirmed improvements.
- If REFUTED, I learn: The warmdown schedule already achieves optimal weight averaging at this training length, confirming that the training loop is well-optimized. The training-loop subsystem can be deprioritized. This is the expected outcome given that warmdown covers 70% of training.
- Expected information gain: medium -- the outcome is skewed toward failure (P(success)=0.15), which makes success highly informative. Failure would confirm the warmdown-EMA redundancy hypothesis from H12's analysis, providing moderate closure on the training-loop subsystem.

**Evaluator P(success):** 0.10
**Belief divergence:** 0.05

**Evaluator notes:** The Evaluator's own cycle 3 notes stated: "If EMA is to be attempted again, beta must be much narrower (0.99 or 0.995, ~100-200 step window) and applied only during the warmdown phase." This hypothesis directly implements that recommendation. However, the exploration directives classify training-loop/weight-averaging as LOW SURPRISAL / diminishing returns. The Researcher's own case for failure is compelling: the warmdown already decays LR to 0.01x, making EMA redundant. I agree with the Researcher's P(success)=0.15 assessment and am even more sceptical at P=0.10. The Muon off-manifold concern (EMA of near-orthogonal matrices is not near-orthogonal) is real but mitigated by the narrow 100-step window. The key question is whether the paper's finding of optimal windows at ~1% of training (18 steps for us) means beta=0.99 (100 steps) is still too wide. Implementation is low-risk and cheap (no architecture changes, negligible VRAM). Approved because it closes out the training-loop subsystem and low belief divergence means this is a low-information experiment that can fill a slot if higher-priority experiments are fast. **Testing priority: 3** -- run after stacking experiments but before bottleneck sweep.

---

## Hypothesis 18: ALiBi Additive Positional Bias in Sliding Window Layers | Status: REJECTED

### Sprint Contract

**Intervention:** Add ALiBi-style linear attention biases to the sliding-window (S) layers ONLY, while keeping RoPE for all layers. In the SSSL window pattern, layers 0, 1, 2 are short-window (S) and layer 3 (mod 4) is long-window (L). For S-layers, add a fixed (non-learnable) linear distance penalty to the attention logits: bias(i,j) = -m * |i - j|, where m is a per-head slope following ALiBi's geometric progression (m_k = 2^(-8*k/n_heads) for head k). This bias is applied AFTER QK-norm and BEFORE softcap, as an additive term to the attention logits. RoPE remains applied to Q and K for all layers. The hypothesis is that sliding-window layers benefit from an explicit recency bias that complements RoPE's rotational encoding.
**Subsystem:** positional (completely untested)
**Papers:** [Domain A] Press et al. 2022, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (arXiv:2108.12409) -- ALiBi adds a fixed linear bias to attention scores, with per-head slopes that create different "attention temperatures" across heads. Key finding: ALiBi achieves comparable perplexity to sinusoidal/learned positional embeddings while enabling length extrapolation. x [Domain B] Spatial frequency filtering in signal processing: in communications, matched filters apply distance-dependent attenuation to incoming signals, with the optimal attenuation profile determined by the channel characteristics. Sliding-window attention is analogous to a bandpass filter: it selects a local frequency band (nearby tokens). Adding ALiBi bias within the window is analogous to applying a matched filter within the passband -- it shapes the frequency response to favor the most relevant part of the local context rather than treating all in-window positions equally.
**Closest prior art:** ALiBi and RoPE are typically used as alternatives, not combined. A few papers (e.g., "Length Generalization of Causal Transformers without Position Encodings" from ACL Findings 2024) explore positional encoding combinations. Our approach is novel in two ways: (1) applying ALiBi only to sliding-window layers, not globally; (2) combining ALiBi with RoPE in the same layers. The rationale: RoPE provides rotational position information for content-based matching, while ALiBi provides an explicit recency inductive bias for local context.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 from original baseline (i.e., val_bpb < 0.956340)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.001
**Predicted confidence:** low
**P(regression):** 0.25 (the additive bias changes the attention logit distribution, and the attention logit regime has been shown to be rigid -- H8 and H13 both regressed from logit modifications. However, ALiBi is additive and fixed (no learnable params), and applied only to S-layers, which is a lighter touch than H8/H13's global modifications. The softcap at 15 provides a safety net against logit saturation.)
**Case for failure:** The attention logit regime is rigid (H8: +0.005, H13: +0.007 regressions). ALiBi's additive bias changes the logit distribution, which is exactly what the previous two failures were doing. The key difference is that ALiBi is fixed (not learned) and additive (not multiplicative like H8 or geometric like H13), but the evaluator has warned that "one more failure with 0 confirmations will trigger BLOCKED" for the attention subsystem. At 2048 context with sliding window of 1024, the ALiBi bias range is modest (max penalty = m * 1024, where m is the steepest slope ~2^(-8/5) = 0.30, giving max bias = ~307 -- this would saturate softcap). CRITICAL: the ALiBi slopes must be much smaller than the standard formula to work within the softcap=15 regime. The slopes should be calibrated so max_bias < softcap/2 = 7.5. With window=1024: m_max = 7.5/1024 = 0.0073. This is much gentler than standard ALiBi.
**Feasibility pre-flight:** Zero new learnable parameters (ALiBi biases are fixed). Memory: the bias matrix is (T, T) but only needs the sliding window portion, and can be computed on-the-fly. However, flash attention (fa3) may not support arbitrary additive biases -- this is a CRITICAL implementation concern. If fa3 does not support additive attention biases, this hypothesis is infeasible without falling back to non-flash attention (which would blow up VRAM). The implementer MUST check fa3's API for bias support before proceeding.
**Implementation sketch:** (1) Check if fa3.flash_attn_func supports an attn_bias or alibi_slopes argument. If not, this hypothesis is INFEASIBLE -- abort. (2) If supported: compute ALiBi slopes for 5 heads using geometric progression, scaled so max_bias = 7.5 within the sliding window. (3) In CausalSelfAttention.forward, for S-layers only, pass the ALiBi slopes to fa3. (4) L-layers (every 4th, plus last) use standard RoPE-only attention. (5) No optimizer changes, no new parameters.

**Information gain analysis:**
- P(success): 0.15
- If CONFIRMED, I learn: The positional subsystem has exploitable structure. Sliding-window layers benefit from an explicit recency bias on top of RoPE. This opens further positional encoding exploration (different slope schedules, learnable slopes, applying to L-layers too).
- If REFUTED, I learn: Combined with H8 and H13, this would BLOCK the attention subsystem entirely (3 failures, 0 confirmations), confirming that attention logit modifications of any kind are counterproductive at this scale. The attention subsystem is fully optimized as-is.
- Expected information gain: high -- this is the first test of the positional subsystem AND the decisive test for the attention/all subsystem. If it fails, attention is BLOCKED. If it succeeds, both positional and attention/structural are opened. The binary outcome has high information content regardless of direction.

**Evaluator P(success):** N/A (REJECTED)
**Belief divergence:** N/A

**Evaluator notes:** REJECTED. Reason: This modifies the attention logit distribution, which is the exact failure mode of both H8 (+0.005 regression) and H13 (+0.007 regression). The findings explicitly state: "Any intervention that changes the logit distribution disrupts the careful balance between [QK-norm and softcap] constraints." ALiBi adds an additive bias to attention logits -- this IS a logit distribution modification. The argument that ALiBi is "fixed, not learned" does not address the core problem: the logit distribution is already well-conditioned, and adding any bias (fixed or learned) disrupts that conditioning. The feasibility is also uncertain: the implementation sketch acknowledges fa3/flex_attention may not support additive biases, making this potentially unimplementable. Furthermore, if this fails, it triggers BLOCKED on the entire attention/all subsystem (3 failures, 0 confirmations), which would foreclose all future attention experiments -- a high cost for a low-probability experiment. The Researcher's own P(regression)=0.25 and P(success)=0.15 confirm this is a poor expected-value bet. The ALiBi slope scaling problem (needing max_bias < 7.5 within softcap) means the actual recency bias would be negligibly small (slope ~0.007 per position). With 5 remaining run slots (17-12=5), this slot is better used elsewhere. If the Researcher wants to explore the positional subsystem, propose an intervention that does NOT modify attention logits (e.g., different RoPE base frequency, or positional information injected into the residual stream rather than the attention logits).

---

## Hypothesis 19: SwiGLU + Factored Embeddings (H6+H14 stack, no PD) | Status: INCONCLUSIVE

### Sprint Contract

**Intervention:** Stack SwiGLU activation (H6, hidden_dim=1792) with factored embeddings (H14, bottleneck_dim=256), WITHOUT the PD derivative term (H4). This is a simpler two-component stack than H15, testing whether H6+H14 produces a stronger result than H6+H4 (which was H10, delta=-0.005692). The rationale: H14's parameter reduction may interact differently with SwiGLU (which changes the MLP structure) than with PD (which changes the residual stream). If H6+H14 > H6+H4, it suggests the regularization benefit of factored embeddings has better synergy with activation changes than with residual dynamics.
**Subsystem:** stacking
**Papers:** [Domain A] Shazeer 2020, "GLU Variants Improve Transformer" (arXiv:2002.05202) x [Domain B] Information theory: rate-distortion theory (Shannon 1959) -- the interaction between input compression (factored embeddings) and intermediate representation quality (SwiGLU gating) tests whether the rate-distortion bottleneck at the input affects the activation function's ability to select features. If the bottleneck removes noise from the embedding, SwiGLU's gating may be more effective because it operates on cleaner input features.
**Closest prior art:** H10 stacked H6+H4 (delta=-0.005692, 37% subadditive). This hypothesis stacks H6+H14 instead, providing a different two-way interaction measurement. The additive sum of H6+H14 would be -0.008 (-0.005 + -0.003). At 37% discount: -0.005. At 20% discount (if these interact less): -0.006.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.005 from original baseline (i.e., val_bpb < 0.954340)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.005 and -0.003
**REFUTED if:** delta > -0.003 or CRASH

**Predicted delta:** -0.005
**Predicted confidence:** low
**P(regression):** 0.05 (both components individually confirmed)
**Case for failure:** SwiGLU at hidden_dim=1792 changes the MLP capacity profile. The factored embedding at 256-dim provides a lower-rank input to the first layer. If SwiGLU's gating mechanism benefits from high-rank input features (to have more diverse features to gate), the factored embedding could throttle SwiGLU's effectiveness. The stacking discount from H10 (37%) already shows components interact subadditively through the residual stream. H6+H14 interact through a different pathway (input representation quality), but the discount may be similar or worse.
**Feasibility pre-flight:** VRAM: H6 alone used 67.5 GB. Factored embeddings are memory-neutral. Expected ~67-68 GB. This is the lightest stacking configuration. SwiGLU changes MLP to 3 matrices (c_fc, c_gate, c_proj) at hidden_dim=1792. Factored embeddings: wte (vocab_size x 256) + wte_proj (256 x 640). Both individually compile. No new scalar params.
**Implementation sketch:** Starting from baseline train.py: (1) Replace MLP with SwiGLU (same as H6 implementation). (2) Keep factored embeddings (already in codebase at 256-dim). (3) Remove Peri-LN from Block.forward. (4) Do NOT add PD derivative terms. (5) Update parameter assertion.

**Information gain analysis:**
- P(success): 0.35
- If CONFIRMED, I learn: SwiGLU and factored embeddings have good synergy, possibly better than SwiGLU+PD. The input compression helps the gated activation. This informs the optimal triple-stack configuration (if H15 and H19 both run, comparing their deltas reveals whether PD adds value on top of H6+H14).
- If REFUTED, I learn: Factored embeddings do not synergise well with SwiGLU -- the lower-rank input may indeed throttle gating quality. This suggests the PD derivative is the better stacking partner for SwiGLU, and the triple-stack should not be expected to exceed H10 by much.
- Expected information gain: medium-high -- the pairwise interaction between embedding compression and activation gating is unknown. The result disambiguates which two-way combination is strongest, directly informing whether H15 (triple-stack) or a simpler H6+H14 is the optimal configuration.

**Evaluator P(success):** 0.25
**Belief divergence:** 0.10

**Evaluator notes:** Both components are individually CONFIRMED. The implementation is straightforward and VRAM-light (~67-68 GB). My concern is the stringent threshold: delta < -0.005 requires the stacking discount to be no worse than ~37.5% (additive sum is -0.008, at 37.5% discount = -0.005). H10 (H6+H4) showed exactly 37% discount, so this threshold assumes H6+H14 interact no worse than H6+H4. But the interaction pathway is different: H14 changes the input representation quality (lower-rank embeddings), which directly affects SwiGLU's input features. If lower-rank embeddings provide less diverse features for SwiGLU's gating to operate on, the discount could be worse. The Researcher's predicted delta of -0.005 is exactly at threshold -- the calibration notes show the Researcher overestimates weak interventions by 2-4x, but stacking experiments have been more accurately predicted (H10: 1.23x overestimate). I give a slight edge to the Researcher's prediction being in the right ballpark. The strategic value of this experiment depends on whether H15 (triple-stack) also runs: if both run, comparing H15, H19, and H10 reveals all three pairwise interaction strengths. If budget is tight, H15 is more valuable because it tests the full stack. **Testing priority: 2** -- run alongside or after H15.

---

## Hypothesis 14: Embed-then-Project with Bottleneck for Token Embeddings | Status: CONFIRMED

### Sprint Contract

**Intervention:** Replace the single large embedding table (vocab_size x n_embd = 50257 x 640) with a factored embedding: a smaller embedding table (vocab_size x bottleneck_dim, where bottleneck_dim=256) followed by a linear projection (bottleneck_dim x n_embd = 256 x 640). The projection is optimized with Muon (it is a 2D weight matrix inside transformer). This factorization reduces the embedding parameter count by 60% while adding a learned linear map that can capture cross-dimension correlations. The freed parameters are "reinvested" -- the model becomes parameter-lighter in the embedding, which may improve generalization through implicit regularization.
**Subsystem:** embeddings
**Papers:** [Domain A] Lan et al. 2020, "ALBERT: A Lite BERT for Self-supervised Learning of Language Representations" (arXiv:1909.11942) -- introduced factored embedding parameterization where vocab_size x hidden_dim is decomposed into (vocab_size x E) x (E x hidden_dim). Showed this reduces parameters without hurting performance, and sometimes improves it at small scale because it acts as a regularizer. x [Domain B] Information theory: rate-distortion theory (Shannon 1959). Rate-distortion theory establishes that for a source with redundant structure, there exists an optimal compression rate below which distortion increases sharply (the rate-distortion function). Token embeddings in language models are redundant: many tokens have similar semantic roles, and the full vocab_size x n_embd matrix has rank much lower than min(vocab_size, n_embd). The bottleneck projection forces the embedding through a lower-dimensional representation, analogous to approaching the rate-distortion bound -- compressing the embedding to a rate that preserves task-relevant information while discarding noise. If the embedding matrix's effective rank is < 256, the bottleneck loses nothing; if it is > 256, the distortion may hurt.
**Closest prior art:** ALBERT's factored embeddings are well-known but have not been tested in this specific GPT configuration with Muon optimizer. The key difference from ALBERT: (a) our embedding is followed by RMSNorm (line 285), which may interact differently with the projection; (b) Muon optimizes the projection matrix, which could make the projection more effective than Adam-trained projections; (c) our model has value embeddings that use separate embedding tables -- the factorization applies ONLY to wte, not to value embeddings.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.001
**Predicted confidence:** low
**P(regression):** 0.25 (the bottleneck may be too aggressive -- if important information is lost in the 256-dim bottleneck that the model cannot recover from, this will regress. The embedding is already followed by norm and then processed by 10 transformer layers, so the model has capacity to compensate, but the first-layer representation quality is critical per the Friis cascade analogy.)
**Case for failure:** The embedding bottleneck reduces the effective input dimensionality to the first layer from 640 to 256. Even though the projection maps it back to 640, the rank of the initial representation is limited to 256. If the model needs more than 256 linearly independent embedding directions (plausible for a 50K+ vocab), the bottleneck destroys information that cannot be recovered. The embedding_lr (0.6) is high for AdamW; the factored embedding's smaller table may train differently at this LR. The projection matrix (256 x 640) will be grouped by Muon due to its 2D shape, which is fine for optimization but means the interaction between the embedding (AdamW) and projection (Muon) is a new regime. Additionally, at 1800 steps, the model may not have time to learn an effective projection -- ALBERT's benefits were shown with much longer training.
**Feasibility pre-flight:** The factored embedding reduces wte from (50257 x 640) = 32.2M params to (50257 x 256) + (256 x 640) = 12.9M + 0.16M = 13.1M params, saving 19.1M params. VRAM: the embedding table is stored in bf16; saving 19M params saves ~38 MB -- negligible in the 68 GB context. The projection adds a small matrix multiply per forward pass. torch.compile: nn.Embedding -> nn.Linear is standard, should compile. Note: lm_head is NOT factored (it maps from 640 -> vocab_size and is separate from wte -- no weight tying in the baseline).
**Implementation sketch:** (1) Replace `self.transformer.wte = nn.Embedding(vocab_size, n_embd)` with `self.transformer.wte = nn.Embedding(vocab_size, 256)` and `self.transformer.wte_proj = nn.Linear(256, n_embd, bias=False)`. (2) In forward: `x = self.transformer.wte_proj(self.transformer.wte(idx))`. (3) In init_weights: init wte with std=1.0 (unchanged), init wte_proj with uniform(-s, s) where s = 3**0.5 * n_embd**-0.5. (4) In setup_optimizer: wte_proj is inside transformer, so it will be auto-grouped by shape (256, 640) into a Muon group. Update the parameter assertion to account for the new parameter. (5) The embedding_params list must still contain only wte (the embedding table itself), not wte_proj. (6) Exclude wte_proj from nparams_exclude in FLOPs computation since it is a learned projection.

**Information gain analysis:**
- P(success): 0.15
- If CONFIRMED, I learn: The token embedding is over-parameterized at this scale, and factorization acts as a useful regularizer. The effective rank of the optimal embedding is < 256. This opens further compression (bottleneck_dim=128, 192) and suggests the model's capacity is better allocated in transformer layers than in embeddings.
- If REFUTED, I learn: The full-rank embedding is necessary at this vocab size and model dimension. The information content of the embedding exceeds what a 256-dim bottleneck can capture. This closes the factored embedding avenue and suggests embedding capacity is already near-optimal.
- Expected information gain: medium -- the embeddings subsystem is completely untested. Success would be mildly surprising; failure would be expected but still informative about embedding rank requirements.

**Rollback plan:** Revert to single nn.Embedding(vocab_size, n_embd). Remove wte_proj.

**Evaluator P(success):** 0.08
**Belief divergence:** 0.07

**Evaluator notes:** The embeddings subsystem is untested, so any result provides new information. However, I am more sceptical than the Researcher. The bottleneck_dim=256 is aggressive -- it reduces the embedding's effective rank by 60%. At 1800 training steps, the model has limited time to learn a good projection from 256 to 640 dimensions. The parameter "savings" of 19M params is misleading: the model does not benefit from having fewer parameters (this is not a regularization-limited regime at 1800 steps), and the freed capacity is not reinvested anywhere. The ALBERT analogy is weak because ALBERT was overparameterized and trained for much longer. My main concern is the interaction with setup_optimizer: the wte_proj matrix (256, 640) will be grouped with Muon by the shape-based loop at line 268. This means wte_proj gets Muon optimization while wte gets AdamW at embedding_lr. This split optimization of two sequential layers (embedding lookup then linear projection) is a new regime. The assertion on line 255 must be updated to account for wte_proj. Note to implementer: (1) wte_proj must NOT be in embedding_params -- it should be picked up by the matrix_params loop since it is inside transformer; (2) update the parameter count assertion; (3) update nparams_exclude in estimate_flops to still exclude wte but NOT exclude wte_proj (it contributes to forward FLOPs); (4) init wte_proj appropriately. Despite low P(success), the experiment is approved because the embeddings subsystem is completely unexplored and the implementation is low-risk.

---

# Cycle 6 Hypotheses

## Hypothesis 20: Learnable Per-Head RoPE Frequencies on H10 Baseline | Status: REJECTED

### Sprint Contract

**Intervention:** Replace the fixed RoPE frequency schedule (inv_freq = 1 / base^(2i/head_dim), base=10000) with learnable per-head log-frequencies. Each attention head gets its own set of 64 frequency parameters (head_dim//2 = 64), initialized to the standard RoPE values. These are optimized with AdamW at scalar_lr. The total addition is 5 heads x 10 layers x 64 freqs = 3200 scalar parameters. Applied on top of the H10 (SwiGLU+PD) baseline.
**Subsystem:** positional
**Papers:** [Domain A] Huang & Chen 2025, "Optimizing the Learnable RoPE Theta Parameter in Transformers" (IEEE Access) -- demonstrates that making RoPE theta learnable with separate learning rates and sigmoid constraints provides consistent improvements in validation loss across Tiny Shakespeare, WikiText-103, and IWSLT'14 datasets. The key insight is that fixed geometric frequency spacing is suboptimal; learned frequencies adapt to the data's positional structure. x [Domain B] Kuramoto model of coupled oscillators with heterogeneous natural frequencies (Kuramoto 1975, Strogatz 2000) -- in physics, systems of coupled oscillators with different natural frequencies synchronize only when coupling strength exceeds a critical threshold. Oscillators with frequencies near the mean synchronize first (forming a "locked" cluster), while outlier frequencies remain incoherent. The analogy: attention heads are coupled oscillators where RoPE frequencies set each head's "natural frequency." Fixed identical frequencies force all heads to operate in the same positional regime. Allowing heterogeneous frequencies lets different heads specialize: some lock to short-range positional patterns (high frequency), others to long-range semantic patterns (low frequency), analogous to the Kuramoto order parameter emerging from frequency heterogeneity.
**Closest prior art:** The learnable RoPE theta paper (Huang & Chen 2025) optimizes a single global theta parameter. CARoPE (arxiv 2507.23083) makes frequencies context-dependent but adds complexity. Our approach is simpler: per-head frequencies initialized from standard RoPE, learned via AdamW. The key difference is we apply this to a small model (640-dim, 5 heads) with Muon optimizer on a short training run (1800 steps), where the positional subsystem is completely untested.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to H10 baseline of 0.953648, i.e., val_bpb < 0.950648)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**P(regression):** 0.15 (frequencies could drift to degenerate values, though sigmoid constraints or init-anchored LR could prevent this; unlike H8 temperature scalars, frequencies do not interact with the rigid QK-norm+softcap logit regime -- they modify which dimensions encode position vs content, not the logit magnitudes)
**Case for failure:** At 2048 context length and 1800 training steps, the model may not see enough positional diversity to learn better frequencies than the geometric default. The standard RoPE base=10000 is already well-tuned across many architectures. With only 5 heads, there is limited capacity for frequency specialization. The 3200 new scalar parameters are trained with AdamW at scalar_lr, which may be too aggressive or too conservative. torch.compile may break if the precomputed cos/sin buffers need to be recomputed per step (they should be computed once from the learned frequencies and cached, but dynamic recomputation would cause graph breaks). The interaction with Muon-optimized Q/K weights is unknown.
**Feasibility pre-flight:** 3200 scalar parameters (negligible). VRAM: ~same as H10 baseline (72.3 GB) plus negligible for the frequency params. torch.compile concern: the cos/sin computation must happen outside the compiled region, or the frequencies must be treated as buffers recomputed each forward pass. Since the frequencies change slowly, computing cos/sin from learnable inv_freq each forward call is feasible -- it is a small matmul (T x 64). The key risk is graph breaks; the safest approach is to compute cos/sin inside forward() from self.inv_freq (a parameter) rather than using precomputed buffers.
**Implementation sketch:** (1) Replace the fixed cos/sin buffers with a learnable parameter: self.inv_freq = nn.Parameter(torch.zeros(n_layer, n_head, head_dim//2)) initialized to the standard geometric schedule (same for all heads initially, or directly from the existing computation). (2) In forward(), compute cos = cos(t_outer @ inv_freq[layer, head]) per layer per head. (3) Pass per-head cos/sin to apply_rotary_emb. (4) Add inv_freq to a new optimizer group with lr=scalar_lr, betas=(0.8, 0.95), weight_decay=0. (5) The Q/K projections already produce per-head outputs; RoPE is applied per-head. The change is surgical: only the frequency source changes from fixed buffer to learnable parameter. (6) Build on top of H10 (SwiGLU+PD) codebase.

**Information gain analysis:**
- P(success): 0.20
- If CONFIRMED, I learn: The positional encoding subsystem is a genuine bottleneck at this scale. Per-head frequency specialization improves the model's ability to attend to different positional scales. This opens a new subsystem for optimization (positional has 0 prior tests).
- If REFUTED, I learn: The standard RoPE frequency schedule is already near-optimal at 2048 context length, and the positional subsystem is not a bottleneck. This matches the v5-v6 prior that "positional encoding changes have minimal effect at 2048 context."
- Expected information gain: high -- the positional subsystem is completely untested (0 experiments). ANY result provides new information. The P(success)=0.20 means both outcomes are plausible, and either would update our model of whether positional encoding is exploitable at this scale.

**Evaluator P(success):** 0.12
**Belief divergence:** 0.08

**Evaluator notes:** REJECTED. With only 1 engineering run remaining (16/17), this experiment's P(success)=0.20 (Researcher) / 0.12 (Evaluator) is too low to justify the budget. The Researcher's own predicted delta (-0.002) is below the confirmation threshold. The torch.compile risk from dynamic per-head cos/sin computation is non-trivial. The positional subsystem is untested and therefore exploratory -- exploration is appropriate when budget permits, but we have no budget for exploration. The authorized follow-up H22 answers a more critical question with higher P(success).

---

## Hypothesis 21: Cross-Block Attention Residual Shortcut (ANCRe-inspired) on H10 Baseline | Status: REJECTED

### Sprint Contract

**Intervention:** Add learnable weighted shortcuts from the attention output of earlier blocks to the attention input of later blocks, inspired by ANCRe's finding that MHSA-to-MHSA connectivity improves convergence. Specifically: each block's attention sublayer receives an additional weighted input from the attention output of the block 2 layers prior (skip-2 pattern). The shortcut coefficient is a learnable scalar per layer, initialized at 0 (passthrough = identity behavior at init). This creates an anticipatory cross-block attention pathway without modifying the sequential residual stream. Applied on top of H10 (SwiGLU+PD) baseline.
**Subsystem:** attention/structural
**Papers:** [Domain A] ANCRe: Adaptive Neural Connection Reassignment (arxiv 2602.09009, 2025) -- demonstrates that learning residual connection topology improves convergence by 24-46% fewer iterations. Key finding: MHSA-to-MHSA connectivity is more informative than FFN-to-FFN. The approach parameterizes shortcut coefficients via softmax normalization and incurs <1% runtime overhead. At 60M params, validation perplexity improved from baseline by establishing cross-block attention connectivity. x [Domain B] Impedance matching in transmission line theory (electrical engineering) -- when signals travel through cascaded transmission line segments with mismatched impedances, reflections occur at each junction, degrading signal quality. Quarter-wave transformers or matching networks minimize these reflections. In a transformer architecture, each block's attention sublayer produces an "attention signal" that propagates through the residual stream. If the attention outputs of different blocks have mismatched "impedance" (different magnitude/direction distributions), the later blocks' attention inputs contain residual "reflections" from earlier mismatches. A cross-block shortcut acts as an impedance matching network: it provides a direct path for attention-relevant signal from an earlier block to reach a later block's attention, bypassing the intervening MLP layers that act as "impedance discontinuities." The initialization at 0 ensures no disruption at start (matched impedance by default).
**Closest prior art:** ANCRe uses full softmax-normalized connectivity across all layers. Our approach is much simpler: fixed skip-2 pattern with a single learnable scalar per receiving layer (8 parameters for layers 2-9). This avoids the O(L^2) connectivity of full ANCRe while capturing the key finding that cross-block MHSA connectivity helps. The v6 finding that "anticipatory modifications work, retrospective do not" is respected: the shortcut feeds forward from earlier to later blocks (anticipatory).

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to H10 baseline of 0.953648, i.e., val_bpb < 0.950648)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**P(regression):** 0.20 (the shortcut adds a new signal source to the attention input, which could interact poorly with QK-norm if the shortcut signal has different magnitude characteristics; however, the init-at-0 ensures identity behavior initially, and the attention subsystem has only been tested with logit-modifying interventions (H8, H13) which both failed -- this structural intervention does not modify logits)
**Case for failure:** At depth=10, the skip-2 pattern means only layers 2-9 receive shortcuts (8 layers). The total path length is short enough that the standard residual stream may already convey attention-relevant information adequately. ANCRe's benefits were demonstrated primarily on deeper models. The attention/all subsystem is fragile (2 failures, 0 confirmations) and one more failure = BLOCKED. However, this intervention is structurally different from H8 (temperature) and H13 (norm change) -- it modifies attention inputs, not the logit computation. The 8 new scalar parameters must avoid being grouped with Muon (they are 1D, so AdamW is correct). The shortcut creates a new data flow path that may cause torch.compile graph changes.
**Feasibility pre-flight:** 8 scalar parameters (negligible). VRAM: need to store attention outputs from 2 blocks ago -- at most 2 tensors of shape (B, T, n_embd) = (128, 2048, 640) = 167M elements = ~335 MB in bf16. Total VRAM increase ~0.7 GB over H10's 72.3 GB = ~73 GB, within budget. torch.compile: storing and reusing intermediate tensors may cause graph breaks. Safest to store in a list outside the compiled forward pass.
**Implementation sketch:** (1) In GPT.forward(), maintain a list attn_outputs of length n_layer. (2) Before each block's attention call, compute shortcut_input = attn_shortcut_lambda[i] * attn_outputs[i-2] if i >= 2 else 0. (3) Add shortcut_input to the normed attention input: x_attn = norm(x) + shortcut_input. (4) Store each block's attention output in attn_outputs[i]. (5) attn_shortcut_lambda is nn.Parameter(torch.zeros(n_layer)), trained with AdamW at scalar_lr * 0.01 (same conservative LR as deriv_lambdas in H4). (6) Build on top of H10 (SwiGLU+PD) codebase.

**Information gain analysis:**
- P(success): 0.20
- If CONFIRMED, I learn: Cross-block attention connectivity provides useful signal even at depth=10. The attention subsystem can be improved through structural changes (not logit modifications). This would be the first positive attention result after 2 failures, and would strongly inform the path forward.
- If REFUTED, I learn: At depth=10, the residual stream adequately conveys attention-relevant information between blocks. The attention subsystem may be fundamentally rigid even to structural changes, not just logit modifications. Combined with H8 and H13, this would trigger BLOCKED status for attention/all.
- Expected information gain: high -- this tests a fundamentally different kind of attention intervention (structural vs logit-modifying). The outcome updates our model of WHY attention interventions fail: is it about the logit regime specifically, or about the attention subsystem in general?

**Evaluator P(success):** 0.10
**Belief divergence:** 0.10

**Evaluator notes:** REJECTED. The attention/all subsystem is at 2 REFUTED, 0 CONFIRMED -- one more failure triggers BLOCKED status. While I acknowledge this is a structurally different intervention (input modification vs logit modification), the subsystem's track record is dire. The VRAM estimate (+0.7 GB from storing intermediate attention outputs) is plausible but the torch.compile risk of storing and reusing cross-block tensors is real. More critically, with only 1 run remaining, risking an attention experiment that has a 90% chance of failure (by my estimate) and would permanently BLOCK the attention subsystem is poor strategy. The impedance matching analogy from transmission line theory is creative but speculative -- at depth=10 the "transmission line" is very short and impedance matching effects are unlikely to be measurable. Save this for a future budget cycle.

---

## Hypothesis 22: PD Residual + Factored Embeddings Stack (H4+H14, no SwiGLU) | Status: CONFIRMED

### Sprint Contract

**Intervention:** Stack PD residual scaling (H4, CONFIRMED, delta=-0.004197) with factored embeddings (H14, CONFIRMED, delta=-0.003215), WITHOUT SwiGLU. This uses the baseline ReluSquared activation. The hypothesis is that the negative SwiGLU-factored interaction is specific to gated activations, and PD+factored will compose positively because PD modifies residual dynamics (not dependent on input rank) and factored embeddings compress the embedding (compatible with ReluSquared which does not use input-dependent gating). This is a standing authorized follow-up from the Evaluator (cycle 5).
**Subsystem:** stacking/activation+residual
**Papers:** [Domain A] PD controller derivative term applied to residual connections (H4 contract, control theory transfer) -- the derivative term exploits layer-wise trajectory smoothness. Its benefit is orthogonal to input representation rank because it operates on the residual stream dynamics, not on the content of the representations. x [Domain B] Rate-distortion theory (Shannon 1959, information theory) -- the factored embedding operates at the rate-distortion bound, compressing 640-dim embeddings to 256-dim without losing task-relevant information. The question is whether PD scaling's gradient pathway (which uses the difference x - x_prev as a velocity signal) is affected by the lower-rank initial representation from factored embeddings.
**Closest prior art:** H10 (SwiGLU+PD, delta=-0.005692) is the only confirmed two-way stack. H15 (SwiGLU+PD+factored) and H19 (SwiGLU+factored) both showed negative interactions, but these included SwiGLU. This experiment isolates the question: is the negative interaction SwiGLU-specific? The Evaluator explicitly authorized this as a follow-up in cycle 5.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to original baseline 0.959340, i.e., val_bpb < 0.956340)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.004
**Predicted confidence:** medium
**P(regression):** 0.05 (both components individually confirmed; ReluSquared does not use input-dependent gating so there is no mechanism for the factored-embedding input-rank problem that affected SwiGLU)
**Case for failure:** The 37% subadditive discount from H10 suggests that even orthogonal components interact through the residual stream gradient coupling. The additive sum of H4 (-0.004) + H14 (-0.003) = -0.007. At 37% discount, expected delta = -0.0044. But the discount could be worse if PD's derivative signal (x - x_prev) is noisier with lower-rank initial representations: the velocity between layers 0 and 1 may be less informative when x0 starts from a bottlenecked embedding. Still, "less informative velocity" is a much weaker degradation mechanism than "gating mechanism starved of features" (the SwiGLU-factored failure mode).
**Feasibility pre-flight:** VRAM: H4 alone used 69.3 GB. Factored embeddings are memory-neutral (save 19M params from embedding, add tiny projection). Expected ~69-70 GB. torch.compile: both components individually compile cleanly. No new scalar params beyond deriv_lambdas (already in H4). The current codebase already has factored embeddings; the changes are: (a) add PD derivative terms (deriv_lambdas, x_prev tracking), (b) ensure MLP stays as ReluSquared (no SwiGLU), (c) keep factored embeddings as-is.
**Implementation sketch:** (1) Start from current codebase (which has factored embeddings). (2) Add deriv_lambdas = nn.Parameter(torch.zeros(n_layer)) as in H4. (3) In forward loop, before each block: compute velocity = x - x_prev; x = resid_lambdas[i] * x + x0_lambdas[i] * x0 + deriv_lambdas[i] * velocity; x_prev = x (before block). (4) MLP stays as ReluSquared (c_fc -> relu().square() -> c_proj). (5) Add deriv_lambdas to optimizer with lr=scalar_lr*0.01, same as H4. (6) Update parameter assertion.

**Information gain analysis:**
- P(success): 0.45
- If CONFIRMED, I learn: The negative SwiGLU-factored interaction is SwiGLU-specific (gated activations need high-rank input). PD+factored is a viable stacking pair. This opens the question of whether PD+factored+some-other-activation (not SwiGLU) could beat H10.
- If REFUTED, I learn: Factored embeddings are fundamentally incompatible with stacking in general, not just with SwiGLU. The bottleneck degrades some shared property (perhaps gradient flow quality through the projection) that all stacked configurations depend on.
- Expected information gain: high -- this is the key disambiguation experiment identified by the Evaluator. The result determines whether the stacking ceiling is SwiGLU-specific or general.

**Evaluator P(success):** 0.35
**Belief divergence:** 0.10

**Evaluator notes:** APPROVED. This is the only experiment that should be run with our final engineering run. Reasons: (1) Standing authorized follow-up from cycle 5 -- the Evaluator explicitly identified this as the key disambiguation experiment. (2) Both components (H4 PD residual, H14 factored embeddings) are individually CONFIRMED. (3) The mechanistic question is precise and the outcome is interpretable regardless of direction. (4) Implementation is straightforward -- both components have been implemented before in isolation. (5) VRAM is well within budget (~69-70 GB). (6) No torch.compile risk -- both components compiled cleanly individually. My P(success)=0.35 is lower than the Researcher's 0.45 because I am skeptical about the 37% subadditive discount assumption -- the PD velocity signal (x - x_prev) may be degraded by the lower-rank initial representation from factored embeddings, even though PD does not explicitly depend on input rank. The velocity at layer 0-1 is computed from the bottlenecked embedding, which may produce noisier velocity estimates. However, even if the result is INCONCLUSIVE or REFUTED, the information gain is high because it closes the open question about stacking compatibility.

Note on success criteria: The contract uses delta < -0.003 relative to the original baseline (0.959340). This is appropriate -- PD alone achieved -0.004 and factored alone achieved -0.003, so a stack that clears -0.003 from baseline is a minimal demonstration of viability. However, the truly informative comparison is against H4 alone (0.955143): if PD+factored achieves val_bpb < 0.955, factored embeddings provide additive benefit on top of PD without the SwiGLU-factored degradation.

---

## Hypothesis 23: SwiGLU MLP with Intra-Block Residual Shortcut on H10 Baseline | Status: REJECTED

### Sprint Contract

**Intervention:** Add a learnable residual shortcut within the MLP block, connecting the MLP input directly to the MLP output via a scalar gate. Specifically: MLP output = mlp_shortcut_lambda * x_input + (1 - mlp_shortcut_lambda) * mlp(x_input), where mlp_shortcut_lambda is a learnable per-layer scalar initialized at 0 (pure MLP at init, identity shortcut can emerge during training). This gives each layer the option to reduce MLP contribution if the attention output alone is sufficient, without adding parameters to the MLP itself. Applied on top of H10 (SwiGLU+PD) baseline.
**Subsystem:** MLP/activation-fn
**Papers:** [Domain A] "Simplifying Transformer Blocks" (arxiv 2311.01906, ICLR 2024) -- demonstrates that skip connections within MLP sublayers affect training dynamics. Removing MLP skip connections degrades per-update training speed, suggesting the identity path through the MLP carries useful gradient signal. Our intervention adds a learnable gate to control how much identity signal passes through, rather than removing the skip entirely. x [Domain B] Adaptive gain control in auditory neuroscience (Rabinowitz et al. 2011, "Contrast Gain Control in Auditory Cortex," Neuron) -- auditory neurons adapt their gain to the local contrast of the stimulus: in high-contrast environments, gain decreases (compressing dynamic range); in low-contrast environments, gain increases (amplifying weak signals). The MLP shortcut lambda acts analogously: when the MLP's contribution is large relative to the input (high contrast), the model can learn to reduce the MLP gain (increase shortcut). When the MLP contribution is small (low contrast), the model preserves it. This per-layer adaptive gain control allows different layers to operate at different effective MLP depths.
**Closest prior art:** PaLM uses parallel attention+MLP (both receive the same input), which is structurally different. The "Simplifying Transformer Blocks" paper examines removing skips entirely. Our approach is the converse: adding a tunable skip that starts at zero (pure MLP) and can learn to let identity through. The v6 finding that "learnable scaling with identity/near-passthrough init" is the winning pattern supports this initialization strategy.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to H10 baseline of 0.953648, i.e., val_bpb < 0.950648)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.001
**Predicted confidence:** low
**P(regression):** 0.10 (the init-at-0 means the model starts as pure MLP, identical to baseline behavior; regression requires the optimizer to actively learn harmful shortcut values)
**Case for failure:** The SwiGLU MLP already has a gating mechanism (c_gate) that controls feature selection. Adding a scalar shortcut on top of this is a coarser-grained version of the same idea -- skip the MLP entirely vs selectively gate features. The scalar shortcut may be redundant with SwiGLU's gate. At depth=10, every layer's MLP probably needs to contribute meaningfully (there are only 10 layers of capacity). The shortcut lambda may stay near 0 throughout training, producing a null effect. The 10 new scalar parameters (one per layer) add negligible compute but interact with the gradient flow through the MLP output projection (c_proj) -- the gradient to c_proj is scaled by (1 - mlp_shortcut_lambda), which could destabilize Muon's Newton-Schulz if lambda drifts significantly from 0.
**Feasibility pre-flight:** 10 scalar parameters (negligible). VRAM: identical to H10 baseline (~72.3 GB). torch.compile: a scalar multiply and add is trivially compilable. No graph changes. No new weight matrices. This is the lightest possible intervention.
**Implementation sketch:** (1) Add mlp_shortcut_lambda = nn.Parameter(torch.zeros(n_layer)) to GPT. (2) In Block.forward: mlp_input = norm(x); mlp_out = self.mlp(mlp_input); x = x + (1 - self.mlp_lambda) * mlp_out + self.mlp_lambda * mlp_input, where self.mlp_lambda is passed from the parent. (3) Actually simpler: x = x + mlp_out + self.mlp_lambda * (mlp_input - mlp_out). Init at 0 gives x = x + mlp_out (standard). (4) Train mlp_shortcut_lambda with AdamW at scalar_lr, betas=(0.8, 0.95). (5) Build on top of H10 (SwiGLU+PD) codebase.

**Information gain analysis:**
- P(success): 0.15
- If CONFIRMED, I learn: Per-layer adaptive MLP contribution strength is beneficial -- some layers benefit from reduced MLP influence. This would suggest the model has layers where attention alone is nearly sufficient (and the MLP adds noise).
- If REFUTED, I learn: At depth=10 with SwiGLU, every layer's MLP contribution is needed at full strength. The SwiGLU gating mechanism already provides sufficient adaptive capacity within the MLP.
- Expected information gain: medium -- the outcome primarily tells us whether MLP contribution is uniform across layers. Success would be surprising (P=0.15) and informative; failure would be unsurprising and mildly informative (confirming SwiGLU's internal gating is sufficient).

**Evaluator P(success):** 0.08
**Belief divergence:** 0.07

**Evaluator notes:** REJECTED. The Researcher's own predicted delta (-0.001) is far below the -0.003 threshold, and P(success)=0.15 is low. With 1 run remaining, this is not a credible candidate. The intervention is conceptually redundant with SwiGLU's internal gating: SwiGLU already performs per-feature adaptive gain control via c_gate. Adding a coarser scalar gate on top of a fine-grained gate is unlikely to help. The auditory neuroscience analogy is a stretch -- contrast gain control operates on continuous stimulus magnitudes, not on gated MLP outputs.

---

## Hypothesis 24: Factored Embeddings + PD + Learnable RoPE Frequencies (Triple Novel Stack) | Status: REJECTED

### Sprint Contract

**Intervention:** Combine three confirmed/untested interventions: (1) factored embeddings (H14, 256-dim, CONFIRMED), (2) PD residual scaling (H4, CONFIRMED), and (3) learnable per-head RoPE frequencies (H20, PROPOSED/untested). This uses baseline ReluSquared activation (no SwiGLU, to avoid the known negative SwiGLU-factored interaction). The rationale: factored embeddings and PD are individually confirmed; learnable RoPE frequencies have not been tested at all. If H22 (PD+factored) confirms positive stacking, adding learnable frequencies tests whether three orthogonal subsystems (embeddings, residuals, positional) compose better than the SwiGLU+PD pair (H10).
**Subsystem:** stacking/activation+residual + positional
**Papers:** Same as H20 (learnable RoPE, Kuramoto) + H22 (PD, rate-distortion). The novel contribution is the stacking hypothesis: three interventions targeting fully orthogonal subsystems (positional, residual dynamics, embedding compression) should exhibit lower subadditive discount than H10 (which targets MLP activation + residual dynamics, both of which interact through MLP gradient flow).
**Closest prior art:** No prior triple-stack has succeeded. H15 (SwiGLU+PD+factored) failed due to the SwiGLU-factored negative interaction. This stack deliberately avoids SwiGLU. The combination of positional + residual + embedding modifications is genuinely novel.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to original baseline 0.959340, i.e., val_bpb < 0.956340)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.004
**Predicted confidence:** low
**P(regression):** 0.10 (all components use identity/zero initialization; the worst case is null effect from learnable frequencies, leaving PD+factored to carry the result)
**Case for failure:** Triple stacking introduces three simultaneous changes, making failure diagnosis difficult. If the result is negative, we cannot determine which component or interaction caused it. The learnable RoPE frequencies are untested and may interact poorly with PD's velocity signal (the velocity computation x - x_prev may be affected by changing positional encoding). At 1800 steps, the model may not have enough training budget to simultaneously optimize frequency parameters, derivative coefficients, and the embedding projection. The combined VRAM of PD (69.3 GB baseline) + per-head frequencies (negligible) + factored embeddings (memory neutral) should be ~69-70 GB, but torch.compile with three modifications may increase activation memory.
**Feasibility pre-flight:** VRAM: ~69-70 GB (same as H22 estimate). Params: 10 deriv_lambdas + 3200 inv_freq params + wte_proj (256x640). All are lightweight. torch.compile: the main risk is the per-head cos/sin computation requiring dynamic shapes. If H20's implementation computes cos/sin from learnable inv_freq each forward pass, this adds a small overhead but should compile.
**Implementation sketch:** Combine H22's implementation (PD + factored embeddings on ReluSquared) with H20's learnable frequency modification. Specifically: (1) factored embeddings (already in codebase), (2) PD derivative terms (deriv_lambdas, x_prev tracking), (3) learnable inv_freq parameter per head per layer, (4) ReluSquared MLP (no SwiGLU).

**Information gain analysis:**
- P(success): 0.25
- If CONFIRMED, I learn: Three orthogonal subsystems compose positively when none involves gated activations. The subadditive discount is lower for truly orthogonal interventions (positional x residual x embedding) than for partly coupled ones (MLP activation x residual, as in H10). This would establish a new best configuration that avoids SwiGLU entirely.
- If REFUTED, I learn: Triple stacking is fundamentally difficult at this training budget, regardless of orthogonality. The 37% discount from H10 may be a floor, not a ceiling, for stacking interactions. Alternatively, one of the three components (likely learnable frequencies, the untested one) is actively harmful.
- Expected information gain: medium -- this is a high-risk, high-reward experiment. If it works, it opens a new frontier. If it fails, the failure is hard to diagnose (which component or interaction caused it?). This is why it is ranked below H22 and H20, which test components individually or in known-good pairs.

**Evaluator P(success):** 0.10
**Belief divergence:** 0.15

**Evaluator notes:** REJECTED. This hypothesis is contingent on H22 confirming and H20 being tested -- neither has happened. Running an untested component (learnable RoPE) inside a triple stack with the final engineering run would produce an uninterpretable result regardless of outcome. If the triple stack fails, we cannot distinguish between: (a) learnable RoPE is harmful, (b) PD+factored does not stack, or (c) a three-way interaction effect. The Researcher acknowledges this: "the failure is hard to diagnose." With 1 run left, interpretability of the result is paramount. H22 alone answers a clean, binary question. This experiment does not.

---

# Cycle 7 Hypotheses

## Hypothesis 25: Learnable Per-Head RoPE Frequency Offsets on H10 Baseline | Status: APPROVED

### Sprint Contract

**Intervention:** Add learnable per-head scalar offsets to the RoPE base frequency, allowing each of the 5 attention heads to specialize on different positional scales. Instead of making all 64 frequency dimensions learnable (as in rejected H20), this uses a single learnable log-base offset per head per layer (50 scalars total), computing inv_freq_head = 1 / (base * exp(offset))^(2i/head_dim). The frequencies are recomputed each forward pass from 50 scalars. Applied on top of H10 (SwiGLU+PD) baseline.
**Subsystem:** positional
**Papers:** [Domain A] "Sensitivity-Positional Co-Localization in GQA Transformers" (arxiv 2604.07766) -- introduces GARFA (GQA-Aware RoPE Frequency Adaptation) which attaches learnable per-KV-head scalar multipliers to RoPE frequencies. Key finding: positional sensitivity concentrates in early/mid layers, anti-localized from task-sensitive layers. The intervention uses per-head frequency multipliers (not per-dimension), showing that coarse frequency adaptation provides meaningful gains. x [Domain B] Multirate filter bank design in signal processing (Vaidyanathan 1993, "Multiband Signal Processing") -- in telecommunications, dyadic filter banks decompose signals into frequency subbands using different sampling rates per band. The key principle is that different frequency bands carry different information (speech energy concentrates in low bands, consonant discrimination in high bands), and optimal processing requires heterogeneous frequency allocation across parallel channels. In multi-head attention, each head is a parallel processing channel. Fixed RoPE gives all heads identical frequency response. A per-head base offset creates a filter bank where each head resolves position at a different scale: low-base heads attend to local patterns (high-frequency positional sensitivity), high-base heads attend to global patterns (low-frequency sensitivity), analogous to the octave-band decomposition in audio processing.
**Closest prior art:** H20 (rejected, never tested) proposed per-head per-dimension learnable frequencies (3200 params). This is drastically simpler: 50 scalars that shift the entire frequency schedule per head, preserving the geometric spacing within each head. GARFA (2604.07766) applies per-KV-head multipliers at 8B scale with LoRA; we apply per-head base offsets at 84M scale during full training. The key difference from H20: we change the base (1 scalar per head-layer) rather than all 64 frequencies, reducing the parameter count by 64x and eliminating the torch.compile risk of per-dimension dynamic computation.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to H10 baseline of 0.953648, i.e., val_bpb < 0.950648)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**Case for failure:** At 2048 context and 1800 steps, positional encoding may not be a binding bottleneck. The standard RoPE base=10000 is well-optimized. With only 5 heads, the capacity for meaningful frequency specialization is limited. The v5-v6 prior notes "positional encoding changes minimal effect at 2048 context." The 50 scalars must be trained with AdamW (not Muon -- they are 1D), and the learning rate for frequency offsets is unclear: too fast causes RoPE frequency collapse (all heads converge to same base), too slow produces null effect. The recomputation of cos/sin each forward pass from learnable parameters may cause torch.compile graph breaks if not handled carefully. The initialization at offset=0 means the model starts at standard RoPE, but the gradient signal for learning optimal offsets may be too weak at 1800 steps.
**Feasibility pre-flight:** 50 scalar parameters (negligible). VRAM: identical to H10 (~72.3 GB). The cos/sin recomputation per forward pass adds negligible compute (50 matmuls of shape [T, 64]). torch.compile risk: the key concern is that the cos/sin computation from learnable parameters creates a dynamic graph. Safest approach: compute all 50 cos/sin tensors in a pre-forward hook outside the compiled region, pass as a list indexed by (layer, head). Alternative: since offsets are scalars, the cos/sin for each head can be computed as cos(t * inv_freq * exp(-offset)), which is a simple element-wise scaling of the base cos/sin -- this may be compilable.
**Implementation sketch:** (1) Add rope_offsets = nn.Parameter(torch.zeros(n_layer, n_head)) to GPT. (2) Remove precomputed cos/sin buffers. (3) In forward(), for each layer i, compute per-head cos/sin: for each head h, base_h = 10000 * exp(rope_offsets[i, h]); inv_freq_h = 1/(base_h^(2j/head_dim)); freqs_h = t_outer(positions, inv_freq_h); cos_h, sin_h = freqs_h.cos(), freqs_h.sin(). (4) Reshape to pass per-head cos/sin to apply_rotary_emb (need to modify apply_rotary_emb to accept per-head cos/sin, or expand before the call). (5) Add rope_offsets to optimizer as AdamW group with lr=scalar_lr*0.1, betas=(0.8, 0.95), weight_decay=0. (6) Build on H10 (SwiGLU+PD) codebase.

**Information gain analysis:**
- P(success): 0.20
- If CONFIRMED, I learn: The positional subsystem is a genuine bottleneck at this scale. Per-head frequency specialization provides signal that standard uniform-base RoPE misses. This opens positional encoding as a viable improvement axis and would be the first positive result in the positional subsystem.
- If REFUTED, I learn: The standard RoPE frequency schedule is near-optimal at 2048 context / 1800 steps. The positional subsystem is not exploitable at this scale, consistent with the v5-v6 prior. This closes the positional subsystem for this programme.
- Expected information gain: high -- the positional subsystem has 0 experiments. ANY result updates our model. The P(success)=0.20 means both outcomes are plausible and informative, and neither would be surprising enough to suggest a bug. This is the highest-information-gain hypothesis because it probes an entirely untested subsystem.

**Evaluator P(success):** 0.12
**Belief divergence:** 0.08

**Evaluator notes:** APPROVED. The positional subsystem has 0 experiments, making this the highest-information-gain hypothesis available. The simplified approach (50 scalar base offsets instead of H20's 3200 per-dimension frequencies) substantially reduces torch.compile risk. The GARFA paper provides credible evidence that per-head frequency adaptation helps. My main concerns: (1) The Researcher's own predicted delta (-0.002) is below the -0.003 threshold, signaling low confidence in crossing the bar. (2) The contract references "5 attention heads" and "50 scalars" but the current config has n_head=6 and n_layer=12, which would give 72 scalars -- the Researcher must adjust for the actual config. (3) The torch.compile risk from recomputing cos/sin from learnable parameters each forward pass is manageable but real -- the safest path is to compute these outside the compiled region. (4) At 2048 context and 1800 steps, the model may not encounter enough positional diversity to learn useful frequency offsets. Despite these concerns, the information gain from probing a completely untested subsystem justifies the run. The P(success)=0.12 reflects that most positional encoding interventions at short context and short training have negligible effect, but I could be wrong (as I was for H14). The belief divergence (0.08) is low, suggesting genuine agreement that this is a long shot with high information value.

---

## Hypothesis 26: Value Residual Mixing with Decaying Lambda on H10 Baseline | Status: APPROVED

### Sprint Contract

**Intervention:** Modify the existing value embedding residual in the attention mechanism. Currently, the code adds gated value embeddings as v = v + gate * ve. Change this to use a depth-decaying lambda schedule: v = (1 - alpha_l) * v_proj + alpha_l * V1, where V1 is the first layer's value projection (not the token-level value embedding), and alpha_l decays linearly from 0.5 at layer 0 to 0.0 at the final layer. This replaces the per-layer value embedding tables (which are expensive: ~5 separate embedding tables) with a single shared first-layer V1 that is mixed into all subsequent layers with decaying weight. This reduces value embedding parameters by ~80% while potentially improving information flow through the attention value pathway. Applied on top of H10 (SwiGLU+PD) baseline.
**Subsystem:** attention/structural
**Papers:** [Domain A] "Value Residual Learning" (arxiv 2410.17897, ResFormer) -- demonstrates that mixing the first layer's value embedding into all subsequent layers via V_n = lambda_1 * V_1 + lambda_2 * H_{n-1} * W_v improves validation loss by equivalent of 16% fewer parameters. The key insight: standard hidden-state residuals fail to preserve token-level information in deeper layers, and explicit value-pathway residuals compensate. The optimal mixing uses a decaying schedule where early layers get more V1 signal and late layers get less. x [Domain B] Eligibility traces in neuroscience (Gerstner et al. 2018, "Eligibility Traces and Plasticity on Behavioral Time Scales," Frontiers in Neural Circuits) -- synaptic eligibility traces maintain a decaying memory of recent presynaptic activity at each synapse. When a reward signal arrives, the trace determines which synapses are strengthened (credit assignment). The trace decays exponentially with time, ensuring recent activity has stronger influence. The analogy: the first-layer value projection V1 captures the "initial synaptic response" to each token. Subsequent attention layers receive a decaying trace of this initial response, ensuring the raw token-level signal is not overwritten by increasingly abstract representations. The decay schedule mirrors the exponential decay of eligibility traces.
**Closest prior art:** The current codebase already uses value embeddings (separate embedding tables per alternating layer). ResFormer uses value residuals from the first layer's value projection. Our intervention replaces the multiple embedding tables with a single first-layer V1 residual, which is closer to ResFormer's SVFormer variant (shared V across layers). The key difference: we use a fixed linear decay schedule rather than learnable lambdas (to avoid scalar parameter interactions with Muon) and apply this on top of H10 (SwiGLU+PD), which has not been tested with value residuals before.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to H10 baseline of 0.953648, i.e., val_bpb < 0.950648)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**Case for failure:** The existing value embedding mechanism already addresses the value-pathway information flow problem. Replacing per-layer value embeddings with a shared first-layer V1 may lose the token-specific gating that the current ve_gate provides. The linear decay schedule (alpha_l from 0.5 to 0.0) may be suboptimal -- the right decay rate is unknown. At depth=10, the value pathway may not suffer from the information loss that ResFormer addresses (which was demonstrated at 24+ layers). The parameter reduction (~80% fewer value embedding params) could hurt if those parameters were contributing useful capacity. The attention subsystem has a poor track record (2 failures, 0 confirmations), though this is a structural change (value pathway) rather than a logit modification.
**Feasibility pre-flight:** Removes ~4 of 5 value embedding tables (~2.6M params per table, saving ~10M params). Adds no new parameters (the decay schedule is fixed). VRAM: should be substantially lower than current (fewer embedding tables to store). torch.compile: the V1 = c_v(x) at layer 0 must be stored and passed to all subsequent layers. This is a single tensor of shape (B, T, n_kv_head * head_dim) = (128, 2048, 640) = ~167M elements = ~335 MB. No graph changes beyond passing one extra tensor. Very compile-safe.
**Implementation sketch:** (1) Remove all value_embeds except layer 0. (2) In CausalSelfAttention.forward, accept v1_ref as an additional argument. (3) At layer 0: compute v normally AND store V1 = v (after c_v projection). (4) At layer i > 0: alpha = max(0, 0.5 * (1 - i / (n_layer - 1))); v = (1 - alpha) * v + alpha * v1_ref. (5) Remove ve_gate (no longer needed -- the mixing is via a fixed schedule). (6) In GPT.forward, after the first block, pass v1 to all subsequent blocks. (7) Adjust parameter assertion and optimizer setup (fewer embedding params).

**Information gain analysis:**
- P(success): 0.25
- If CONFIRMED, I learn: Value-pathway residuals provide useful signal at depth=10 when formulated as first-layer-V1 mixing. The per-layer value embeddings were suboptimal -- a shared V1 with decay is better. This opens a new dimension for attention improvement (value pathway) that is orthogonal to the blocked logit modifications.
- If REFUTED, I learn: At depth=10, the value pathway does not suffer from information degradation. The existing per-layer value embeddings are already optimal. The attention subsystem is resistant to structural modifications as well as logit modifications. If this is the third attention failure, the attention/all subsystem should be BLOCKED.
- Expected information gain: high -- this tests a fundamentally different attention axis (value pathway vs logit distribution). If it fails, combined with H8 (logit scaling) and H13 (logit norm), it establishes that the entire attention subsystem is rigid at depth=10. If it succeeds, it opens value-pathway modifications as a viable direction. Both outcomes are strongly model-updating.

**Evaluator P(success):** 0.15
**Belief divergence:** 0.10

**Evaluator notes:** APPROVED with reservations. This is the most mechanistically interesting hypothesis in the batch because it tests a genuinely different attention axis (value pathway) versus the two prior attention failures (both logit modifications). The ResFormer paper provides credible evidence. However, I have serious concerns: (1) The current baseline ALREADY has a sophisticated value embedding mechanism (per-layer value embedding tables with learned gates). This proposal REPLACES that mechanism with something simpler (shared V1 with fixed linear decay). The burden of proof is on the new mechanism: it must beat not just a no-value-embedding baseline but the existing tuned value embedding system. The contract does not clearly acknowledge this -- the implementation removes the per-layer value embedding tables and their gates. (2) The fixed linear decay schedule (alpha from 0.5 to 0.0) is arbitrary and not learnable. The existing gate mechanism is learnable and more expressive. Replacing learnable with fixed is a downgrade in flexibility. (3) If this fails, it will be the 3rd attention failure (after H8 and H13), triggering BLOCKED for attention/all. The Researcher should be aware of this consequence. (4) The parameter REDUCTION (~10M fewer value embedding params) could hurt -- those embeddings may be providing useful capacity. Despite these concerns, I approve because the value pathway axis is genuinely orthogonal to prior attention failures, the information gain is high regardless of outcome, and the implementation is clean and compile-safe. My P(success)=0.15 is lower than the Researcher's 0.25 because I weight the risk of downgrading from the existing learned value embedding mechanism.

---

## Hypothesis 27: SwiGLU with Hidden Dim 1920 on H10 Baseline | Status: APPROVED

### Sprint Contract

**Intervention:** Increase SwiGLU hidden_dim from 1792 (H6/H10) to 1920 (3x model_dim, a clean multiple of 128), adding ~1.6M parameters. The H6 finding that increasing hidden_dim from 1664 to 1792 (7.7% wider) doubled the delta suggests the model is still in the width-superlinear regime for gated activations. A further increase to 1920 (7.1% wider than 1792) tests whether this superlinear scaling continues. Applied on top of H10 (SwiGLU+PD) baseline, measuring delta against H10.
**Subsystem:** activation/gating
**Papers:** [Domain A] Shazeer 2020, "GLU Variants Improve Transformer" (arXiv:2002.05202) -- establishes that gated linear units with wider hidden dimensions consistently improve over narrower variants. The mechanism: wider hidden dims give the gate projection (c_gate) more capacity to learn discriminative feature selection patterns. The findings establish that GLU variants benefit from width more than ungated activations because the gate's selectivity improves with more features to select from. x [Domain B] Channel capacity theorem (Shannon 1948, "A Mathematical Theory of Communication") -- the capacity of a noisy channel increases logarithmically with bandwidth (C = B * log2(1 + S/N)). In a transformer MLP, the hidden dimension is analogous to bandwidth: it determines how many independent "channels" of feature processing are available. The gate (c_gate in SwiGLU) acts as a signal-to-noise discriminator, suppressing noisy channels. With more bandwidth (wider hidden dim), the gate has more channels to evaluate, and the effective channel capacity (useful information throughput) increases. However, Shannon's theorem predicts diminishing returns: each additional unit of bandwidth adds less capacity. The superlinear regime observed between 1664 and 1792 may reflect crossing a critical bandwidth threshold below which the gate cannot discriminate effectively.
**Closest prior art:** H6 (SwiGLU, hidden_dim=1792, delta=-0.004964) is the direct predecessor. H1 (SwiGLU, hidden_dim=1664, delta=-0.002460, INCONCLUSIVE) showed the width sensitivity. No test at hidden_dim=1920 has been done. The parameter increase (~1.6M, ~2% of total) is modest. The finding in findings.md that "width bottleneck for gated activations is severe at small scale" directly motivates this extension.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to H10 baseline of 0.953648, i.e., val_bpb < 0.950648)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.002
**Predicted confidence:** low
**Case for failure:** The superlinear width scaling observed between 1664 and 1792 may have already plateaued. Shannon's channel capacity theorem predicts logarithmic (sublinear) scaling beyond a threshold. At hidden_dim=1920, the model gains 128 more features (7.1% increase), but this may be past the knee of the width-benefit curve. The additional ~1.6M parameters consume training compute that could be better allocated elsewhere. VRAM increases slightly (~0.5-1 GB) from the wider activation tensors. The parameter count assertion in setup_optimizer must accommodate the new dimensions. Wall-clock may increase slightly from larger matmuls.
**Feasibility pre-flight:** Adding ~1.6M params. VRAM: H10 used ~72.3 GB; wider SwiGLU adds activation memory proportional to (1920-1792)*B*T = 128*128*2048 ~= 33M elements in bf16 ~= 66 MB. So ~72.4 GB total, well within budget. torch.compile: no new ops, just wider matmuls. Perfectly compile-safe. Wall-clock: the larger matmuls add ~2-3% to MLP compute, staying well within 1200s.
**Implementation sketch:** (1) Start from H10 codebase (SwiGLU+PD). (2) Change hidden_dim from 1792 to 1920. This affects c_fc, c_gate, and c_proj dimensions in the SwiGLU MLP. (3) No other changes needed. (4) Verify parameter assertion. This is the simplest possible intervention.

**Information gain analysis:**
- P(success): 0.30
- If CONFIRMED, I learn: The width-superlinear regime for SwiGLU extends beyond 1792. The model's gating mechanism continues to benefit from additional features at 1920 dims. This maps a point on the width-performance curve and suggests even wider dims (2048) could help further.
- If REFUTED, I learn: The superlinear regime ended at or near 1792. The width-performance curve has flattened, and further width increases provide only marginal (sub-threshold) benefit. This closes the activation/gating width exploration axis.
- Expected information gain: medium -- this is a parameter sweep point, not a mechanistic test. The outcome maps a curve rather than testing a hypothesis about a mechanism. Success is moderately informative (confirms continued scaling); failure is moderately informative (maps the plateau). Neither outcome is highly surprising.

**Evaluator P(success):** 0.20
**Belief divergence:** 0.10

**Evaluator notes:** APPROVED. This is the simplest and lowest-risk hypothesis in the batch: a one-number change to hidden_dim. The implementation is trivial (change 1792 to 1920), compile-safe, VRAM-safe. The question is whether the width-superlinear regime extends past 1792. I am more sceptical than the Researcher (P=0.20 vs 0.30) because: (1) The H1->H6 jump (1664->1792, delta doubled) likely reflected crossing a critical threshold, not a sustained superlinear trend. Shannon's channel capacity theorem predicts logarithmic (sublinear) returns beyond the threshold. (2) The 1920 dim adds ~1.6M params (~2% of model), and at 1800 training steps these extra params may not be fully utilized. (3) The Researcher's own predicted delta (-0.002) is below the -0.003 threshold. However, the experiment is cheap, safe, and maps a genuinely uncertain point on the width-performance curve. Even an INCONCLUSIVE result narrows the viable parameter space. NOTE: The contract references dimensions consistent with n_embd=640 (hidden_dim=1792, 1920). If the actual baseline config is n_embd=768, these dimensions need to be recalculated. The Researcher must verify the correct config before implementing.

---

## Hypothesis 28: PD Residual with Layer-Dependent Derivative Gain on H10 Baseline | Status: WITHDRAWN (self-critique: too weak, low information gain)

### Sprint Contract

**Intervention:** Replace the single per-layer deriv_lambda scalar (H4/H10) with a structured layer-dependent initialization: deriv_lambdas are initialized to a linearly increasing schedule (0.0 at layer 0, 0.1 at layer 9) rather than all zeros. The motivation is that H4's derivative term provides anticipatory correction, and this correction should be stronger in later layers where the residual stream trajectory is smoother (more correlated between consecutive layers) and the velocity signal is more informative. The key change is ONLY the initialization -- the parameters remain learnable. Applied on top of H10 (SwiGLU+PD) baseline.
**Subsystem:** residuals/derivative-scaling
**Papers:** [Domain A] "A unified high-resolution ODE framework for first-order methods" (arxiv 2603.07075) -- demonstrates that the optimal damping coefficient in heavy-ball-type ODEs is NOT constant but increases with iteration count. The framework shows that second-order ODE discretizations with time-varying coefficients converge faster than fixed-coefficient variants. The key insight: the optimal derivative (momentum) coefficient grows as the system approaches the optimum because the trajectory becomes smoother and the velocity signal becomes more reliable. x [Domain B] Variable-gain PD control in robotics (Slotine & Li 1991, "Applied Nonlinear Control") -- in robotic trajectory tracking, the derivative gain K_d is typically set higher for slow, smooth motions and lower for fast, jerky motions. The principle: when the system dynamics are smooth (low jerk), the derivative signal is clean and high gain exploits it. When dynamics are noisy (high jerk), the derivative signal is noisy and high gain amplifies noise. In a 10-layer transformer, early layers produce large, diverse updates (high "jerk" -- the residual stream changes character rapidly as it moves from raw embeddings to abstract representations). Later layers produce smaller, more incremental updates (low "jerk" -- the representation is nearly converged). Therefore, derivative gain should increase with depth.
**Closest prior art:** H4 initialized all deriv_lambdas at 0 and let them learn freely. This is an initialization change only -- we provide a structured prior that encodes the "increasing gain with depth" principle from control theory. The parameters still learn freely with the same lr=scalar_lr*0.01. If the initialization prior is wrong, the parameters should converge to the same values as H4 (approximately). If the prior is correct, the parameters start closer to their optimal values and the model benefits from better early-training dynamics.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to H10 baseline of 0.953648, i.e., val_bpb < 0.950648)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.001
**Predicted confidence:** low
**Case for failure:** H4's zero initialization already works well (delta=-0.004). The learnable deriv_lambdas may converge to the same values regardless of initialization at 1800 steps, making the initialization change irrelevant. With lr=scalar_lr*0.01 = 0.005 for 1800 steps, the parameters have ample time to move from any reasonable initialization. The linearly increasing prior may be wrong -- perhaps derivative gain should be highest in the middle layers or decrease at the end. At depth=10, the "smooth later layers" assumption may not hold: the warmdown layers (layers 7-9 are in the "S" pattern layers before the final "L" layer) might have different dynamics. The risk of regression is low (init at 0.1 for the deepest layer is a small perturbation) but the expected benefit is also small.
**Feasibility pre-flight:** Zero additional parameters. Zero VRAM change. Zero torch.compile risk. This is a one-line change to the initialization code. The simplest possible intervention.
**Implementation sketch:** (1) In GPT.init_weights(), change self.deriv_lambdas.fill_(0.0) to: for i in range(n_layer): self.deriv_lambdas.data[i] = 0.1 * i / (n_layer - 1). (2) No other changes. Build on H10 (SwiGLU+PD) codebase.

**Information gain analysis:**
- P(success): 0.15
- If CONFIRMED, I learn: The initialization of derivative gains matters at 1800 steps -- the learning rate is not fast enough for the scalars to find the optimal configuration from zero init alone. Layer-dependent derivative gain is the right structure. This opens the door to more structured initialization of PD parameters.
- If REFUTED, I learn: The scalar learning rate (0.005) is sufficient for 1800 steps to find optimal deriv_lambdas regardless of initialization. The initialization does not matter for scalar parameters at this training budget. This narrows the residuals/derivative-scaling axis to mechanism changes (not initialization changes).
- Expected information gain: low -- the most likely outcome is INCONCLUSIVE (small positive or small negative delta that does not clear threshold). The intervention is too small to produce a clear signal. Ranked lowest of the 4 hypotheses.

---

## Hypothesis 29: Parallel Attention-MLP Block on H10 Baseline | Status: APPROVED

### Sprint Contract

**Intervention:** Change the block structure from sequential (attention then MLP) to parallel (attention and MLP receive the same input, outputs are summed). Specifically: instead of x = x + attn(norm(x)); x = x + mlp(norm(x)), use x = x + attn(norm(x)) + mlp(norm(x)). Both sublayers see the same normed input. This eliminates the serial dependency between attention and MLP within each block, meaning the MLP no longer depends on the attention output. Applied on top of H10 (SwiGLU+PD) baseline.
**Subsystem:** stacking/activation+residual
**Papers:** [Domain A] "Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design" (arxiv 2305.13297) -- compares parallel vs sequential attention+FFN blocks. Key finding: parallel design matches sequential quality at scale (used in PaLM 540B) but can underperform at smaller scales. The paper identifies that "the FFN learns to compensate for attention errors" in sequential blocks, and parallel blocks lose this error-correction property. The critical question is whether this trade-off differs when PD residual scaling provides an alternative error-correction mechanism. x [Domain B] Superposition principle in linear systems theory (control theory) -- for linear time-invariant systems, the response to a sum of inputs equals the sum of individual responses. When two subsystems (attention and MLP) are fed the same input in parallel, their outputs superpose in the residual stream. If the subsystems are approximately linear (which Pre-LN transformers approximate at init due to zero-init output projections), superposition holds exactly and parallel = sequential. The deviation from linearity during training determines whether parallel processing loses information. In control theory, parallel feedback loops are preferred over cascade loops when the individual loops have well-separated frequency responses (one loop handles low-frequency disturbances, the other handles high-frequency). Attention (global context mixing) and MLP (local feature transformation) have naturally separated "frequency responses," suggesting parallel processing preserves their complementary roles.
**Closest prior art:** PaLM (Chowdhery et al. 2022) uses parallel blocks at 540B scale. GPT-J uses parallel blocks at 6B. Both report negligible quality difference vs sequential at large scale. Neither tests at 84M with Muon optimizer and PD residual scaling. The critical novelty: PD residual scaling (the velocity term) may interact favorably with parallel blocks because the velocity signal captures the combined attention+MLP update direction, which is potentially more stable when both sublayers see the same input.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (relative to H10 baseline of 0.953648, i.e., val_bpb < 0.950648)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.001
**Predicted confidence:** low
**Case for failure:** At 84M scale (small), the sequential error-correction property (MLP compensates for attention errors) may be important. PaLM/GPT-J results showing quality parity are at much larger scale where individual layer errors are smaller relative to the residual stream. The parallel block processes attention and MLP independently, losing the implicit regularization of the sequential composition. The PD velocity term captures the combined update, but this does not compensate for the lost within-block error correction. The expected delta is near zero (quality-neutral) rather than positive, making confirmation unlikely. The main value of this experiment is to determine whether parallel blocks are COMPATIBLE with PD scaling (i.e., not harmful), opening the door to wall-clock savings from parallel execution.
**Feasibility pre-flight:** Zero additional parameters. VRAM: slightly higher because both norm(x) inputs must be computed before either sublayer runs (one extra norm call's activation memory, ~0.5 GB). torch.compile: the parallel structure changes the computation graph (both sublayers see norm(x) instead of attention seeing norm(x) and MLP seeing norm(x + attn_out)). This is a graph structure change that torch.compile should handle fine -- no new ops. Wall-clock: parallel blocks enable concurrent attention+MLP execution if the GPU has spare SMs, but in practice at batch_size=128 this is unlikely. The main wall-clock effect is removing one norm() call from the critical path.
**Implementation sketch:** (1) In Block.forward, change from: x = x + self.attn(norm(x), ...); x = x + self.mlp(norm(x)) to: nx = norm(x); x = x + self.attn(nx, ...) + self.mlp(nx). (2) This removes the sequential dependency. (3) No other changes. Build on H10 (SwiGLU+PD) codebase.

**Information gain analysis:**
- P(success): 0.20
- If CONFIRMED, I learn: Parallel blocks are not merely neutral but actively beneficial when combined with PD residual scaling. The PD velocity term interacts favorably with the parallel structure (perhaps because the velocity of a parallel update is more consistent/smooth than a sequential update). This would establish parallel blocks as a viable stacking partner for H10.
- If REFUTED, I learn: At 84M scale, sequential error correction (MLP compensating attention errors) is load-bearing. Parallel blocks are inappropriate at small scale regardless of PD scaling. This closes parallel block design as a viable direction for this programme.
- Expected information gain: medium-high -- the P(success)=0.20 means both outcomes are plausible. The interaction between parallel blocks and PD scaling has never been tested and the outcome is genuinely uncertain. Success would be moderately surprising; failure would be expected but informative about the scale-dependence of parallel block quality.

**Evaluator P(success):** 0.10
**Belief divergence:** 0.10

**Evaluator notes:** APPROVED. The implementation is the simplest in the batch (one structural change in Block.forward, no new parameters). Compile-safe, VRAM-neutral. However, I am more sceptical than the Researcher (P=0.10 vs 0.20) because: (1) The literature is clear that parallel blocks underperform sequential at small scale. PaLM/GPT-J results showing parity are at 6B-540B, three orders of magnitude larger than 84M. (2) At 84M with only 10-12 layers, the MLP's error-correction role (compensating attention errors in sequential mode) is likely more important than at scale because each layer carries proportionally more of the total computation. (3) The Researcher's own predicted delta (-0.001) is far below the -0.003 threshold, and the case for failure is more convincing than the case for success. (4) The superposition principle argument is weak: transformers are highly nonlinear, and the linear superposition assumption holds only approximately at initialization. Despite low P(success), the experiment is approved because it probes an untested structural variant with zero implementation risk. The PD velocity interaction with parallel blocks is a genuinely novel question. Even a null result (INCONCLUSIVE near zero) would be informative about whether parallel blocks are at least compatible with the H10 configuration.
