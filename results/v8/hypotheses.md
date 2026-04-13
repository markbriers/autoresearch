# Hypotheses

## Engineering Run Counter: 0/5 | Phase: RESEARCH

---

## Hypothesis H15: Dimension-Reduced SwiGLU (hidden_dim=1728) | Status: CONFIRMED

**Evaluator note (pre-run):** Approved. This is the encouraged follow-up to H10's resource-constrained refutation. VRAM estimate of 72.2 GB is plausible given the researcher validated against H10 actuals. KEY CAVEAT: this replaces confirmed ShrinkReLU (-0.006). The sprint contract measures delta against the original baseline (0.959552), but the actual bar to clear for net programme benefit is -0.006 (matching ShrinkReLU). A result of, say, -0.004 would be CONFIRMED per contract but represent a net regression. The Researcher should note that if SwiGLU-1728 confirms but underperforms ShrinkReLU, the programme should revert to ShrinkReLU. Recommended run order: run H16 first (stacks with ShrinkReLU, lower risk), then H15 (replaces ShrinkReLU, higher risk/reward).

### Sprint Contract

**Intervention:** Replace ReluSquared MLP with SwiGLU using hidden_dim=1728 (2.7x n_embd) instead of the full 4x n_embd, matching the current 8x n_embd^2 total MLP parameter budget (3 matrices at 1728 vs 2 matrices at 2560).

**Subsystem:** MLP/gated-activation

**Papers:** [Domain A] Shazeer 2020 "GLU Variants Improve Transformer" (arXiv:2002.05202) x [Domain B] Candes & Tao 2006 "Near-Optimal Signal Recovery from Random Projections" (IEEE TIT 52(12):5406-5425, compressed sensing / information theory)

**Closest prior art:** H10 tested SwiGLU at full 4x hidden_dim and achieved -0.014435 val_bpb but failed on VRAM (79.1 GB) and wall-clock (1261s). This is a dimension-reduced variant specifically designed to fit within resource constraints. LLaMA uses SwiGLU at 2/3 * 4 * n_embd = 2.67x n_embd, which is exactly the ratio used here. The key difference from H10: hidden_dim is 1728 (67.5% of 2560), reducing the extra activation tensor from 12.5 GB to 4.4 GB.

**Cross-domain mapping:** Candes-Tao showed that stable signal recovery is possible from far fewer measurements than the ambient dimension, provided the measurement matrix satisfies the Restricted Isometry Property (RIP). The insight applied here: SwiGLU's gating mechanism performs a learned structured projection where the gate selects which dimensions carry signal. With ReluSquared at hidden_dim=2560, the hard threshold wastes capacity on dimensions that are always zero (sparsity ~90%). SwiGLU at hidden_dim=1728 uses soft gating to concentrate signal into fewer dimensions more efficiently -- analogous to how compressed sensing achieves near-optimal recovery with fewer measurements by exploiting signal structure. The gating provides a structured measurement matrix that adapts to the input distribution, whereas ReluSquared's fixed threshold is a crude non-adaptive measurement.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.008
**Predicted confidence:** medium-high
**Case for failure:** (1) The reduced hidden dimension may lose more quality than the gating mechanism recovers -- SwiGLU's advantage comes partly from the larger effective capacity of the gating interaction, and at 1728 dims the gate-value product has 33% less capacity than the current 2560-dim ReluSquared. (2) Muon optimizes all matrix params the same way; SwiGLU has a gate projection and a value projection with different functional roles that might benefit from different LR/momentum, but we use a single Muon group per shape. Since c_fc and c_gate have identical shape (640x1728), they will be stacked and orthogonalized together, which might blur their distinct roles. (3) The ShrinkReLU activation (confirmed in H12) would be removed by this change since SwiGLU replaces the entire activation; if SwiGLU's benefit is smaller than expected, we lose ShrinkReLU's confirmed -0.006.

**Feasibility pre-flight:**
- Params: 3 matrices of 640x1728 = 3,317,760 per layer vs current 2 matrices of 640x2560 = 3,276,800. Delta: +40,960 per layer, +409,600 total (~0.4M). Negligible.
- VRAM: The critical cost is activation storage. SwiGLU stores 2 tensors at hidden_dim (gate output + value output) for backward, vs current 1 tensor at 4*n_embd. Delta: (2*1728 - 2560) * 128 * 2048 * 2 * 10 = 4.38 GB. Predicted VRAM: 67.8 + 4.4 = 72.2 GB. Validated against H10: the model predicts H10's 11.3 GB increase within 10% accuracy.
- Wall-clock: 3 matmuls at (640,1728) vs 2 matmuls at (640,2560). FLOP ratio: 3*1728 / (2*2560) = 1.01x (essentially equal). Predicted wall-clock: ~930s (within budget).
- torch.compile: SiLU is a native op, element-wise multiply is trivial. No new scalar params, no graph breaks. The c_gate matrix is a standard nn.Linear that torch.compile handles identically to c_fc.
- Muon compatibility: c_fc and c_gate have shape (1728, 640). They will be grouped and stacked together in Muon's optimizer. This is safe -- the Newton-Schulz orthogonalization operates on each matrix independently within the stack.

**Implementation sketch:**
- In MLP.__init__: replace c_fc with c_fc = nn.Linear(n_embd, 1728, bias=False) and add c_gate = nn.Linear(n_embd, 1728, bias=False). Keep c_proj = nn.Linear(1728, n_embd, bias=False).
- In MLP.forward: replace `x = self.c_fc(x); x = F.relu(x).square(); x = self.c_proj(x)` with `x = F.silu(self.c_gate(x)) * self.c_fc(x); x = self.c_proj(x)` (need to save input x before projections -- actually just reference the input arg directly).
- In init_weights: init c_gate same as c_fc (uniform [-s, s]). Init c_proj to zeros (same as current).
- c_gate and c_fc will be automatically collected by the Muon param grouping since they are in transformer.h and have shape (1728, 640). c_proj has shape (640, 1728) -- a different Muon group from c_fc/c_gate.
- Remove ShrinkReLU raw_tau parameter if present in the codebase (SwiGLU replaces the entire activation).

---

## Hypothesis H16: Learned RMSNorm Gain (Affine RMSNorm) | Status: REFUTED

**Evaluator note (pre-run):** Approved. Clean minimal-parametric-extension following the confirmed pattern (B1). Stacks with ShrinkReLU (does not replace it). Zero resource risk. The only concern is whether 13,440 per-dimension parameters converge meaningfully in ~5000 steps -- identity init ensures no harm if they do not. Implementation sketch is messy (three options discussed); the Researcher MUST use the preferred approach (gains stored inside Block, not ParameterList on GPT) and must update setup_optimizer assertions. Run this BEFORE H15 to preserve the ShrinkReLU baseline for comparison.

### Sprint Contract

**Intervention:** Add a learnable per-dimension gain vector (initialized to ones) to each RMSNorm application, transforming the parameter-free `rms_norm(x)` into `gamma * rms_norm(x)` where gamma is a learned vector of size n_embd per normalization site.

**Subsystem:** normalisation

**Papers:** [Domain A] Zhang & Sennrich 2019 "Root Mean Square Layer Normalization" (NeurIPS 2019, arXiv:1910.07467, introduced RMSNorm and showed learnable gain is beneficial) x [Domain B] Perez 1984 "Automatic Gain Control" (ch. 6 in Perez "Wireless Communications Design Handbook", telecommunications: AGC circuits maintain constant output level by continuously adjusting gain based on input signal strength, with per-channel gain adaptation in wideband receivers)

**Closest prior art:** The current codebase uses parameter-free RMSNorm (`F.rms_norm(x, (x.size(-1),))`). LLaMA, Gemma-2, and most modern LLMs use RMSNorm WITH learnable gain. This is a well-established technique that has NOT been tested in this codebase. The attn_temperature parameter already provides per-head gain on Q (a coarse version of this), but the pre-attention and pre-MLP norms have no learnable parameters. This hypothesis follows the confirmed "minimal parametric extension with identity init" pattern exactly.

**Cross-domain mapping:** In telecommunications, wideband AGC circuits apply per-subchannel gain adaptation to normalize signal levels across frequency bands while preserving the relative information content within each band. The analogy to RMSNorm gain is precise: RMSNorm normalizes the total signal energy (equivalent to AGC's level detection), and the learnable gain gamma selectively amplifies or attenuates individual feature dimensions (equivalent to per-subchannel gain). Without gain, RMSNorm treats all dimensions equally, forcing the subsequent linear layer to undo any dimension-wise scaling mismatch. With learnable gain, the normalization adapts its per-dimension sensitivity, reducing the burden on the downstream linear layer. In AGC, this per-channel adaptation is critical for wideband signals where different frequency bands have different noise floors -- analogously, different embedding dimensions in a transformer carry different amounts of information and benefit from different normalization gains.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.005
**Predicted confidence:** medium
**Case for failure:** (1) The per-dimension gain might conflict with Muon's Newton-Schulz orthogonalization, which normalizes weight matrices -- adding a learnable input scaling could reduce the effective rank of the gradient signal that Muon processes. However, this concern is mitigated by the fact that many Muon implementations (e.g., modded-nanogpt) use LayerNorm with learnable parameters successfully. (2) The gain parameters are per-dimension (640 per norm site, ~21 sites = 13,440 params). While these are Adam-optimized (not Muon), the per-dimension nature means they introduce a diagonal scaling that changes the geometry of the loss landscape in a way that is NOT the "scalar-per-layer" pattern that has been confirmed -- it is more aggressive. (3) torch.compile should handle element-wise multiply trivially, but the new parameters add to the Adam optimizer state. (4) With 21 norm sites, there are 21 new parameter tensors, which could cause torch.compile graph fragmentation if not handled carefully.

**Feasibility pre-flight:**
- Params: 640 per norm site. Sites: 2 per block (pre-attn, pre-MLP) * 10 blocks + 1 final norm = 21 sites * 640 = 13,440 params. Negligible.
- VRAM: 13,440 params * (2 bytes param + 2 bytes Adam m + 2 bytes Adam v) = ~80 KB. Negligible.
- Wall-clock: One element-wise multiply per norm site. Negligible vs matmul cost.
- torch.compile: element-wise multiply fuses trivially with RMSNorm. No graph breaks. The gain parameter is a simple nn.Parameter, not a new module.
- Muon compatibility: The gain vectors are Adam-optimized, not Muon-optimized. They change the input distribution to Muon-optimized matrices, but this is exactly what standard LayerNorm/RMSNorm with gain does in every LLaMA-class model trained with Muon.

**Implementation sketch:**
- Option A (minimal): In GPT.__init__, create `self.norm_gains = nn.ParameterList([nn.Parameter(torch.ones(config.n_embd)) for _ in range(2 * config.n_layer + 1)])`. In the `norm` function, pass gain as argument: `def norm(x, gain=None): y = F.rms_norm(x, (x.size(-1),)); return y * gain if gain is not None else y`.
- Option B (cleaner): Create a small `AffineRMSNorm` module. But this might change the parameter collection logic.
- In Block.forward: `x = x + self.attn(norm(x, gain_attn), ve, cos_sin, window_size)` and `x = x + self.mlp(norm(x, gain_mlp))`.
- Init: all gain vectors to ones (identity init, preserving baseline behavior).
- Optimizer: add norm_gains to a new Adam group with lr=scalar_lr, same betas. Or add to the existing embedding Adam group.
- IMPORTANT: the norm gains should NOT be added to a Muon group (they are 1D vectors, not 2D matrices).
- CRITICAL: The `setup_optimizer` method has an assertion that all parameters are accounted for (line 253 in train.py). The norm_gains must be explicitly added to the assertion count and to a param_group. Store them as `self.norm_gains` on the GPT model and add a dedicated Adam group. Alternatively, store the gains inside each Block (2 per block) and the final norm gain on GPT, and update the assertion accordingly.
- Preferred approach: store gains inside Block as `self.norm_gain_attn = nn.Parameter(torch.ones(n_embd))` and `self.norm_gain_mlp = nn.Parameter(torch.ones(n_embd))`, plus `self.final_norm_gain = nn.Parameter(torch.ones(n_embd))` on GPT. Then in setup_optimizer, collect them separately from matrix_params (filter by ndim==1 and not being attn_temperature) and add to a new Adam group.

---

## Hypothesis H10: SwiGLU MLP | Status: REFUTED

### Sprint Contract

**Intervention:** Replace ReluSquared MLP with SwiGLU (gate = Silu(Wx) * Vx), keeping hidden dim at 4*n_embd for both projections (total 12*n_embd params per MLP vs current 8*n_embd, offset by stronger per-param efficiency).

**Subsystem:** MLP

**Papers:** [Domain A] Shazeer 2020 "GLU Variants Improve Transformer" (arXiv:2002.05202) x [Domain B] Donoho & Johnstone 1994 "Ideal Spatial Adaptation by Wavelet Shrinkage" (Biometrika 81(3):425-455, signal processing)

**Closest prior art:** SwiGLU is well-known in LLaMA etc. The prior knowledge says "SwiGLU beats ReluSquared per-step (v4)" but was tested in combination with attention stacking (H6, cycle 2) which showed destructive interference. SwiGLU has NEVER been tested in isolation in this codebase.

**Cross-domain mapping:** Donoho-Johnstone showed that soft thresholding (shrinking small wavelet coefficients to zero) achieves near-minimax optimal denoising. SwiGLU's gating mechanism performs an analogous soft selection: the sigmoid-weighted gate continuously interpolates between passing and suppressing each hidden dimension, acting as a learned soft threshold that adapts to the input distribution. ReluSquared is a hard threshold (zero below 0, quadratic above) which discards gradient information for negative pre-activations. The signal processing insight is that soft shrinkage preserves more signal while still enforcing sparsity.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.006
**Predicted confidence:** medium
**Case for failure:** SwiGLU adds 50% more MLP parameters (12x vs 8x n_embd). With fixed step budget, the extra params may not be utilized efficiently. The 50% increase in MLP FLOPs may push wall-clock over budget. Also, Muon optimizes all matrix params the same way -- SwiGLU's gate projection may need different LR than the value projection, but we use a single Muon group per shape. Finally, torch.compile may handle the SiLU*linear product differently than relu().square().

**Feasibility pre-flight:** Extra params: ~10M (from 4*640*640*2 to 4*640*640*3). VRAM increase ~2-3GB for activations and optimizer state. Should stay under 76GB. torch.compile should handle SiLU natively. No new scalar params, no graph breaks.

**Implementation sketch:**
- In MLP.__init__: add self.c_gate = nn.Linear(n_embd, 4*n_embd, bias=False) alongside c_fc
- In MLP.forward: replace `x = F.relu(x).square()` with `x = F.silu(self.c_gate(orig_x)) * self.c_fc(orig_x)` (need to save orig_x before c_fc)
- In init_weights: init c_gate same as c_fc (uniform [-s, s])
- Add c_gate to Muon param collection (already automatic since it's in transformer.h)

---

## Hypothesis H11: Causal EMA Pre-Filter on QK Inputs | Status: REJECTED

**Rejection reason:** Causal EMA requires a sequential scan over the sequence dimension (first-order IIR filter), which is highly likely to break torch.compile fullgraph mode or cause severe wall-clock regression. No proven torch.compile-compatible implementation path was provided. The sketch hand-waves "parallel scan or torch associative scan" but torch._higher_order_ops.associative_scan is experimental and not guaranteed to work under torch.compile with fullgraph=True.

### Sprint Contract

**Intervention:** Before Q and K projections, apply a lightweight causal exponential moving average (EMA) over the sequence dimension with a learned per-channel decay rate, providing local-context smoothing that complements the global attention mechanism.

**Subsystem:** attention

**Papers:** [Domain A] Ma et al. 2022 "MEGA: Moving Average Equipped Gated Attention" (arXiv:2209.10655) x [Domain B] Brown 1963 "Smoothing, Forecasting and Prediction of Discrete Time Series" (control theory / signal processing, exponential smoothing as optimal 1st-order IIR low-pass filter)

**Closest prior art:** MEGA uses EMA as a full parallel pathway with gated attention. We propose a much simpler approach: just EMA-smooth the input to Q/K projections (not V), acting as a causal low-pass pre-filter. This has NOT been tested in this codebase. Positional encoding changes were found minimal (v5-v6), but this is different -- it modifies the feature representation, not the position encoding.

**Cross-domain mapping:** In control theory, exponential smoothing is the optimal minimum-variance estimator for a random walk with noise (Kalman filter degenerates to EMA for scalar state). The attention mechanism computes global pairwise similarities but has no built-in local smoothing -- nearby tokens that should have similar QK representations may differ due to noise. An EMA pre-filter on Q/K inputs acts as a low-pass filter, suppressing high-frequency noise in the query/key space while preserving the slow-varying semantic signal. This is mechanistically different from positional encoding (which adds position info) -- it smooths the content signal.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.004
**Predicted confidence:** low
**Case for failure:** (1) The causal EMA requires a sequential scan over the sequence dimension, which may be slow and break torch.compile's fusion. (2) At 2048 context length with sliding window attention, local context is already well-captured. (3) The EMA may blur token boundaries, hurting the model's ability to distinguish adjacent tokens. (4) Learnable per-channel decay is a new scalar param that needs Adam optimization -- potential torch.compile graph break. (5) May conflict with the existing QK-norm which already normalizes Q/K representations.

**Feasibility pre-flight:** EMA scan is O(T*C) per layer, small relative to attention. Learnable decay: n_embd=640 scalar params per layer (10 layers = 6400 params). Torch.compile can handle cumulative ops but may struggle with the sequential scan -- may need to implement as a parallel scan or use torch.cumsum trick. VRAM negligible.

**Implementation sketch:**
- Add learnable log_alpha parameter (n_embd,) per CausalSelfAttention, initialized to -1.0 (alpha ~0.37, moderate smoothing)
- In forward, before Q/K projection: compute alpha = sigmoid(self.log_alpha), then apply causal EMA: y[t] = alpha * x[t] + (1-alpha) * y[t-1]. Implement via parallel scan or the torch associative scan.
- Only apply to Q/K input, not V (V needs the raw token representation for accurate output)
- Add log_alpha params to scalar optimizer group

---

## Hypothesis H12: Learned Soft-Shrinkage Activation (ShrinkReLU) | Status: CONFIRMED

**Evaluator note:** The implementation sketch is confused with multiple formulations. Use the simplified form: `tau = F.softplus(self.raw_tau); x = F.relu(x - tau).square()`. This is the natural extension of ReluSquared (shifted threshold). Do NOT use the symmetric soft-thresholding variant, which changes behavior for negative inputs.

### Sprint Contract

**Intervention:** Replace ReluSquared with a soft-thresholding activation: ShrinkReLU(x) = sign(x) * max(|x| - tau, 0) * max(|x| - tau, 0), where tau is a small learned per-layer threshold. This is ReluSquared but with a learned positive shift, implementing Donoho-Johnstone wavelet shrinkage directly as an activation function.

**Subsystem:** activation

**Papers:** [Domain A] Zhao et al. 2019 "Deep Residual Shrinkage Networks for Fault Diagnosis" (IEEE TII, uses soft thresholding as learned denoising in CNNs) x [Domain B] Donoho & Johnstone 1994 "Ideal Spatial Adaptation by Wavelet Shrinkage" (Biometrika, signal processing: soft thresholding is minimax-optimal for Gaussian noise removal in wavelet domain)

**Closest prior art:** Deep Residual Shrinkage Networks use attention-based threshold estimation. We use a simpler per-layer learned scalar threshold. xIELU (v6) modifies the activation with learned parameters but uses a different mechanism (input-dependent exponential gating). ShrinkReLU preserves the ReluSquared structure but adds adaptive denoising via the threshold shift.

**Cross-domain mapping:** In signal processing, soft thresholding at level tau transforms coefficient c to sign(c)*max(|c|-tau, 0). Applied to MLP hidden activations, this acts as a denoising operator: small activations (likely noise from imperfect earlier layers) are zeroed, while large activations (signal) are preserved with reduced bias. Squaring after soft threshold maintains the sparsity-promoting property of ReluSquared. The key insight from Donoho-Johnstone is that the optimal threshold scales with noise level -- making tau learnable lets the network adapt the denoising strength per layer.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.004
**Predicted confidence:** medium
**Case for failure:** (1) The threshold tau is a new scalar per layer (10 params total) -- but v6 showed learnable scaling with identity init works, so this should be safe. (2) tau needs to stay positive; if it goes negative the activation becomes leaky-ReLU-like which may not help. Need to parameterize as softplus(raw_tau). (3) The benefit may be too small at this model scale -- denoising matters more in deeper/wider networks. (4) Per-layer threshold may be too coarse; per-channel would be better but adds n_embd*n_layer scalar params.

**Feasibility pre-flight:** Only 10 new scalar parameters (one per layer). Zero VRAM impact. No new matrix ops. torch.compile handles element-wise ops trivially. Parameterize tau = softplus(raw_tau) with raw_tau initialized to -2.0 (tau ~0.13, small initial threshold). Add to scalar optimizer group.

**Implementation sketch:**
- In MLP.__init__: add self.raw_tau = nn.Parameter(torch.tensor(-2.0))
- In MLP.forward: replace `x = F.relu(x).square()` with `tau = F.softplus(self.raw_tau); x = F.relu(x.abs() - tau).square() * x.sign()` -- but since we're squaring, sign doesn't matter, so simplify to: `tau = F.softplus(self.raw_tau); x = F.relu(x - tau).square() + F.relu(-x - tau).square()` -- Actually even simpler: `tau = F.softplus(self.raw_tau); x = (F.relu(x - tau))**2 + (F.relu(-x - tau))**2`
- Actually the simplest correct form: soft_threshold then square. soft_threshold(x, tau) = sign(x) * relu(|x| - tau). Then square it. Since squaring removes sign: `(relu(|x| - tau))^2 = relu(x - tau)^2 + relu(-x - tau)^2` when x >= 0 only first term active, when x < 0 only second. But ReluSquared only passes positive x. So this is more like: keep ReluSquared but shift the threshold: `F.relu(x - tau).square()` where tau >= 0.
- Add raw_tau to scalar optimizer group in setup_optimizer

---

## Hypothesis H13: Divisive Normalization in Attention Logits | Status: INCONCLUSIVE

**Evaluator note:** The reformulation (output normalization instead of logit normalization) is a substantial departure from the original hypothesis. The neuroscience cross-domain mapping about gain control on attention distribution no longer cleanly applies to the reformulated output normalization. The mechanism is now closer to "adaptive output scaling" than "divisive normalization." Accept with this caveat.

### Sprint Contract

**Intervention:** After computing attention logits (QK^T) but before softmax, apply divisive normalization inspired by cortical gain control: divide each attention logit by a pool of nearby logit magnitudes plus a semi-saturation constant sigma. This replaces the fixed softcap with an adaptive, context-dependent normalization.

**Subsystem:** attention

**Papers:** [Domain A] Attention softcapping in Gemma-2 (Team et al. 2024, uses tanh-based logit capping) x [Domain B] Carandini & Heeger 2012 "Normalization as a canonical neural computation" (Nature Reviews Neuroscience 13:51-62, neuroscience: divisive normalization explains gain control across sensory cortices)

**Closest prior art:** Current code uses tanh softcapping at 12 (for output logits, softcap=15). Divisive normalization differs from softcap: softcap is a fixed sigmoid-like bound, while divisive normalization is context-dependent -- the suppression strength depends on the activity of the normalization pool. This is fundamentally different from QK-norm (which normalizes Q/K vectors before dot product) and from attention temperature (which is a fixed per-head scalar).

**Cross-domain mapping:** In visual cortex, a neuron's response R to stimulus is R = c^n / (sigma^n + sum(c_j^n)), where the denominator pools over neighboring neurons. This implements contrast gain control: in high-activity contexts, all responses are suppressed; in low-activity contexts, weak signals are amplified. Applied to attention: when many keys are relevant (high total logit energy), divisive normalization sharpens the attention distribution more than softmax alone. When few keys match (sparse attention), the normalization is weaker, preserving gradient flow. This is exactly the adaptive gain control that fixed softcap cannot provide.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003
**Predicted confidence:** low
**Case for failure:** (1) We use FlashAttention3 which computes QK^T internally -- we CANNOT modify the attention logits without breaking FA3. This is a FATAL constraint unless we apply divisive normalization BEFORE the dot product (e.g., on Q or K). (2) If applied pre-dot-product, the mechanism changes: we'd be normalizing Q/K representations by their local energy, which is partially redundant with QK-norm. (3) Adding a learnable sigma parameter interacts with attention temperature. (4) Any modification to Q/K after QK-norm may destabilize training.

**Feasibility pre-flight:** CRITICAL ISSUE: FlashAttention3 does not expose attention logits. Must reformulate as pre-softmax normalization of Q or K. Alternative: apply divisive normalization to V after attention (normalize attention output by a function of attention entropy). This changes the hypothesis significantly. VRAM: negligible. torch.compile: element-wise ops only.

**REFORMULATED INTERVENTION:** Apply divisive normalization to the attention OUTPUT (after FA3): out = out / (1 + sigma * ||out||_dim), where sigma is a learnable per-head scalar initialized near zero (identity passthrough). This normalizes the attended values by their own energy, implementing gain control on the attention output rather than logits.

**Implementation sketch:**
- In CausalSelfAttention.__init__: add self.div_norm_sigma = nn.Parameter(torch.zeros(n_head))
- In forward, after y = fa3.flash_attn_func(...): apply y = y / (1 + F.softplus(self.div_norm_sigma).view(1,1,n_head,1) * y.norm(dim=-1, keepdim=True))
- Add div_norm_sigma to scalar optimizer group
- Initialize to zero so initial behavior is identity (passthrough)

---

## Hypothesis H14: Frequency-Aware Residual Gating | Status: REJECTED

**Rejection reason:** Fatal causality violation. The proposed decomposition computes x.mean(dim=1, keepdim=True) over the full sequence dimension, which leaks future token information into every position's residual stream. This breaks the autoregressive causal structure. During inference, the mean would shift with each new token, destabilizing earlier representations. Unlike RMSNorm (which normalizes over the embedding dimension per-token), this is a cross-position operation that violates causality.

### Sprint Contract

**Intervention:** Replace the fixed per-layer resid_lambdas scaling with a frequency-dependent residual gate that applies different scaling to the low-frequency and high-frequency components of the residual stream, using a simple mean/deviation decomposition (no FFT needed).

**Subsystem:** residuals

**Papers:** [Domain A] ResFormer / DeepNet residual scaling (Wang et al. 2022 "DeepNet: Scaling Transformers to 1000 Layers", arXiv:2203.00555) x [Domain B] Multi-resolution analysis in wavelet theory (Mallat 1989 "A Theory for Multiresolution Signal Decomposition", IEEE PAMI 11(7):674-693, signal processing)

**Closest prior art:** Current code has resid_lambdas (per-layer scalar) and x0_lambdas (skip connection to initial embedding). This treats the residual stream as a single signal. DeepNet uses fixed alpha scaling. Neither considers that the residual stream carries both low-frequency (document-level semantics, mean activation) and high-frequency (token-level variation, deviation from mean) information that may benefit from different scaling.

**Cross-domain mapping:** Mallat's multi-resolution analysis decomposes a signal into approximation (low-freq) and detail (high-freq) coefficients at each scale, with different processing at each level. In our transformer, each layer's residual stream x can be decomposed into mean (low-freq: batch/sequence-averaged activation pattern) and deviation (high-freq: per-token variation). A learnable gate that scales these differently per layer allows the network to control the balance between preserving global context (low-freq, important for coherent generation) and local detail (high-freq, important for token prediction). This is a minimal multi-resolution decomposition at each residual connection.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003
**Predicted confidence:** low
**Case for failure:** (1) The mean/deviation decomposition across the sequence dimension may not correspond to meaningful low/high frequency content. (2) Two scalar params per layer (20 total) is minimal but adds complexity. (3) Prior knowledge says "learnable scaling with identity/near-passthrough init" is the winning pattern (v6) -- this hypothesis follows that pattern but adds the frequency decomposition which may not help. (4) Could conflict with existing resid_lambdas and x0_lambdas. (5) Per-dimension scaling conflicts with Muon's Newton-Schulz -- but this is per-LAYER not per-dimension, so should be safe.

**Feasibility pre-flight:** 20 new scalar params (2 per layer). Zero VRAM. No matrix ops. torch.compile handles mean and subtraction trivially. Initialize both gate scalars to equal values so initial behavior matches current resid_lambdas.

**Implementation sketch:**
- In GPT.__init__: add self.resid_lo = nn.Parameter(torch.ones(n_layer)) and self.resid_hi = nn.Parameter(torch.ones(n_layer))
- In forward loop: decompose x = x_mean + x_dev where x_mean = x.mean(dim=1, keepdim=True) and x_dev = x - x_mean. Then apply: x = self.resid_lo[i] * x_mean + self.resid_hi[i] * x_dev + self.x0_lambdas[i] * x0. Remove resid_lambdas (replaced by resid_lo/resid_hi).
- Add resid_lo, resid_hi to scalar optimizer group (same LR as resid_lambdas: scalar_lr * 0.01)
- Initialize both to 1.0 (identity init, matching current resid_lambdas=1.0 behavior)

---

## Hypothesis H17: Per-Head Attention Output Scaling (HeadScale) | Status: APPROVED

**Evaluator note (pre-run):** APPROVED. Clean B1-pattern intervention: O(n_head * n_layer) = 72 scalar params (not 50 -- the contract uses n_head=5, n_layer=10 but actual config is n_head=6, n_layer=12; correct the implementation accordingly). Identity init, element-wise op, trivially torch.compile-safe. The Muon-orthogonalization argument is sound: Muon removes scale information from weight matrices, so per-head output scaling provides a non-redundant degree of freedom. Cross-domain papers check out (ML + neuroscience). No pivot violations. CRITICAL PRE-CONDITION: the codebase currently has SwiGLU (H15); MUST revert to ShrinkReLU before running this experiment. Delta is measured against raw baseline (0.959552) as always, but the result should beat ShrinkReLU's -0.005798 to demonstrate marginal value of HeadScale on top of ShrinkReLU.

### Sprint Contract

**Intervention:** Add a single learnable scalar per attention head that scales the head's contribution to the output before concatenation and c_proj. Initialize to 1.0 (identity). This gives the network the ability to up-weight or down-weight individual heads during training without modifying the attention logits or requiring changes inside FlashAttention3.

**Subsystem:** attention/gating

**Papers:** [Domain A] Bhojanapalli et al. 2021 "Leveraging Redundancy in Attention with Reuse Transformers" (NeurIPS 2021 workshop) -- showed attention heads develop redundant patterns and benefit from explicit head-level gating x [Domain B] Turrigiano 2008 "The Self-Tuning Neuron: Synaptic Scaling of Excitatory Synapses" (Annual Review of Neuroscience 31:25-46) -- synaptic scaling is a homeostatic plasticity mechanism where neurons multiplicatively scale all synaptic inputs by a single gain factor to maintain a target firing rate. This is distinct from Hebbian plasticity (which changes individual synapses) and operates at the neuron level.

**Closest prior art:** The codebase already has per-head attn_temperature (scales Q before dot product). HeadScale is fundamentally different: it scales the OUTPUT of each head after attention is computed. attn_temperature controls attention sharpness (how peaked the softmax distribution is); HeadScale controls how much each head's output contributes to the residual stream. These are orthogonal: temperature affects WHERE the head attends, HeadScale affects HOW MUCH the attended result matters. Differential attention (tried and failed 3 times in v3-v5) is also different -- it subtracts pairs of heads, which is a destructive operation. HeadScale is purely multiplicative scaling, preserving the head's attention pattern.

**Cross-domain mapping:** In neuroscience, synaptic scaling adjusts the strength of ALL synapses on a neuron by a common multiplicative factor, maintaining the relative pattern of synaptic weights while adjusting the overall gain. This is exactly what HeadScale does for attention heads: the attention pattern (which tokens attend to which) is preserved by FlashAttention3, but the overall contribution of each head is scaled. Turrigiano showed this mechanism is critical for maintaining stable network dynamics during learning -- without it, Hebbian plasticity causes runaway excitation or silencing. Analogously, without HeadScale, heads that learn redundant patterns cannot be down-weighted, wasting representational capacity. The scalar-per-head granularity (5 params total) is directly analogous to the neuron-level granularity of synaptic scaling (one gain per neuron, not per synapse).

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.004
**Predicted confidence:** medium
**Case for failure:** (1) With only 5 heads, the per-head scaling has very few degrees of freedom. If all heads are equally useful, the scalars will stay near 1.0 and produce no benefit. (2) The c_proj linear layer already provides head mixing -- it can implicitly learn to up/down-weight heads through its weight matrix rows. HeadScale may be redundant with what c_proj already does. (3) Muon optimizes c_proj, and adding a scalar scaling before c_proj changes the effective gradient scale for c_proj's rows, which could interact unpredictably with Muon's Newton-Schulz orthogonalization. However, attn_temperature already does this for Q, so the precedent exists. (4) The 5 scalar params are small enough that scalar_lr=0.5 should be safe (same scale as attn_temperature's 5 params per layer -- wait, actually n_head=5 per layer, so 5 params per layer = 50 total, which is still O(n_layer*n_head), within the safe zone for scalar_lr).

**Feasibility pre-flight:**
- Params: 5 per layer * 10 layers = 50 new scalar params. Negligible.
- VRAM: Zero additional VRAM. The scaling is an element-wise multiply on y before reshape, reusing the existing tensor.
- Wall-clock: One element-wise multiply per layer. Negligible.
- torch.compile: Element-wise multiply with a broadcast scalar. Trivially fusible. No graph breaks.
- Muon compatibility: The head_scale params are 1D scalars, Adam-optimized. They do not enter any Muon group. The c_proj input is scaled, but this is analogous to attn_temperature scaling Q before the dot product -- a pattern already in the codebase that works with Muon.
- LR: scalar_lr=0.5 for 50 params (O(n_head * n_layer)). This is within the safe zone established by attn_temperature (also 50 params at scalar_lr). Identity init (1.0) means initial behavior is unchanged.

**Implementation sketch:**
- In CausalSelfAttention.__init__: add `self.head_scale = nn.Parameter(torch.ones(config.n_head))`
- In CausalSelfAttention.forward, after `y = fa3.flash_attn_func(...)` (y shape: B, T, n_head, head_dim): apply `y = y * self.head_scale.view(1, 1, self.n_head, 1)`
- Then proceed with existing `y = y.contiguous().view(B, T, -1)` and `y = self.c_proj(y)`
- In setup_optimizer: collect head_scale params alongside attn_temperature params (same lr=scalar_lr, same betas). Simplest: add them to the temp_params list, or create a parallel collection. Update the assertion.
- Init: ones (identity). Already done by nn.Parameter(torch.ones(...)).

---

## Hypothesis H18: Learned Sublayer Output Scaling (SubScale) | Status: APPROVED

**Evaluator note (pre-run):** APPROVED with caution. B1-pattern compliant: 24 scalar params (not 20 -- actual config has 12 layers, not 10). Identity init, conservative LR (scalar_lr*0.01). The Muon-orthogonalization argument provides genuine theoretical motivation. YELLOW FLAG: H9 (cycle 2) tested "sublayer scaling" in the residuals subsystem and was REFUTED. The contract claims differences (identity init, conservative LR) distinguish it from H9, but H9's exact configuration is unknown. If H9 also used identity init, this is perilously close to repetition. PROCEED with awareness that the residuals/scaling subsystem is 0/1 (H9 refuted). A second failure here would put the subsystem at 0/2, one step from BLOCKED. Parameter counts use wrong config values (n_layer=10 instead of 12); correct in implementation. CRITICAL PRE-CONDITION: revert codebase from SwiGLU to ShrinkReLU before running.

### Sprint Contract

**Intervention:** Add two learnable scalar parameters per layer -- one for the attention sublayer output and one for the MLP sublayer output -- that scale the sublayer contribution before it is added to the residual stream. Initialize to 1.0 (identity). This decouples the attention/MLP contribution magnitudes from the residual stream scaling (resid_lambdas), which currently scales the INPUT to each block but not the individual sublayer OUTPUTS.

**Subsystem:** residuals/scaling

**Papers:** [Domain A] Wang et al. 2022 "DeepNet: Scaling Transformers to 1000 Layers" (arXiv:2203.00555) -- showed that sublayer-specific scaling (alpha * sublayer_output) with carefully chosen constants enables stable training of very deep transformers x [Domain B] Turrigiano & Nelson 2004 "Homeostatic Plasticity in the Developing Nervous System" (Nature Reviews Neuroscience 5:97-107) -- showed that biological neural networks maintain stable activity through multiplicative gain adjustment on excitatory and inhibitory pathways INDEPENDENTLY, not through a single global gain. The key insight is that excitatory (attention = aggregation) and inhibitory (MLP = transformation) pathways require different homeostatic scaling to maintain stable dynamics.

**Closest prior art:** The codebase has resid_lambdas (scales residual stream before each block) and x0_lambdas (skip connection to initial embedding). These operate on the INPUT. DeepNet uses FIXED alpha scaling on sublayer outputs. H9 (cycle 2, REFUTED) tested "sublayer scaling" but details are sparse -- the key difference here is: (1) we use identity init (not DeepNet's formula), (2) we use scalar_lr*0.01 (same as resid_lambdas, conservative), (3) we scale BOTH attn and MLP outputs independently. The existing resid_lambdas + x0_lambdas already operate at scalar_lr*0.01, establishing that per-layer residual scalars work at this LR.

**Cross-domain mapping:** In developing neural circuits, homeostatic plasticity independently adjusts the gain of excitatory and inhibitory synapses to maintain a target firing rate. If only a global gain were adjusted, the excitation/inhibition (E/I) balance would be locked. Independent scaling allows the network to shift E/I balance as needed during learning. In our transformer, attention (aggregation from context) and MLP (per-token transformation) play analogous roles to excitatory and inhibitory pathways. Currently, resid_lambdas scales the combined residual stream, locking the attention/MLP balance. SubScale allows the network to independently adjust how much attention vs MLP contribution flows into the residual stream at each layer, analogous to independent E/I gain control.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.004
**Predicted confidence:** medium
**Case for failure:** (1) resid_lambdas already provides per-layer scaling, and the downstream linear layers can implicitly adjust the relative attention/MLP contribution. SubScale may be redundant. (2) H9 (cycle 2) tested sublayer scaling and was REFUTED -- though we do not know the exact details, this is a warning sign. The key differences (identity init, conservative LR) may or may not be sufficient to change the outcome. (3) With only 20 new params, the signal may be too small to clear the -0.003 threshold. (4) The attention output is already scaled by attn_temperature (indirectly, through Q scaling) and potentially HeadScale (if H17 is also tested). Adding SubScale on top could over-parameterize the attention path.

**Feasibility pre-flight:**
- Params: 2 per layer * 10 layers = 20 new scalar params. Negligible.
- VRAM: Zero. Element-wise scalar multiply on existing tensors.
- Wall-clock: Two element-wise multiplies per layer. Negligible.
- torch.compile: Trivially fusible scalar-tensor multiply. No graph breaks.
- Muon compatibility: Not Muon-optimized. The scaled outputs feed into the residual stream which then enters the next block's norm+linear. Since the scaling is a simple scalar multiply on the sublayer output, it is equivalent to scaling the c_proj output, which is already zero-initialized. No conflict.
- LR: scalar_lr * 0.01 = 0.005, same as resid_lambdas. This is conservative and appropriate for residual stream scaling params. 20 params at lr=0.005 is well within the safe zone.

**Implementation sketch:**
- In Block.__init__: add `self.attn_scale = nn.Parameter(torch.ones(1))` and `self.mlp_scale = nn.Parameter(torch.ones(1))`
- In Block.forward: change `x = x + self.attn(norm(x), ve, cos_sin, window_size)` to `x = x + self.attn_scale * self.attn(norm(x), ve, cos_sin, window_size)` and `x = x + self.mlp(norm(x))` to `x = x + self.mlp_scale * self.mlp(norm(x))`
- In setup_optimizer: collect attn_scale and mlp_scale params from all blocks. Add to a new Adam group with lr=scalar_lr*0.01, same betas as resid_lambdas. Update the assertion.
- Init: ones (identity).

---

## Hypothesis H19: Learned Output Softcap Temperature | Status: APPROVED

**Evaluator note (pre-run):** APPROVED, but low expectations. 1 scalar parameter is the absolute minimum B1-pattern intervention. Predicted delta at threshold boundary (-0.003, low confidence) is a risky prediction -- the Researcher is essentially predicting a coin flip. The implementation is clean and zero-risk (identity init, trivial torch.compile). LR choice is under-specified in the contract: the implementation sketch suggests scalar_lr with beta1=0.96 but does not commit. The Researcher MUST specify the exact optimizer group in implementation. Recommend using the x0_params group (lr=scalar_lr, betas=(0.96, 0.95)) since softcap is a global training dynamic parameter like x0_lambdas, not a per-layer architectural parameter. Cross-domain papers check out (ML + physics). CRITICAL PRE-CONDITION: revert codebase from SwiGLU to ShrinkReLU before running.

### Sprint Contract

**Intervention:** Replace the fixed softcap=15 on output logits with a learnable softcap temperature: `logits = softcap * tanh(logits / softcap)` where `softcap = 15 * softplus(raw_softcap) / softplus(0)` so that raw_softcap=0 gives softcap=15 (identity init). This allows the model to learn the optimal logit range for the vocabulary distribution, rather than using a fixed value chosen by human tuning.

**Subsystem:** embeddings

**Papers:** [Domain A] Team Gemma 2024 "Gemma 2: Improving Open Language Models at a Practical Size" (arXiv:2408.00118) -- introduced logit softcapping as a stabilization technique, using fixed constants (softcap=30 for attn logits, softcap=30 for output logits) x [Domain B] Jaynes 1957 "Information Theory and Statistical Mechanics" (Physical Review 106(4):620-630) -- the maximum entropy principle states that the optimal distribution is the one that maximizes entropy subject to known constraints (expected values). The softcap constrains the entropy of the output distribution: a larger softcap allows sharper (lower entropy) distributions, while a smaller softcap forces more uniform (higher entropy) distributions. The optimal softcap is the one that matches the true entropy of the target distribution, which varies with training progress.

**Closest prior art:** Current code uses fixed softcap=15. Gemma-2 uses fixed softcap=30. No prior work (to our knowledge) makes the softcap temperature learnable. The attn_temperature parameter in the codebase already establishes the pattern of learnable temperature scaling for softmax-like operations (attention). This extends the pattern to the output logit softcap.

**Cross-domain mapping:** Jaynes showed that the partition function temperature in statistical mechanics determines the sharpness of the Boltzmann distribution: high temperature = uniform, low temperature = peaked. The softcap plays an analogous role for the output logit distribution: it bounds the dynamic range of logits, which controls how peaked the softmax output can be. A fixed softcap=15 imposes a fixed constraint on the output distribution's entropy. But the optimal constraint changes during training: early in training, when the model's predictions are noisy, a tighter softcap (smaller value) regularizes by preventing overconfident predictions. Late in training, a looser softcap (larger value) allows the model to express high-confidence predictions for tokens it has learned well. Making softcap learnable allows the model to find its own optimal "temperature" for the output distribution, analogous to a system finding its thermodynamic equilibrium temperature.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003
**Predicted confidence:** low
**Case for failure:** (1) The fixed softcap=15 may already be near-optimal for this model size and vocabulary. If so, the learnable parameter will stay near 15 and produce no benefit. (2) The softcap operates on float32 logits AFTER the lm_head linear layer. Changing it affects the loss gradient, which backpropagates through lm_head (Adam-optimized) into the final norm and residual stream. A rapidly changing softcap could destabilize the gradient scale. (3) This is a single scalar parameter -- the improvement potential from one degree of freedom is limited. (4) The softcap interacts with the unembedding_lr: if the model can learn a larger softcap, it may be equivalent to simply increasing unembedding_lr, which was presumably already tuned. (5) torch.compile: softplus and multiplication are trivial ops, but the parameter enters the loss computation graph. This should be fine since it is just a scalar multiply/tanh/multiply chain.

**Feasibility pre-flight:**
- Params: 1 scalar param. Negligible.
- VRAM: Zero. No new tensors.
- Wall-clock: Zero measurable cost. softplus + scalar multiply.
- torch.compile: softplus, div, tanh, mul are all native ops. The scalar parameter enters the compiled forward function graph but causes no graph breaks (same as resid_lambdas which are also scalars used in forward).
- Muon compatibility: Not Muon-optimized. Affects only the loss gradient magnitude, which flows through Adam-optimized lm_head. No interaction with Muon.
- LR: scalar_lr=0.5 for 1 param. This may be too aggressive for a parameter that controls gradient scale. Consider using scalar_lr*0.1=0.05 or x0_lambda's LR. Alternative: use scalar_lr with beta1=0.96 (same as x0_lambdas) for slower adaptation.

**Implementation sketch:**
- In GPT.__init__: add `self.raw_softcap = nn.Parameter(torch.zeros(1))` (init to 0 = identity at softcap=15)
- In GPT.forward: replace `softcap = 15` with `softcap = 15.0 * F.softplus(self.raw_softcap) / F.softplus(torch.zeros(1, device=self.raw_softcap.device))`. Simplify: precompute softplus(0) = ln(2) ~= 0.6931. So `softcap = 15.0 * F.softplus(self.raw_softcap) / 0.6931`.
- In setup_optimizer: add raw_softcap to a new Adam group or to x0_params group (lr=scalar_lr, betas=(0.96, 0.95), weight_decay=0). Update assertion.
- Init: raw_softcap=0 gives softcap = 15 * softplus(0)/softplus(0) = 15. Identity.

---

## Hypothesis H20: Learned Per-Layer Norm Scaling (NormScale) | Status: APPROVED

**Evaluator note (pre-run):** APPROVED. This is a well-motivated, mechanistically distinct retry of the normalisation subsystem after H16's catastrophic failure. The key distinction is explicit and correct: H16 used 13,440 per-dimension params at scalar_lr=0.5 (violated B1 pattern); NormScale uses 25 scalar params (not 21 -- actual config has 12 layers, so 2*12+1=25 sites) at scalar_lr=0.5 (within B1 safe zone). The Muon-orthogonalization argument is the strongest theoretical motivation among H17-H20: if Muon normalizes weight matrices, input scale is a genuine lost degree of freedom that these scalars restore. Cross-domain papers: ML (Zhang & Sennrich) + Physics (Boltzmann/Gibbs). The physics paper citation is unusual (1877/1902 foundational works rather than a specific applied paper), but the mapping to temperature/partition function is legitimate. CRITICAL PRE-CONDITION: revert codebase from SwiGLU to ShrinkReLU before running.

### Sprint Contract

**Intervention:** Add a single learnable scalar per RMSNorm site (21 sites total: 2 per block + 1 final) that scales the norm OUTPUT: `norm(x) * gamma_scalar` where gamma_scalar is a scalar (not per-dimension). Initialize to 1.0 (identity). This is the scalar-per-site version of H16's per-dimension gain, explicitly designed to stay within the safe B1 pattern (O(n_sites) = 21 scalar params, not O(n_embd * n_sites) = 13,440 per-dimension params).

**Subsystem:** normalisation/scalar-gain

**Papers:** [Domain A] Zhang & Sennrich 2019 "Root Mean Square Layer Normalization" (arXiv:1910.07467) -- showed that the gain parameter in LayerNorm/RMSNorm captures important per-layer scaling information x [Domain B] Boltzmann 1877 / Gibbs 1902 -- in statistical mechanics, the partition function Z normalizes the Boltzmann distribution. The temperature parameter T (a single scalar) controls the "sharpness" of the distribution across all energy states simultaneously. Per-energy-level weighting (analogous to per-dimension gain) is NOT how thermodynamic systems operate -- a single temperature controls the entire distribution. RMSNorm with a scalar gain is the exact analog: one scalar per normalization site controls the overall scale of the normalized output, affecting all dimensions equally.

**Closest prior art:** H16 tested per-DIMENSION gain (640 params per site, 13,440 total) at scalar_lr=0.5 and suffered catastrophic regression (+0.136). The failure was attributed to LR misconfiguration (too high for 13K params). NormScale avoids this entirely by using only 21 SCALAR params -- firmly within the B1 safe zone. The distinction is critical: H16 added O(n_embd * n_sites) parameters; NormScale adds O(n_sites) parameters. The evaluator's note on H16 says "any retry MUST use lr in the range 1e-3 to 1e-2" for per-dimension params, but NormScale is not per-dimension, so scalar_lr is appropriate.

**Cross-domain mapping:** In statistical mechanics, a system's macroscopic behavior is determined by a single temperature parameter T, not by individual per-state scalings. The partition function Z = sum(exp(-E_i/T)) normalizes the distribution, and T controls the overall concentration of probability mass. Analogously, RMSNorm normalizes the feature vector's energy, and the scalar gain controls how much of the normalized signal passes through. Per-dimension gain (H16's approach) is like assigning per-state temperatures -- more expressive but harder to optimize and potentially overfitting to noise. Scalar gain is like the thermodynamic temperature -- a single control that captures the most important degree of freedom (overall scale) while being trivially optimizable.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003 (1.5x seed variance)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003
**Predicted confidence:** low-medium
**Case for failure:** (1) With only 21 scalar params, the improvement may be too small to clear -0.003. A scalar gain on norm output is equivalent to scaling the input to the next linear layer, which the linear layer can already compensate for by adjusting its weights. The gain may be redundant. (2) However, this argument also applies to resid_lambdas and x0_lambdas (per-layer scalars on the residual stream), which ARE in the model and presumably help. (3) With Muon's Newton-Schulz orthogonalization, the effective scale of the linear layer's weights is constrained. A scalar gain before the linear layer may therefore be non-redundant -- it provides a degree of freedom that Muon's orthogonalization removes. This is actually a STRONG argument for NormScale: Muon normalizes weight matrices, so the input scale to those matrices matters and cannot be compensated by weight adjustment. (4) The 21 sites may have different optimal scales, especially pre-attention vs pre-MLP norms.

**Feasibility pre-flight:**
- Params: 21 scalar params. Negligible.
- VRAM: Zero. Scalar multiply on existing tensor.
- Wall-clock: 21 scalar-tensor multiplies. Negligible.
- torch.compile: Trivially fusible. No graph breaks.
- Muon compatibility: Not Muon-optimized. The scalar gain changes the input scale to Muon-optimized matrices. This is potentially beneficial because Muon's orthogonalization normalizes weight scales, meaning the input scale is a meaningful degree of freedom that the optimizer cannot otherwise adjust.
- LR: scalar_lr=0.5 for 21 params. This is within the safe zone (O(n_sites), comparable to resid_lambdas at 10 params and attn_temperature at 50 params). Both resid_lambdas (at scalar_lr*0.01) and attn_temperature (at scalar_lr) work. Use scalar_lr (0.5) since these are true scalar params.

**Implementation sketch:**
- Modify the `norm` function to accept an optional scalar gain: `def norm(x, gain=None): y = F.rms_norm(x, (x.size(-1),)); return y * gain if gain is not None else y`
- In Block.__init__: add `self.norm_gain_attn = nn.Parameter(torch.ones(1))` and `self.norm_gain_mlp = nn.Parameter(torch.ones(1))`
- In Block.forward: `x = x + self.attn(norm(x, self.norm_gain_attn), ve, cos_sin, window_size)` and `x = x + self.mlp(norm(x, self.norm_gain_mlp))`
- In GPT.__init__: add `self.final_norm_gain = nn.Parameter(torch.ones(1))`
- In GPT.forward: replace final `x = norm(x)` with `x = norm(x, self.final_norm_gain)`
- In setup_optimizer: collect all norm_gain params (21 total). Add to a new Adam group with lr=scalar_lr, betas=adam_betas. Update the assertion.
- Init: ones (identity).
