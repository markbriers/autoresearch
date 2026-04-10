# Hypotheses

## Engineering Run Counter: 1/5 | Phase: IMPLEMENT

---

## Hypothesis 5: xIELU + Attention Temperature Stack | Status: PENDING_EVALUATION

### Sprint Contract

**Intervention:** Combine xIELU activation (F1) with learnable per-head attention temperature (F3) in a single model, both applied to the baseline.
**Subsystem:** activation/function-choice + attention/temperature-scaling (stacking)
**Papers:** Huang et al. 2024 ("Deriving Activation Functions via Integration") x Zhang et al. NeurIPS 2024 ("Selective Attention")
**Closest prior art:** F1 (xIELU alone, delta=-0.003633) and F3 (attn_temp alone, delta=-0.004839) confirmed independently. No stacking test exists yet.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.006 (combined improvement exceeds best individual)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.006 and -0.003
**REFUTED if:** delta > -0.003 or CRASH

**Predicted delta:** -0.007 +/- 0.002
**Predicted confidence:** medium-high
**Case for failure:** The two interventions may not be fully additive. xIELU changes the MLP activation landscape, which changes the gradient distribution flowing back through attention. Attention temperature may have partially compensated for suboptimal MLP activation in the baseline, so fixing the activation with xIELU may reduce the marginal value of temperature scaling. Partial additivity (e.g., -0.005) would land in INCONCLUSIVE.
**Feasibility pre-flight:** 20 scalar params (xIELU) + 60 scalar params (attn_temp, 6 heads x 10 layers) = 80 total new scalars. [Evaluator note: contract originally said 50 attn_temp params assuming 5 heads; the model has 6 heads.] Zero VRAM overhead beyond baseline. torch.compile handles both changes independently (confirmed in cycle 1). No new Muon groups needed.
**Implementation sketch:** Apply both F1 (xIELU activation replacing ReluSquared in MLP) and F3 (per-head attn_temperature scalar multiplying q after QK-norm) to baseline train.py. Add xIELU class with per-layer alpha_p, alpha_n params. Add attn_temperature param to CausalSelfAttention. Both scalar param sets go into AdamW scalar_lr group. Cast temperature to tensor dtype before multiplication.

---

## Hypothesis 6: SwiGLU + Attention Temperature Stack | Status: APPROVED

### Sprint Contract

**Intervention:** Combine SwiGLU gated MLP (F2) with learnable per-head attention temperature (F3) in a single model, both applied to the baseline.
**Subsystem:** MLP/gating-structure + attention/temperature-scaling (stacking)
**Papers:** Shazeer 2020 ("GLU Variants Improve Transformer") x Zhang et al. NeurIPS 2024 ("Selective Attention")
**Closest prior art:** F2 (SwiGLU alone, delta=-0.003363) and F3 (attn_temp alone, delta=-0.004839) confirmed independently. No stacking test exists yet.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.006 (combined improvement exceeds best individual)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.006 and -0.003
**REFUTED if:** delta > -0.003 or CRASH

**Predicted delta:** -0.006 +/- 0.002
**Predicted confidence:** medium
**Case for failure:** SwiGLU's gating mechanism (SiLU gate) already provides some of the expressivity that attention temperature adds -- both modulate information flow. SwiGLU's standalone delta (-0.003363) was weaker than xIELU (-0.003633), so the stack may also be weaker. The three-matrix SwiGLU structure changes the gradient landscape more substantially than xIELU, potentially creating interference with temperature scaling optimization.
**Feasibility pre-flight:** SwiGLU replaces MLP with 3 matrices at hidden_dim=1664 (parameter-matched). 50 scalar params for temperature. VRAM slightly below baseline for SwiGLU alone. torch.compile handles both (confirmed in cycle 1). New Muon groups for 640x1664 and 1664x640 shapes.
**Implementation sketch:** Replace MLP with SwiGLU (gate: 640->1664, up: 640->1664, down: 1664->640, all bias=False). Forward: down(F.silu(gate(x)) * up(x)). Init gate/up uniform, down zeros. Add attn_temperature to CausalSelfAttention. Temperature params in AdamW scalar_lr group.

---

## Hypothesis 7: xIELU-Gated GLU + Attention Temperature | Status: REJECTED

**Evaluator rejection reason:** Contract is underspecified (mentions testing "both raw xIELU and sigmoid(xIELU) variants" -- that is two experiments, not one). Three-way combination (new activation + new MLP structure + temperature) prevents clean attribution of results. xIELU is mechanistically unsuited for gating (unbounded positive branch). Test xIELU-as-gate in isolation first if the idea has merit.

### Sprint Contract

**Intervention:** Replace SiLU in SwiGLU's gate with xIELU to create xIELU-GLU, combined with attention temperature. This answers Open Question #2 and stacks with F3.
**Subsystem:** activation/function-choice + MLP/gating-structure + attention/temperature-scaling
**Papers:** Huang et al. 2024 ("Deriving Activation Functions via Integration") x Shazeer 2020 ("GLU Variants") x Zhang et al. NeurIPS 2024 ("Selective Attention")
**Closest prior art:** SwiGLU (F2, delta=-0.003363) uses SiLU as gate activation. xIELU (F1, delta=-0.003633) replaces ReluSquared in standard MLP. No one has used xIELU as the gating function in a GLU structure.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.006 (must beat both H5 and H6 expectations to justify the novelty)
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.006 and -0.003
**REFUTED if:** delta > -0.003 or CRASH

**Predicted delta:** -0.007 +/- 0.003
**Predicted confidence:** low-medium
**Case for failure:** xIELU was designed as an activation function (positive quadratic + negative ELU), not as a gating function. SiLU's sigmoid-like shape is naturally suited for gating (output in ~[0,1] for moderate inputs), while xIELU's quadratic positive branch can produce unbounded outputs that may destabilize the gate. The learnable alpha_p/alpha_n may not converge well when used in a gating context where the optimal shape differs from the MLP activation context. This is the most speculative hypothesis in the cycle.
**Feasibility pre-flight:** Same structure as SwiGLU (3 matrices, hidden_dim=1664) plus 20 xIELU scalar params + 50 temperature scalars. VRAM similar to SwiGLU baseline. torch.compile: xIELU uses torch.where which compiled fine in cycle 1. The xIELU gate output is unbounded, so we may need to add a sigmoid wrapper or clamp -- implementer should test both raw xIELU and sigmoid(xIELU) variants.
**Implementation sketch:** SwiGLU structure but replace F.silu(gate(x)) with xIELU(gate(x)). The xIELU module has per-layer alpha_p, alpha_n. Forward: down(xIELU(gate(x)) * up(x)). Plus attn_temperature on attention. All scalar params in AdamW group.

---

## Hypothesis 8: Static Hyper-Connections (n=2) | Status: REJECTED

**Evaluator rejection reason:** VRAM estimate is unrealistic. Contract estimates +3-4 GB for dual streams, but omits torch.compile recompilation overhead on entirely new tensor shapes (typically +3-8 GB). Realistic total: ~77+ GB, exceeding 76 GB ceiling. Implementation sketch is too vague for the architectural complexity involved. Paper showed gains at 24+ layers; this 10-layer model is below the demonstrated benefit range.

### Sprint Contract

**Intervention:** Replace standard residual connections with static hyper-connections (expansion rate n=2), creating two parallel residual streams that interact via learnable scalar mixing weights.
**Subsystem:** residuals
**Papers:** Zhu et al. ICLR 2025 ("Hyper-Connections") x signal propagation theory (Noci et al. 2024, "Transformers Get Stable")
**Closest prior art:** The model already has resid_lambdas and x0_lambdas for residual scaling. Hyper-connections generalize this by allowing cross-stream mixing. Peri-LN (D1, REFUTED) targeted residual magnitude but via normalization, not multi-stream routing.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.004 +/- 0.002
**Predicted confidence:** medium
**Case for failure:** n=2 doubles the residual stream width, adding ~3.3 GB VRAM (128 batch * 2048 seq * 640 dim * 10 layers * 2 bytes * 2 for fwd/bwd). This could push past the 76 GB ceiling. The existing resid_lambdas/x0_lambdas already provide some residual scaling control, so hyper-connections may offer diminishing returns -- similar to how Peri-LN was partially redundant. The 10-layer model may not be deep enough to benefit from multi-stream residuals (the paper showed gains at 24+ layers). torch.compile must trace a new graph shape with the expanded hidden state, which may cause recompilation overhead.
**Feasibility pre-flight:** Static HC adds n*(n+2) = 8 learnable scalars per layer = 80 total. VRAM increase from dual stream: estimated 3-4 GB (tight but within budget if baseline is 67.8 GB). The contraction at the final layer must project back to d_model for the LM head. torch.compile: the expanded residual stream changes tensor shapes throughout, requiring full recompilation but no dynamic control flow.
**Implementation sketch:** Expand embedding output from (B,T,d) to (B,T,2,d) using learned expansion weights. Each Block receives the 2-stream state, aggregates to single stream for sublayer input (via learned A_m weights), applies sublayer, then distributes output back to 2 streams (via learned B and A_r weights). At final layer, contract back to (B,T,d) for LM head. Init: stream 0 carries the standard residual, stream 1 initialized as identity pass-through. HC matrix initialized per the paper's scheme. All HC scalars in AdamW scalar_lr group.

---

## Hypothesis 9: Learnable Sublayer Output Scaling (GPAS-lite) | Status: APPROVED

### Sprint Contract

**Intervention:** Add a learnable per-sublayer scalar that scales the sublayer output before residual addition, initialized to 1.0 (identity). Two scalars per layer (one for attention, one for MLP) = 20 total.
**Subsystem:** residuals
**Papers:** GPAS (Chen et al. 2025, "Gradient-Preserving Activation Scaling") x ReZero (Bachlechner et al. 2020) adapted for Pre-LN setting
**Closest prior art:** ReZero initializes at 0.0 (which is wrong for Pre-LN -- it would zero out all sublayers). GPAS uses stop-gradient which may conflict with torch.compile. Our variant initializes at 1.0 (neutral) and omits stop-gradient, making it a pure learnable scaling. The existing resid_lambdas scale the FULL residual (x), not the sublayer OUTPUT -- this is mechanistically distinct.

**Success criteria (all must be met for CONFIRMED):**
- val_bpb delta < -0.003
- VRAM < 76 GB
- Wall-clock < 1200s
- No NaN/Inf

**INCONCLUSIVE if:** delta between -0.003 and +0.001
**REFUTED if:** delta > +0.001 or CRASH

**Predicted delta:** -0.003 +/- 0.002
**Predicted confidence:** low-medium
**Case for failure:** The existing resid_lambdas and x0_lambdas already provide per-layer scaling control on the residual stream. Adding per-sublayer output scaling may be redundant -- the optimizer can achieve similar effects by jointly adjusting resid_lambdas and the sublayer's c_proj weights. Without the stop-gradient trick from GPAS, the scaling parameter's gradient may be dominated by the magnitude of activations rather than their quality, leading to poor optimization. At only 10 layers, the depth-dependent signal propagation issues that GPAS addresses may not be severe enough to yield measurable gains. This is a low-cost, high-information-gain experiment: if it works, it reveals that sublayer-level scaling control matters beyond what resid_lambdas provides; if it fails, it confirms that resid_lambdas are sufficient.
**Feasibility pre-flight:** 20 scalar params, zero VRAM overhead. torch.compile: trivial scalar multiplication, no graph changes. No interaction with Muon (scalars use AdamW). Implementation is ~10 lines of code.
**Implementation sketch:** Add self.attn_scale = nn.Parameter(torch.ones(1)) and self.mlp_scale = nn.Parameter(torch.ones(1)) to Block.__init__. In Block.forward: x = x + self.attn_scale.to(x.dtype) * self.attn(norm(x), ...) and x = x + self.mlp_scale.to(x.dtype) * self.mlp(norm(x)). Add the 20 scale params to AdamW scalar_lr group in setup_optimizer.

---

## Priority Ranking (by expected information gain)

1. **H5 (xIELU + attn_temp)** -- Directly answers Open Question #1, low risk, high expected delta
2. **H6 (SwiGLU + attn_temp)** -- Directly answers Open Question #1, provides comparison with H5
3. **H9 (sublayer output scaling)** -- Cheap experiment, untested subsystem, high info gain per VRAM dollar
4. **H8 (hyper-connections)** -- Novel residual mechanism, untested subsystem, but VRAM risky
5. **H7 (xIELU-GLU + attn_temp)** -- Most speculative, only run if budget permits after H5/H6

---

## Cycle 1 Archive

### H1: xIELU Activation Function
- **Status**: CONFIRMED (delta = -0.003633)
- **Subsystem**: activation/function-choice

### H2: Peri-LN Post-Sublayer RMS Normalization
- **Status**: REFUTED (delta = -0.001184, VRAM +6.4 GB)
- **Subsystem**: normalisation/post-sublayer

### H3: SwiGLU Gated MLP
- **Status**: CONFIRMED (delta = -0.003363)
- **Subsystem**: MLP/gating-structure

### H4: Learnable Per-Head Attention Temperature
- **Status**: CONFIRMED (delta = -0.004839)
- **Subsystem**: attention/temperature-scaling

### H5 (old): xIELU + Peri-LN Combined
- **Status**: REJECTED (pre-empted; H2 was refuted)
- **Subsystem**: activation + normalisation
