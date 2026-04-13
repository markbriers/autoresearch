# Findings

## Confirmed Mechanisms

### F1: ShrinkReLU -- Learned Threshold Shift on ReluSquared (H12, Cycle 3)

**What:** Adding a single learned positive threshold tau per layer to shift the ReLU activation point -- `relu(x - tau).square()` with `tau = softplus(raw_tau)` -- yields -0.005798 val_bpb improvement at zero resource cost (10 scalar params total).

**Why it works:** This is the minimal parametric extension to the existing activation. It preserves the ReluSquared structure (sparsity, gradient properties) while allowing each layer to adapt its denoising threshold. The signal processing interpretation (Donoho-Johnstone soft thresholding) is apt: small activations below tau are zeroed, reducing noise propagation without affecting strongly activated features.

**Relation to prior confirmed mechanisms:** Extends the v6 pattern where "learnable scaling with identity/near-passthrough init" succeeded. ShrinkReLU initializes with tau ~0.13 (small threshold, close to original ReLU at 0), maintaining near-identity startup. The common thread: tiny learnable modifications to existing operations, initialized to preserve baseline behavior, with clear gradient paths.

**Stacking potential:** High. Zero VRAM/wall-clock overhead. Compatible with any MLP-external modification. Specifically, if a dimension-reduced SwiGLU is attempted in future, ShrinkReLU's threshold mechanism would need to be replaced by SwiGLU's gating, so they do not stack.

### F2: Dimension-Reduced SwiGLU (H15, Cycle 4) -- Confirmed but Not Retained

**What:** SwiGLU with hidden_dim=1728 (2.7x n_embd, matching the 8x total MLP parameter budget) yields -0.005781 val_bpb improvement. VRAM 68.1 GB, wall-clock 1017.3s -- both within budget.

**Why it works:** SwiGLU's soft gating mechanism (SiLU gate * linear value) provides smoother gradient flow and more expressive feature selection than ReluSquared's hard threshold. Even at reduced dimension, the gating interaction compensates for the 33% fewer hidden units.

**Why it is not retained:** The result is statistically indistinguishable from ShrinkReLU H12 (-0.005798 vs -0.005781, delta = 0.000017). ShrinkReLU is preferred because it uses 30s less wall-clock, 1.9 GB less VRAM, and preserves the original MLP architecture for future stacking.

**Relation to prior mechanisms:** Validates the SwiGLU mechanism from H10 (which achieved -0.014 at full dimension but was resource-refuted). The dimension reduction from 4x to 2.7x sacrificed ~60% of the quality benefit while bringing resources within budget. This suggests the SwiGLU benefit scales roughly linearly with hidden dimension.

---

## Dead Ends

### D1: SwiGLU at Full Hidden Dimension (H10, Cycle 3) -- Resource-Constrained Refutation

**What:** SwiGLU with hidden_dim = 4*n_embd (12x n_embd total MLP params) produced the best val_bpb improvement ever observed (-0.014435) but exceeded both VRAM (79.1 vs 76 GB) and wall-clock (1261.2s vs 1200s) limits. Prior cycle 2 tested SwiGLU stacked with attention temperature (H6) which also failed, but due to destructive interference rather than resources.

**Why it failed the contract:** The feasibility pre-flight underestimated VRAM by 4x (predicted +2-3 GB, actual +11.3 GB). The 50% parameter increase drives proportional increases in optimizer state (Muon stores momentum), activations, and gradient buffers. The 38% wall-clock increase comes from the extra matrix multiply per MLP layer (gate projection).

**What this tells us:** SwiGLU as a mechanism is extremely effective for this architecture (-0.014435 is 2.5x the second-best result). The failure is purely resource-based. A dimension-reduced variant (hidden_dim ~ 2.67*n_embd to match 8x total params) or a knowledge-distillation approach could recover part of this benefit within budget.

**Classification:** NOT a dead end mechanistically. The MLP/gated-activation subcategory remains OPEN. The specific configuration "SwiGLU at full 4x hidden dim" is infeasible.

### D3: RMSNorm Per-Dimension Gain at scalar_lr (H16, Cycle 4) -- Optimizer Misconfiguration

**What:** Adding learnable per-dimension gain vectors (init ones) to all 21 RMSNorm sites produced a catastrophic +0.136 val_bpb regression. Training converged to loss ~3.07 (vs expected ~2.7-2.8). No NaN/Inf, so the model trained but to a terrible minimum.

**Why it failed:** The 13,440 per-dimension gain parameters were placed in an Adam group with lr=0.5 (the scalar_lr). This LR is calibrated for O(10) scalar parameters (e.g., ShrinkReLU tau), not O(13K) per-dimension vectors. At lr=0.5, the gain vectors would have been updated by ~0.5 per step in early training, rapidly destabilizing the normalization and corrupting all downstream computation. LLaMA-class models that successfully use RMSNorm gain train them with lr in the range 1e-3 to 1e-2.

**What this tells us:** The "minimal parametric extension with identity init" pattern (B1) has a critical boundary condition: it works for O(n_layer) scalar parameters but breaks for O(n_embd * n_layer) per-dimension parameters when using scalar_lr. The pattern should be refined: identity init is necessary but not sufficient; the LR must also be appropriate for the parameter count and dimensionality.

**Classification:** Ambiguous -- the mechanism (RMSNorm gain) is well-established and almost certainly works with correct LR. The experiment tested the mechanism under a broken optimizer configuration. If retried with lr ~0.01, it would likely confirm. However, the contract is REFUTED regardless.

### D2: MLP + Attention Stacking Shows Destructive Interference (H6, Cycle 2)

**What:** SwiGLU + attention temperature stacking was refuted in cycle 2. The individual mechanisms may each help, but combining MLP activation changes with attention modifications in a single experiment makes it impossible to attribute effects and risks destructive interference.

**Lesson:** Test one subsystem at a time. H10 (SwiGLU alone) validated the MLP mechanism in isolation.

---

## Architecture Inductive Biases

### B1: The "Minimal Parametric Extension with Identity Init" Pattern

All successful or near-successful interventions share these properties:
1. Add very few parameters -- specifically O(n_layer) or O(n_head) scalars, NOT O(n_embd * n_layer) per-dimension vectors
2. Initialize to preserve baseline behavior (identity, zero, or near-passthrough)
3. Modify an existing operation rather than adding a new computational path
4. Use element-wise operations that torch.compile handles trivially
5. (NEW, cycle 4) Use an LR appropriate to the parameter count: scalar_lr (0.5) is safe for O(10) params; per-dimension params need lr ~0.01 or lower

Confirmed examples: ShrinkReLU (H12: 10 scalar params, tau init near zero). Dim-reduced SwiGLU (H15: confirmed but replaces rather than extends, not retained).
Near-miss: Divisive normalization (H13: sigma init at zero = identity passthrough, delta -0.0029).

Violations that failed: Full SwiGLU (H10: added entire new projection matrix, not minimal). Sublayer scaling (H9, cycle 2: details in prior cycle). RMSNorm gain (H16: 13,440 per-dimension params at scalar_lr=0.5, catastrophic regression -- violated properties 1 and 5).

### B2: Resource Feasibility Pre-Flights Are Systematically Underestimated

H10's VRAM was underestimated by 4x (predicted +2-3 GB, actual +11.3 GB). Any intervention adding a new nn.Linear layer must account for: (1) parameter storage, (2) Muon optimizer momentum state (same size as params), (3) activation storage for backward pass, (4) gradient storage. Rule of thumb: a new nn.Linear(d_in, d_out, bias=False) adds ~4 * d_in * d_out * bytes_per_param in VRAM at peak.

---

## Cross-Domain Transfer Patterns

### X1: Donoho-Johnstone Soft Thresholding Maps Directly to Activation Design

The signal processing concept of soft thresholding (shrink small coefficients, preserve large ones) translated directly into a confirmed activation improvement (ShrinkReLU). The key mapping: MLP hidden activations = coefficients, learned tau = noise-adaptive threshold, squaring after threshold = sparsity promotion. This is a clean cross-domain transfer where the mathematical operation (soft thresholding) was applied almost literally.

### X2: Neuroscience Divisive Normalization Shows Weak Transfer to Attention

Carandini-Heeger divisive normalization applied to attention output (H13) produced a real but sub-threshold effect (-0.0029). The reformulation from logit normalization to output normalization (forced by FlashAttention3's opaque computation) may have weakened the transfer. The original neuroscience mechanism operates on responses before competition (analogous to pre-softmax logits), not after.

---

## Open Questions

1. ~~Can dimension-reduced SwiGLU (matching 8x param budget) retain a meaningful fraction of the -0.014 improvement within VRAM/wall-clock limits?~~ RESOLVED (cycle 4): H15 confirmed at -0.005781 but does not outperform ShrinkReLU. SwiGLU at 2.7x retained ~40% of full-dim benefit.
2. Does ShrinkReLU's learned tau converge to different values across layers, and if so, what does the layer-wise threshold profile tell us about noise structure?
3. Would divisive normalization applied to Q/K before the dot product (rather than attention output) produce a stronger effect, despite partial redundancy with QK-norm?
4. (NEW, cycle 4) Would RMSNorm gain with a properly calibrated LR (~0.01) confirm, and if so, does it stack with ShrinkReLU? H16's failure was almost certainly an optimizer misconfiguration, not a mechanism failure.
5. (NEW, cycle 4) The programme now has two confirmed MLP mechanisms (ShrinkReLU, dim-reduced SwiGLU) that produce identical results via completely different mechanisms (threshold shift vs gated activation). Why do they converge to the same delta? Is -0.006 a ceiling for single-subsystem MLP modifications at this model scale?
