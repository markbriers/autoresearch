# Evaluations

## Evaluation: Hypothesis H10 -- SwiGLU MLP

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.014435)
- VRAM < 76 GB: FAIL (actual: 79.1 GB)
- Wall-clock < 1200s: FAIL (actual: 1261.2s)
- No NaN/Inf: PASS

**Verdict: REFUTED**

**Prediction calibration:** predicted -0.006 at medium confidence, actual -0.014435. Directionally correct. Magnitude error: 2.4x too pessimistic. The val_bpb improvement was substantial, but the intervention failed on infrastructure constraints (VRAM +11.3 GB over baseline, wall-clock +38% over baseline). The extra gate projection (12x vs 8x n_embd params) was underestimated in the feasibility pre-flight which claimed "2-3GB" VRAM increase; actual increase was 11.3 GB.

**Evaluator notes:** SwiGLU produces the largest val_bpb improvement seen in this research programme (-0.014435), but at prohibitive resource cost. The Researcher's feasibility pre-flight was dangerously wrong on VRAM (predicted +2-3 GB, actual +11.3 GB). A dimension-reduced SwiGLU (e.g., hidden_dim = 2.67*n_embd to match 8x parameter count) could recover this benefit within budget. This is a resource-constrained refutation, not a mechanism refutation.

---

## Evaluation: Hypothesis H12 -- Learned Soft-Shrinkage Activation (ShrinkReLU)

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.005798)
- VRAM < 76 GB: PASS (actual: 66.2 GB)
- Wall-clock < 1200s: PASS (actual: 988.3s)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

**Prediction calibration:** predicted -0.004 at medium confidence, actual -0.005798. Directionally correct. Magnitude error: 1.4x too pessimistic. Prediction was reasonably well-calibrated.

**Evaluator notes:** Clean confirmation. 10 scalar parameters (one per layer) produce a meaningful -0.005798 improvement with zero resource overhead. This extends the pattern from v6: learnable scaling/threshold with identity-adjacent init is the winning formula. ShrinkReLU adds only a learned positive shift to the existing ReluSquared, staying within the "minimal parametric extension" paradigm. This is the first CONFIRMED result in this cycle.

---

## Evaluation: Hypothesis H13 -- Divisive Normalization in Attention Output

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: -0.002900, missed threshold by 0.0001)
- VRAM < 76 GB: PASS (actual: 69.4 GB)
- Wall-clock < 1200s: PASS (actual: 1003.0s)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

**Prediction calibration:** predicted -0.003 at low confidence, actual -0.002900. Directionally correct. Magnitude error: within range (0.97x). Prediction was well-calibrated but the predicted value was itself at the threshold boundary, which is a risky prediction strategy.

**Evaluator notes:** Missed the -0.003 threshold by only 0.0001. The mechanism (adaptive gain control on attention output) shows a real but small effect. The reformulation from logit normalization to output normalization weakened the original hypothesis, as noted in the pre-run evaluator caveat. The identity-init (sigma=0) passthrough design is sound. This is borderline enough to warrant a follow-up consideration (see B.6).

---

## Evaluation: Hypothesis H16 -- Learned RMSNorm Gain (Affine RMSNorm)

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: +0.135743, massive regression)
- VRAM < 76 GB: PASS (actual: 67.9 GB)
- Wall-clock < 1200s: PASS (actual: 733.2s)
- No NaN/Inf: PASS

**Verdict: REFUTED**

**Prediction calibration:** predicted -0.005 at medium confidence, actual +0.135743. Directionally WRONG. This is a catastrophic misprediction -- not merely wrong magnitude but wrong sign. The Researcher predicted improvement; the result was a +0.136 regression, one of the worst results in the programme.

**Evaluator notes:** The most likely cause is optimizer misconfiguration. The 13,440 per-dimension gain parameters were trained with scalar_lr (0.5), which is calibrated for single scalar parameters (10 total, e.g., ShrinkReLU tau). Per-dimension gains at lr=0.5 would wildly overshoot within the first few hundred steps, destabilizing all downstream computation. LLaMA-class models use RMSNorm gains with carefully tuned LR (typically 1e-3 to 1e-2, NOT 0.5). This is an optimizer configuration failure, but the contract is REFUTED regardless of cause.

This result refines the B1 pattern: "minimal parametric extension with identity init" works ONLY for scalar-per-layer parameters (10 params), NOT for per-dimension parameters (13,440 params) when using the scalar optimizer LR. The safe zone is explicitly: O(n_layer) or O(n_head) parameters, not O(n_embd * n_layer).

---

## Evaluation: Hypothesis H15 -- Dimension-Reduced SwiGLU (hidden_dim=1728)

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.005781)
- VRAM < 76 GB: PASS (actual: 68.1 GB)
- Wall-clock < 1200s: PASS (actual: 1017.3s)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

**Prediction calibration:** predicted -0.008 at medium-high confidence, actual -0.005781. Directionally correct. Magnitude error: 1.4x too optimistic. The Researcher overestimated the benefit of dimension-reduced SwiGLU.

**Evaluator notes:** CONFIRMED per contract, but the programme-level implication is nuanced. H15 (-0.005781) is statistically indistinguishable from ShrinkReLU H12 (-0.005798). The difference is 0.000017 -- well within seed variance (~0.002). Since H15 replaces ShrinkReLU (they cannot coexist), the programme should RETAIN ShrinkReLU because: (1) ShrinkReLU uses 988s wall-clock vs H15's 1017s (30s faster), (2) ShrinkReLU uses 66.2 GB VRAM vs H15's 68.1 GB (1.9 GB less), (3) ShrinkReLU preserves the original MLP architecture, leaving more headroom for future stacking experiments. H15 confirms that SwiGLU at 2.7x hidden_dim is a viable mechanism, but it does not improve upon the incumbent.

**Programme directive:** Revert to ShrinkReLU as the active MLP mechanism. H15's confirmation validates the SwiGLU mechanism at reduced dimension but does not justify replacing ShrinkReLU.

---

## Subsystem Tracker

| Subsystem | Tested | Confirmed | Refuted/Inconclusive | Status |
|-----------|--------|-----------|---------------------|--------|
| MLP/gated-activation | H10, H15 | 1 (H15) | 1 (H10 resource) | OPEN |
| MLP/activation-threshold | H12 | 1 (H12) | 0 | OPEN |
| MLP+attention/stacking | H6 (cycle 2) | 0 | 1 | OPEN (caution) |
| attention/gain-control | H13 | 0 | 1 (inconclusive) | OPEN |
| attention/additive-params | -- | -- | -- | OPEN |
| attention/gating | -- | -- | -- | OPEN |
| residuals/scaling | H9 (cycle 2) | 0 | 1 | OPEN |
| normalisation/per-dim-gain | H16 | 0 | 1 (catastrophic regression) | OPEN (see notes) |
| activation/function-choice | -- | -- | -- | OPEN |
| positional | -- | -- | -- | OPEN |
| embeddings | -- | -- | -- | OPEN |
| training-loop | -- | -- | -- | OPEN |

**Notes on normalisation/per-dim-gain:** H16's failure is likely an optimizer LR misconfiguration (scalar_lr=0.5 applied to 13,440 per-dimension params), not necessarily a mechanism failure. The subsystem remains OPEN but any retry MUST use a dedicated Adam group with lr in the range 1e-3 to 1e-2, NOT the scalar_lr. This is a single failure so does not trigger BLOCKED.

## Calibration Notes

**As of cycle 3:** Across 3 predictions this cycle, the Researcher was directionally correct on all 3 (3/3). Magnitude calibration: H10 predicted -0.006, actual -0.014 (2.4x too pessimistic); H12 predicted -0.004, actual -0.006 (1.4x too pessimistic); H13 predicted -0.003, actual -0.0029 (1.03x, essentially correct). The Researcher is systematically too pessimistic on magnitude by ~1.6x on average, meaning actual improvements tend to be larger than predicted. However, the Researcher is also systematically overoptimistic on resource costs -- H10's VRAM was underestimated by 4x. The Researcher should: (1) increase predicted deltas by ~50% for mechanism estimates, (2) increase VRAM estimates by 3-4x for interventions adding new nn.Linear layers.

**As of cycle 4:** Two new predictions evaluated. Running totals: 5 predictions across cycles 3-4.

- H15: predicted -0.008 (medium-high), actual -0.005781. Directionally correct. 1.4x too optimistic.
- H16: predicted -0.005 (medium), actual +0.135743. Directionally WRONG. Catastrophic misprediction.

**Cumulative calibration (cycles 3-4):** 4/5 directionally correct (80%). The one directional failure (H16) was a catastrophic regression, not a mild miss. The Researcher's previous bias was too pessimistic on val_bpb; H15 shows a new pattern of being too optimistic (1.4x). The average magnitude error excluding H16 is ~1.6x (mix of optimistic and pessimistic). Including H16, average directional accuracy drops to 80%.

**Key lesson from H16:** The Researcher failed to identify a critical optimizer configuration risk. Per-dimension parameters (13,440 at lr=0.5) are fundamentally different from per-layer scalars (10 at lr=0.5). The feasibility pre-flight discussed torch.compile and Muon compatibility but completely missed the LR mismatch. The Researcher MUST: (1) whenever adding parameters to a new optimizer group, explicitly state the LR and justify it relative to the parameter count and dimensionality, (2) never reuse scalar_lr for parameter groups with more than O(n_layer) parameters.

## Pivot Directives

No subsystems have reached the 3-failure BLOCKED threshold.

**Advisory (UPDATED cycle 4):** MLP/gated-activation now has one CONFIRMED (H15) and one resource-refuted (H10). The mechanism is validated. However, dim-reduced SwiGLU does not outperform ShrinkReLU, so further MLP/gated-activation work has diminishing returns unless a novel variant is proposed.

**Advisory (not BLOCKED):** attention/gain-control (H13) showed a real but sub-threshold effect. The reformulation from logit-space to output-space may have diluted the mechanism. Further attention modifications remain OPEN.

**Advisory (not BLOCKED):** normalisation/per-dim-gain (H16) had a catastrophic regression likely due to optimizer LR misconfiguration. If retried, MUST use lr in range 1e-3 to 1e-2 (not scalar_lr=0.5). The mechanism itself (RMSNorm gain) is standard in LLaMA-class models and should not be written off from one misconfigured experiment.

## Follow-Up Authorisations

**NO FOLLOW-UP AUTHORISED for H13.** (cycle 3 decision, retained)

**ENCOURAGED (cycle 3, now RESOLVED):** Dimension-reduced SwiGLU was tested as H15 and CONFIRMED at -0.005781, but does not outperform ShrinkReLU. This encouragement is now closed.

**NO FOLLOW-UP AUTHORISED for H15.** CONFIRMED per contract but does not improve upon incumbent ShrinkReLU. No variant warranted.

**NO FOLLOW-UP AUTHORISED for H16.** Although the failure is likely due to LR misconfiguration rather than mechanism failure, the programme should not spend another engineering run on a known technique (RMSNorm gain) when the fix is obvious (lower the LR). If the Researcher chooses to retry with correct LR, it should compete for a hypothesis slot on its own merits, not receive special follow-up status.
