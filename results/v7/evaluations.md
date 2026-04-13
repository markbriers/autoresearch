# Evaluations

## Evaluation: Hypothesis 1 -- xIELU Activation Function

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.003633)
- VRAM < 76000 MB: PASS (actual: 67805.5 MB)
- Wall-clock < 1200s: PASS (actual: 1059.8s)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

**Prediction calibration:** predicted < -0.003 at moderate confidence, actual -0.003633. Directionally correct. Magnitude: prediction was conservatively stated as a threshold rather than a point estimate, so calibration is acceptable. The result just clears the bar by 0.000633.

**Evaluator notes:** xIELU replicates the v6 finding. The mechanism is well-understood: learnable positive quadratic + negative gradient flow outperforms the hard-zero ReluSquared. VRAM unchanged, wall-clock normal. Clean confirmation.

---

## Evaluation: Hypothesis 2 -- Peri-LN Post-Sublayer RMS Normalization

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: -0.001184)
- VRAM < 72000 MB: FAIL (actual: 74224.2 MB)
- Wall-clock < 1200s: PASS (actual: 993.8s)
- No NaN/Inf: PASS

**Verdict: REFUTED**

**Prediction calibration:** predicted < -0.003, actual -0.001184. Directionally correct but magnitude 2.5x too optimistic. The VRAM prediction was also wrong -- the contract set a 72 GB ceiling, but actual usage was 74.2 GB (6.4 GB above baseline), indicating the Researcher underestimated the cost of storing additional normalization activations for the backward pass.

**Evaluator notes:** Two criteria failed. The val_bpb improvement exists but is less than half the required threshold. The VRAM blowup (6.4 GB above baseline) is concerning -- this consumes most of the remaining headroom. The hypothesis noted the risk that Peri-LN might interact poorly with the existing resid_lambdas/x0_lambdas scaling scheme, and this appears to have materialized: the existing residual magnitude control already partially solves the problem Peri-LN addresses, leaving diminishing returns. This is not a borderline result -- it missed the val_bpb threshold by 61% and blew the VRAM budget by 2.2 GB. No follow-up warranted.

---

## Evaluation: Hypothesis 3 -- SwiGLU Gated MLP

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.003363)
- VRAM < 76000 MB: PASS (actual: 67159.3 MB)
- Wall-clock < 1200s: PASS (actual: 1056.0s)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

**Prediction calibration:** predicted < -0.003, actual -0.003363. Directionally correct. Magnitude within range. Slightly better than threshold.

**Evaluator notes:** SwiGLU replicates the v4 finding. The parameter-matched design (hidden_dim=1664) kept VRAM slightly below baseline despite the three-matrix structure. Wall-clock comparable. Clean confirmation. Note: H1 (xIELU) and H3 (SwiGLU) both modify the MLP activation/structure. They may not stack -- xIELU replaces the activation function, while SwiGLU replaces the entire MLP architecture including the activation (SiLU gating). These are likely mutually exclusive interventions. The Researcher must choose one or design a combined variant (e.g., xIELU-gated GLU) if stacking is desired.

---

## Evaluation: Hypothesis 4 -- Learnable Per-Head Attention Temperature

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.004839)
- VRAM < 70000 MB: PASS (actual: 67804.2 MB)
- Wall-clock < 1200s: PASS (actual: 978.0s)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

**Prediction calibration:** predicted < -0.003, actual -0.004839. Directionally correct. This is the strongest result in the cycle, exceeding the threshold by 61%. The Researcher underestimated the benefit.

**Evaluator notes:** Strongest result this cycle. The initial crash (float32 dtype mismatch from temperature parameter upcasting q) was caught and fixed -- this is a known torch pattern but worth noting for future scalar-parameter interventions: always cast scalar params to match tensor dtype before multiplication. The 60 temperature scalars (6 heads x 10 layers, not 50 as stated in the hypothesis -- the hypothesis said 5 heads, but the model has 6) add negligible overhead. This is a clean, high-value intervention: per-head attention sharpness control on top of QK-norm. The mechanism is distinct from all other confirmed results (activation subsystem vs attention subsystem), so stacking potential is high.

---

## Subsystem Tracker

| Subsystem | Tested | Confirmed | Refuted/Inconclusive | Status |
|-----------|--------|-----------|----------------------|--------|
| activation/function-choice | H1 | 1 | 0 | OPEN |
| normalisation/post-sublayer | H2 | 0 | 1 | OPEN |
| MLP/gating-structure | H3 | 1 | 0 | OPEN |
| attention/temperature-scaling | H4 | 1 | 0 | OPEN |
| residuals | -- | 0 | 0 | OPEN |
| positional | -- | 0 | 0 | OPEN |
| embeddings | -- | 0 | 0 | OPEN |
| training-loop | -- | 0 | 0 | OPEN |

## Calibration Notes

**As of cycle 1:** The Researcher's predicted deltas were stated as thresholds (< -0.003) rather than point estimates, making calibration coarse. Out of 4 predictions, 3 were directionally correct and met threshold. H2 was directionally correct but 2.5x too optimistic in magnitude and also underestimated VRAM cost by ~6.4 GB. H4 was 1.6x better than threshold, suggesting the Researcher underestimated attention-subsystem interventions. Overall: slight optimism on normalisation, slight pessimism on attention. The Researcher should provide point estimates (not just threshold claims) in future cycles for better calibration tracking.

## Pivot Directives

No pivot directives issued. Cycle 1 has insufficient data for any subsystem to reach 3 failures. All subsystems remain OPEN.

## Follow-Up Authorisations

No follow-ups authorised. H2 (Peri-LN) is the only non-confirmed result and it failed both val_bpb and VRAM thresholds -- this is not a borderline case warranting a second attempt.
