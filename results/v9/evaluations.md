# Evaluations

## Subsystem Tracker

| Subsystem | Tested | Confirmed | Refuted/Inconclusive | Status |
|-----------|--------|-----------|---------------------|--------|
| attention/temperature | H8 | 0 | 1 (REFUTED) | BLOCKED (see attention/all) |
| attention/normalization-type | H13 | 0 | 1 (REFUTED) | BLOCKED (see attention/all) |
| attention/all | H8, H13 | 0 | 2 (both REFUTED) | OPEN (1 more failure = BLOCKED) |
| activation/gating | H1, H6 | 1 (H6) | 1 (H1 INCONCLUSIVE) | OPEN |
| activation/normalization | H2 | 0 | 1 (REFUTED) | BLOCKED |
| residuals/derivative-scaling | H4 | 1 | 0 | OPEN |
| residuals/retrospective | H3, H7 | 0 | 2 (both INCONCLUSIVE) | BLOCKED |
| stacking/activation+residual | H10 | 0 | 1 (INCONCLUSIVE) | OPEN |
| stacking/with-factored-embeddings | H15, H19, H22 | 0 | 3 (H15 INCONCLUSIVE, H19 INCONCLUSIVE, H22 negative interaction) | BLOCKED |
| MLP/activation-fn | H1, H6 | 1 (H6) | 1 (H1 INCONCLUSIVE) | OPEN |
| MLP/mid-layer-norm | H2 | 0 | 1 (REFUTED) | BLOCKED |
| positional | -- | 0 | 0 | OPEN |
| normalisation/output-norm | H11 | 0 | 1 (INCONCLUSIVE) | OPEN (1 test) |
| embeddings/bottleneck-256 | H14 | 1 (H14) | 0 | OPEN |
| embeddings/bottleneck-192 | H16 | 0 | 1 (INCONCLUSIVE) | OPEN (1 test) |
| training-loop/weight-averaging | H12, H17 | 0 | 2 (H12 REFUTED, H17 REFUTED) | BLOCKED |

## Evaluation: Hypothesis 1 -- SwiGLU Activation

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: -0.002460)
- VRAM < 76 GB: PASS (actual: 65.6 GB)
- Wall-clock < 1200s: PASS (actual: 1082.7s)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

**Prediction calibration:**
- Researcher predicted -0.005 at medium confidence. Evaluator independently predicted P(success) = 0.45. Belief divergence was 0.10.
- Actual outcome: -0.002460. Directionally correct for both (both expected improvement), but magnitude fell short. Researcher was too optimistic by 2x.

**Surprisal score:** low. Both agents predicted directional improvement. The shortfall from the threshold is modest and within the range of reasonable outcomes. Neither agent's model is seriously wrong.

**Evaluator notes:** SwiGLU improved over baseline but not enough to clear the -0.003 threshold. The 35% reduction in MLP hidden dim (2560 -> 1664) at matched parameter count likely limited the gain. This is a borderline result worth a follow-up with adjusted hidden dim.

---

## Evaluation: Hypothesis 2 -- Divisive Normalization in MLP

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: +0.004629)
- VRAM < 76 GB: PASS (actual: 66.6 GB)
- Wall-clock < 1200s: PASS (actual: 1059.1s)
- No NaN/Inf: PASS

**Verdict: REFUTED**

**Prediction calibration:**
- Researcher predicted -0.003 at low confidence. Evaluator independently predicted P(success) = 0.15. Belief divergence was 0.10.
- Actual outcome: +0.004629. Directionally wrong for both -- neither predicted a regression. Researcher was more wrong (predicted improvement, got degradation).

**Surprisal score:** medium. Both agents expected this was unlikely to help, but neither predicted it would actively hurt. A +0.005 regression from a zero-parameter intervention is noteworthy -- it means divisive normalization is actively destructive to the learned representations, not merely neutral.

**Evaluator notes:** Divisive normalization after ReluSquared actively hurts. The mechanism is clear: ReluSquared produces highly sparse activations, and dividing by a local channel mean amplifies noise in near-zero regions while suppressing the few active channels. This destroys the sparse feature selection that ReluSquared provides. Cortical divisive normalization operates on dense firing rates, not sparse activations -- the biological analogy breaks down precisely at the sparsity boundary.

---

## Evaluation: Hypothesis 3 -- Predictor-Corrector Residual Connections

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: -0.000554)
- VRAM < 76 GB: PASS (actual: 69.3 GB)
- Wall-clock < 1200s: PASS (actual: 1019.9s)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

**Prediction calibration:**
- Researcher predicted -0.004 at low confidence. Evaluator independently predicted P(success) = 0.20. Belief divergence was 0.10.
- Actual outcome: -0.000554. Directionally correct for both but magnitude was 7x less than predicted. Researcher was significantly too optimistic.

**Surprisal score:** low. Both agents expected this was unlikely to work well. The Evaluator's low P(success) was closer to correct. With beta init=0, the safe initialization means the model barely departed from baseline, which is the expected failure mode.

**Evaluator notes:** The near-zero delta suggests the learnable betas stayed near zero throughout training -- the model found no benefit from inter-layer momentum. This is consistent with 10 layers being too few for ODE-style integration to matter. The predictor-corrector framing is more suited to deep networks (50+ layers) where accumulated truncation error is substantial.

---

## Evaluation: Hypothesis 4 -- PD Residual Scaling

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.004197)
- VRAM < 76 GB: PASS (actual: 69.3 GB)
- Wall-clock < 1200s: PASS (actual: 1016.9s)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

**Prediction calibration:**
- Researcher predicted -0.003 at low confidence. Evaluator independently predicted P(success) = 0.15. Belief divergence was 0.05.
- Actual outcome: -0.004197. Exceeded both predictions. Researcher was directionally correct and conservatively accurate. Evaluator was wrong -- predicted 85% chance of failure, got a clear confirmation.

**Surprisal score:** high. Both agents gave this low probability of success (Researcher P=0.20, Evaluator P=0.15), yet it is the strongest result in the batch. This reveals a gap in both agents' models: the derivative term in the residual stream provides meaningful signal even at depth=10, contrary to expectations. The "discrete derivative is too noisy" concern was wrong.

**Evaluator notes:** This is the most informative result of the batch. The PD controller mechanism works despite both agents predicting failure. The velocity term (x - x_prev) provides useful anticipatory correction even across just 10 layers. The very low learning rate for deriv_lambdas (scalar_lr * 0.01) may have been critical -- it constrains the derivative term to be small, avoiding the noise amplification that both agents feared. This is a clean confirmation of the control-theory-to-residual-stream transfer. Follow-up should investigate what values the deriv_lambdas converged to.

---

## Evaluation: Hypothesis 15 -- Triple Stack H6+H4+H14 (SwiGLU + PD Residual + Factored Embeddings)

**Sprint contract thresholds:**
- val_bpb delta < -0.006: FAIL (actual: -0.003087)
- VRAM < 76 GB: PASS (actual: 72.1 GB)
- Wall-clock < 1200s: PASS (actual: 1124.8s)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

The delta of -0.003087 falls within the INCONCLUSIVE band (-0.006 to -0.003) per the sprint contract. Critically, H15 (0.956253) is WORSE than H10 (0.953648, SwiGLU+PD without factored embeddings) by +0.002605. Adding factored embeddings to the SwiGLU+PD stack degraded performance. This is a negative interaction, not merely subadditivity.

**Prediction calibration:**
- Researcher predicted delta=-0.006 at medium confidence, P(success)=0.45. Evaluator independently predicted P(success)=0.30. Belief divergence was 0.15.
- Actual outcome: delta=-0.003087. Both agents predicted improvement but overestimated magnitude. The Researcher was 1.94x too optimistic. Neither agent predicted the negative interaction with factored embeddings.

**Surprisal score:** high. Neither agent anticipated that adding a CONFIRMED improvement (H14, factored embeddings) to a strong configuration (H10, SwiGLU+PD) would make it worse. The triple stack (0.956253) underperforms the double stack (0.953648) by a substantial margin. This is the first case where a confirmed intervention has a negative interaction with another confirmed intervention. Both agents' stacking models are fundamentally wrong -- the assumption that confirmed improvements compose monotonically is violated.

**Evaluator notes:** The result is unambiguous: factored embeddings hurt when combined with SwiGLU+PD. The mechanism is likely that SwiGLU's gated activation benefits from high-rank input representations. Factored embeddings compress the input through a 256-dim bottleneck, reducing the rank of features available for SwiGLU's gate to operate on. The PD derivative term also depends on the residual stream's layer-wise trajectory, which changes character when the initial representation is lower-rank. The 37% subadditive discount from H10 was already a warning; this negative interaction shows the discount can exceed 100% for certain combinations. H10 (SwiGLU+PD, val_bpb=0.953648) remains the best configuration.

---

## Evaluation: Hypothesis 19 -- SwiGLU + Factored Embeddings (H6+H14 stack, no PD)

**Sprint contract thresholds:**
- val_bpb delta < -0.005: FAIL (actual: -0.003075)
- VRAM < 76 GB: PASS (actual: 69.2 GB)
- Wall-clock < 1200s: PASS (actual: 1113.6s)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

The delta of -0.003075 falls within the INCONCLUSIVE band (-0.005 to -0.003) per the sprint contract. H19 (0.956265) is worse than H6 alone (0.954376) by +0.001889, confirming the negative interaction between SwiGLU and factored embeddings.

**Prediction calibration:**
- Researcher predicted delta=-0.005 at low confidence, P(success)=0.35. Evaluator independently predicted P(success)=0.25. Belief divergence was 0.10.
- Actual outcome: delta=-0.003075. Both agents predicted improvement but overestimated magnitude. Researcher was 1.63x too optimistic. Neither predicted negative interaction.

**Surprisal score:** high. Same pattern as H15 -- factored embeddings degrade SwiGLU's performance. H19 (SwiGLU+factored, 0.956265) is worse than H6 (SwiGLU alone, 0.954376). This confirms the negative interaction is specifically between SwiGLU and factored embeddings (not PD-related, since H19 has no PD).

**Evaluator notes:** The consistency between H15 and H19 is diagnostic. Both include SwiGLU + factored embeddings; both achieve ~0.956, which is approximately the performance of factored embeddings alone (H14, 0.956125). This means SwiGLU adds essentially nothing when factored embeddings are present, while SwiGLU alone adds -0.005. The factored embedding bottleneck appears to neutralize SwiGLU's gating benefit. The mechanism: SwiGLU's c_gate projection learns to select features based on the diversity of input representations. When the input comes through a 256-dim bottleneck, the feature diversity is reduced, and the gate cannot discriminate as effectively. The c_gate becomes redundant when input rank is constrained.

---

## Evaluation: Hypothesis 17 -- Tail-Phase EMA (beta=0.99, warmdown only)

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL -- BASELINE CONTAMINATED (actual reported: -0.003207, but this is from H14 factored embeddings already in codebase, NOT from EMA)
- VRAM < 76 GB: PASS (actual: 68.2 GB)
- Wall-clock < 1200s: PASS (actual: 1018.9s)
- No NaN/Inf: PASS

**BASELINE CONTAMINATION DETECTED:** The research log for Run 15 explicitly states: "The -0.003207 delta vs baseline (0.959340) is due to the codebase already including H14 factored embeddings (256-dim), not the EMA intervention." The last iterate (0.956133) is within noise of H14 alone (0.956125). The EMA weights (0.956294) were worse than the last iterate. The EMA intervention contributed delta = +0.000169 (slight degradation) relative to the H14-contaminated baseline, or effectively zero.

**Verdict: REFUTED**

The EMA intervention itself had zero or slightly negative effect. Both the last-iterate and EMA results are fully explained by the H14 factored embeddings that were already present in the codebase. The EMA did not improve over the last iterate (EMA 0.956294 > last iterate 0.956133).

**Prediction calibration:**
- Researcher predicted delta=-0.001 at low confidence, P(success)=0.15. Evaluator independently predicted P(success)=0.10. Belief divergence was 0.05.
- Actual outcome: EMA intervention delta ~0.000 (null effect). Both agents correctly predicted likely failure. The reported -0.003207 is an artifact of baseline contamination.

**Surprisal score:** low. Both agents predicted failure and got failure. The EMA-during-warmdown hypothesis confirmed the prior from H12: weight averaging is redundant with the warmdown schedule. The narrow beta=0.99 (100-step window) avoided the catastrophic failure of H12's beta=0.999, but still provided no benefit. The Evaluator's cycle 3 note was correct: "warmdown already provides implicit averaging."

**Evaluator notes:** The baseline contamination is a process failure. The implementer did not revert H14 before running H17. However, the contamination is transparent (the research log documents it) and does not affect the verdict: even if we compare against the H14 baseline (0.956125), the EMA result (0.956294) is worse. The training-loop/weight-averaging subsystem now has 2 failures (H12, H17) and 0 confirmations. Beta=0.999 (broad window) produced catastrophic failure. Beta=0.99 (narrow window, warmdown only) produced null effect. The warmdown schedule is the dominant averaging mechanism; explicit EMA adds nothing. This subsystem should be BLOCKED.

---

## Evaluation: Hypothesis 16 -- Factored Embeddings Bottleneck Dim Sweep (192-dim)

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: -0.000563)
- VRAM < 76 GB: PASS (actual: 67.9 GB)
- Wall-clock < 1200s: PASS (actual: 1013.6s)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

The delta of -0.000563 falls within the INCONCLUSIVE band (-0.003 to +0.001). The 192-dim bottleneck (0.958777) performed substantially worse than the 256-dim bottleneck (H14, 0.956125), a difference of +0.002652. This locates the rate-distortion knee between 192 and 256 dims.

**Prediction calibration:**
- Researcher predicted delta=-0.003 at low confidence, P(success)=0.30. Evaluator independently predicted P(success)=0.20. Belief divergence was 0.10.
- Actual outcome: delta=-0.000563. Both agents predicted improvement but overestimated magnitude. Researcher was 5.3x too optimistic. The Evaluator's calibration-adjusted estimate (delta -0.001 to -0.002) was closer but still too optimistic.

**Surprisal score:** low. The Evaluator's cycle 3 notes predicted this outcome: "If I apply the calibration correction, the actual delta may be closer to -0.001 to -0.002, which would be INCONCLUSIVE." The actual delta (-0.000563) was even weaker than the calibration-corrected estimate but directionally consistent. No model update required.

**Evaluator notes:** This is a clean parametric sweep result. The rate-distortion knee for token embeddings at 1800 training steps is between 192 and 256 dims. At 192-dim, too much token-discriminative information is lost; the bottleneck crosses from beneficial regularization into harmful information compression. The sharp performance cliff between 256 (-0.003215) and 192 (-0.000563) -- a difference of 0.002652 from just 64 fewer dimensions -- suggests the knee is steep. The optimal bottleneck_dim is likely in the 224-256 range. Further sweeps (e.g., 224-dim) would have low expected information gain since 256-dim is already CONFIRMED and the knee location is constrained.

---

## Calibration Notes

**As of cycle 1:** The Researcher's predicted deltas are consistently 1.5-7x too optimistic. Out of 4 predictions, 3 were directionally correct (predicted improvement, got improvement) but 0 were magnitude-accurate. The one directional miss (H2) was a severe miss: predicted -0.003, got +0.005. The Researcher should reduce predicted magnitudes by at least 50% and add explicit regression risk for non-learnable interventions.

**Surprisal summary (cycle 1):** Highest-surprisal result: H4 (PD Residual Scaling). Both agents predicted failure with high confidence; it clearly succeeded. This reveals that control-theory-inspired residual modifications are more effective than either agent's model anticipated. The derivative term's utility at shallow depth is the key unexpected finding.

**As of cycle 2:** Cycle 2 results: 1 CONFIRMED (H6), 1 INCONCLUSIVE (H7), 1 REFUTED (H8). Out of 3 predictions:
- H6: Researcher P=0.45, Evaluator P=0.35. Result: CONFIRMED. Both directionally correct (expected plausible success); Evaluator was less well-calibrated (was more sceptical of a genuine success). Predicted delta=-0.003, actual=-0.005. The Researcher underestimated the benefit, which is a departure from cycle 1's optimism pattern.
- H7: Researcher P=0.20, Evaluator P=0.12. Result: INCONCLUSIVE (failed). Both directionally correct (expected likely failure). Evaluator was better calibrated. Predicted delta=-0.002, actual=-0.0007. Researcher still 2.9x too optimistic on magnitude.
- H8: Researcher P=0.25, Evaluator P=0.20. Result: REFUTED (+0.005 regression). Both directionally wrong -- neither predicted a regression. Predicted delta=-0.002, actual=+0.005. Severity: 7 mBPB error (predicted -2, got +5). This is the worst prediction error in the programme.

**Cumulative calibration (7 experiments, cycles 1+2):**
- Directional accuracy: 5/7 experiments had directional agreement between predictions and outcome. The 2 misses (H2, H8) were both regressions that neither agent anticipated.
- Pattern: The Researcher has never predicted a regression. Both refuted hypotheses (H2: +0.005, H8: +0.005) produced regressions of similar magnitude. The Researcher should add explicit regression probability to future contracts. Interventions that modify intermediate representations (mid-MLP normalization, attention temperature) are the regression-prone category.
- Magnitude calibration: Still too optimistic on average. The Researcher's predicted deltas average 1.5-3x the actual deltas for experiments that improve.
- The Evaluator was better calibrated on H7 but worse on H6 (too sceptical). The Evaluator's systematic scepticism is broadly correct but can miss genuine improvements in well-motivated follow-ups.

**Surprisal summary (cycle 2):** Highest-surprisal result: H8 (Per-Head Attention Temperature). Both agents expected neutral-to-mild improvement; the result was a clear regression of similar magnitude to H2. This is the second-most informative result in the programme (after H4) because it reveals that the attention subsystem actively resists scalar modifications even when initialised at identity. The mechanism (QK-norm + softcap creating a rigid logit regime) was not part of either agent's model.

**As of cycle 3:** Cycle 3 results: 0 CONFIRMED-by-stacking (H10 INCONCLUSIVE), 1 CONFIRMED (H14), 2 REFUTED (H12, H13). Out of 4 predictions:
- H10: Researcher P=0.60, Evaluator P=0.55. Result: INCONCLUSIVE (failed stacking threshold by 0.000308). Both directionally correct (predicted improvement) but wrong about stacking magnitude. Predicted delta=-0.007, actual=-0.005692. Researcher 1.23x too optimistic.
- H12: Researcher P=0.30, Evaluator P=0.15. Result: REFUTED (+0.004 last-iterate, EMA catastrophic). Researcher was directionally wrong (predicted improvement, got regression). Evaluator was more sceptical but also did not predict regression. NOTE: the Researcher assigned P(regression)=0.05, which was far too low. The Evaluator warned about beta=0.999 being too wide.
- H13: Researcher P=0.15, Evaluator P=0.10. Result: REFUTED (+0.007). Both correctly predicted likely failure. Researcher assigned P(regression)=0.30, the highest in cycle 3 -- this was well-calibrated. Directionally correct for both (expected failure).
- H14: Researcher P=0.15, Evaluator P=0.08. Result: CONFIRMED (-0.003215). Both agents were wrong -- predicted failure, got success. Researcher predicted delta=-0.001, actual=-0.003215 (3.2x underestimate). HIGH SURPRISAL.

**Cumulative calibration (11 experiments, cycles 1-3):**
- Directional accuracy: 7/11 experiments had directional agreement between predictions and outcome. The 4 misses: H2 (regression, neither predicted), H8 (regression, neither predicted), H12 (regression, neither predicted; but Researcher assigned P(regression)=0.05 -- still wrong), H14 (improvement, both predicted failure).
- Pattern update: The Researcher now includes P(regression) estimates (new in cycle 3). For H13, P(regression)=0.30 was well-calibrated. For H12, P(regression)=0.05 was grossly underestimated. For H10, P(regression)=0.05 was correct (no regression). Overall, regression prediction is still inconsistent.
- Magnitude calibration: The Researcher's predicted deltas remain 1.2-3.2x too optimistic for experiments that show improvement (H10: 1.23x, H14: 0.31x -- the first time the Researcher UNDERESTIMATED a positive result). The H14 underestimate is anomalous and explains its high surprisal.
- The Evaluator's systematic scepticism was correct for H12 (Evaluator P=0.15 vs Researcher P=0.30 -- Evaluator was closer) but wrong for H14 (Evaluator P=0.08, the lowest prediction for a CONFIRMED result in the programme). The Evaluator must update: parameter-reduction interventions in embedding layers can work as regularizers even at 1800 steps.
- New pattern: the Evaluator has now been surprised by two results where "parameter reduction = regularization" was the mechanism (H4's derivative term was also a minimal-parameter addition that regularized the residual stream). The Evaluator's model undervalues interventions that reduce or minimally add parameters. This is a systematic bias to correct.

**Surprisal summary (cycle 3):** Highest-surprisal result: H14 (Factored Embeddings). Both agents predicted failure with high confidence (P=0.08-0.15) and the Researcher explicitly assigned P(regression)=0.25. The result was CONFIRMED. This reveals that the embedding layer is over-parameterized at this scale and that factored embeddings with Muon-optimized projection are a productive direction -- a finding neither agent anticipated. Second-highest surprisal: H10 (SwiGLU + PD stacking) -- the INCONCLUSIVE verdict (falling 0.000308 short) reveals that confirmed improvements interact subadditively at a 37% discount, which both agents underestimated.

**As of cycle 4:** Cycle 4 results: 1 INCONCLUSIVE (H11). Out of 1 prediction:
- H11: Researcher P=0.25, Evaluator P=0.20. Result: INCONCLUSIVE (failed). Both directionally correct (predicted likely failure). Predicted delta=-0.002, actual=-0.000522. Researcher 3.8x too optimistic on magnitude.

**Cumulative calibration (12 experiments, cycles 1-4):**
- Directional accuracy: 8/12 experiments had directional agreement between predictions and outcome. The 4 misses remain H2, H8, H12, H14 (unchanged from cycle 3).
- Magnitude calibration: The Researcher's predicted deltas remain systematically too optimistic. For INCONCLUSIVE/weak results, the overshoot is typically 2-4x (H3: 7.2x, H7: 2.9x, H11: 3.8x). For CONFIRMED results, predictions are more accurate (H4: 0.7x underestimate, H6: 1.7x underestimate, H14: 3.2x underestimate). The pattern: the Researcher overestimates weak interventions and underestimates strong ones.
- The Evaluator's systematic scepticism was correct for H11 (P=0.20 vs Researcher P=0.25). H11 is a low-surprisal result that reinforces the prior: interventions addressing deep-network problems at depth=10 produce negligible effects.

**Surprisal summary (cycle 4):** No high-surprisal results. H11 was the only experiment and both agents correctly predicted likely failure. The normalisation subsystem at depth=10 behaves exactly as expected. No model updates required.

**As of cycle 5:** Cycle 5 results: 0 CONFIRMED, 3 INCONCLUSIVE (H15, H19, H16), 1 REFUTED (H17). Out of 4 predictions:
- H15: Researcher P=0.45, Evaluator P=0.30. Result: INCONCLUSIVE (delta=-0.003087, needed -0.006). Both predicted improvement but overestimated magnitude. Researcher 1.94x too optimistic. Neither predicted the critical finding: factored embeddings DEGRADE the SwiGLU+PD stack.
- H19: Researcher P=0.35, Evaluator P=0.25. Result: INCONCLUSIVE (delta=-0.003075, needed -0.005). Both predicted improvement but overestimated magnitude. Researcher 1.63x too optimistic. Same negative interaction pattern as H15.
- H17: Researcher P=0.15, Evaluator P=0.10. Result: REFUTED (EMA delta ~0, baseline contaminated). Both correctly predicted likely failure. Low surprisal.
- H16: Researcher P=0.30, Evaluator P=0.20. Result: INCONCLUSIVE (delta=-0.000563, needed -0.003). Both predicted improvement but overestimated magnitude. Researcher 5.3x too optimistic.

**Cumulative calibration (16 experiments, cycles 1-5):**
- Directional accuracy: 11/16 experiments had directional agreement between predictions and outcome. The 5 misses: H2, H8, H12 (regressions neither predicted), H14 (improvement both predicted failure), and now the H15/H19 negative interaction (improvement predicted but smaller than expected due to unanticipated negative interaction).
- NEW PATTERN: The Researcher's stacking predictions are systematically wrong. H10 predicted -0.007, got -0.005692 (1.23x overestimate). H15 predicted -0.006, got -0.003087 (1.94x overestimate). H19 predicted -0.005, got -0.003075 (1.63x overestimate). The Researcher assumes confirmed improvements compose positively; the data shows they can interact negatively. The Researcher must model negative interactions explicitly for any future stacking experiments.
- Magnitude calibration: For weak interventions, the Researcher continues to overestimate by 2-5x (H16: 5.3x, H17: effectively infinite -- predicted improvement, got null). The pattern identified in cycle 4 is reinforced: the Researcher overestimates weak interventions and underestimates strong ones.
- The Evaluator's calibration-corrected predictions were closer for H16 (predicted delta -0.001 to -0.002, got -0.000563) but still too optimistic. The Evaluator must further discount parametric sweeps near the boundary of confirmed results.

**Surprisal summary (cycle 5):** Highest-surprisal results: H15 and H19 (joint). The discovery that factored embeddings DEGRADE performance when combined with SwiGLU is the most important finding of this cycle. Neither agent's model predicted negative interaction between two confirmed improvements. This reveals a fundamental gap in the stacking model: confirmed improvements are NOT guaranteed to compose positively. The interaction between input representation rank (factored embeddings) and activation gating quality (SwiGLU) is antagonistic. This is a model-updating result for both agents.

## Pivot Directives

**BLOCKED: residuals/retrospective** -- 2 failures (H3 INCONCLUSIVE delta=-0.0006, H7 INCONCLUSIVE delta=-0.0007), 0 confirmations. Rationale: both retrospective residual modifications (1-step momentum and exponential accumulation) produced negligible deltas. The pattern is clear: backward-looking information in the residual stream at depth=10 is not useful. Only anticipatory (derivative/velocity) modifications work. Do not propose further hypotheses that add terms based on accumulated past block outputs.

**BLOCKED: activation/normalization** -- 1 failure (H2 REFUTED delta=+0.005), 0 confirmations. Rationale: any normalization-style operation on sparse ReluSquared activations destroys the sparsity structure. This applies to divisive normalization, group normalization, local response normalization, and similar pool-then-divide schemes applied after ReluSquared. Note: this does NOT block normalization on dense activations (e.g., before attention, after the residual stream) -- only on sparse post-activation hidden states.

**BLOCKED: training-loop/weight-averaging** -- 2 failures (H12 REFUTED, H17 REFUTED), 0 confirmations. Rationale: H12 (beta=0.999, full training) produced catastrophic EMA failure and last-iterate regression. H17 (beta=0.99, warmdown only) produced null effect -- EMA was worse than last iterate. The warmdown schedule (70% of training, LR decays to 0.01x) already provides optimal implicit averaging. Explicit EMA adds nothing on top. Do not propose further weight-averaging hypotheses unless the training schedule fundamentally changes (e.g., no warmdown).

**BLOCKED: stacking/with-factored-embeddings** -- 2 failures (H15 INCONCLUSIVE, H19 INCONCLUSIVE), 0 confirmations. Rationale: Factored embeddings (H14) negatively interact with SwiGLU (H6). H15 (triple stack SwiGLU+PD+factored, 0.956253) is worse than H10 (SwiGLU+PD, 0.953648). H19 (SwiGLU+factored, 0.956265) is worse than H6 (SwiGLU, 0.954376). The mechanism: factored embeddings reduce input rank, which degrades SwiGLU's gating quality. Do not stack factored embeddings with SwiGLU.

**OPEN despite 2 failures: attention/all** -- H8 (temperature, +0.005 regression) and H13 (Linf norm, +0.007 regression) are both REFUTED. These are mechanistically different interventions (scalar temperature vs norm geometry change), yet both produced large regressions. The attention subsystem is approaching BLOCKED status. One more failure with 0 confirmations will trigger BLOCKED. The attention subcategories attention/temperature and attention/normalization-type are individually BLOCKED. Only structurally different attention interventions (e.g., head count, window pattern, attention pattern structure) remain testable. Any future attention hypothesis must explain why it avoids the failure modes of both H8 and H13.

## Exploration Directives

**HIGH SURPRISAL (explore further):** stacking/interaction-mechanisms -- H15 and H19 revealed that confirmed improvements can interact NEGATIVELY. This is the most model-updating finding in cycle 5. The Researcher should investigate WHY factored embeddings degrade SwiGLU performance. Specifically: does the negative interaction extend to other activation functions, or is it specific to gated activations that depend on input rank? The Researcher should also investigate whether H14 stacks with H4 alone (PD+factored, without SwiGLU) -- this combination has not been tested and would isolate whether the problem is SwiGLU-specific.

**MEDIUM SURPRISAL (continue exploring):** embeddings -- H14 (256-dim) is confirmed but H16 (192-dim) failed, locating the rate-distortion knee between 192 and 256 dims. The bottleneck dim space is now reasonably well-mapped. Further sweeps (224-dim) have diminishing returns. The more interesting question is why factored embeddings interact negatively with SwiGLU. Also: attention -- still OPEN with 2 failures, but only structural (non-logit-modifying) interventions remain.

**LOW SURPRISAL (diminishing returns):** residuals/retrospective -- BLOCKED, well understood. activation/normalization -- BLOCKED, well understood. training-loop/weight-averaging -- now BLOCKED after H17 confirmed H12's failure pattern. normalisation/output-norm -- depth=10 is too shallow, well understood.

## Follow-Up Authorisations

**FOLLOW-UP AUTHORISED:** Hypothesis 1 (SwiGLU). Reason: borderline INCONCLUSIVE result (delta=-0.00246, threshold=-0.003). Authorised variation: increase hidden_dim from 1664 to 1792 (nearest multiple of 128 above 8/3 * 640), accepting a modest parameter count increase (~3-5%). The mechanistic rationale is that the 35% width reduction from 2560 to 1664 was the binding constraint -- SwiGLU's gating quality was offset by insufficient hidden capacity. A wider hidden dim tests whether the gating benefit exceeds the width cost. Write as new sprint contract H6 with status PROPOSED.

**NO FOLLOW-UP:** Hypothesis 3 (Predictor-Corrector). The delta (-0.000554) is too far from the threshold to warrant a parameter tweak. The mechanistic comparison with H4 is more informative: retrospective momentum (H3) does not work; anticipatory velocity (H4) does. The predictor-corrector framing should be considered a weaker variant of PD scaling, not a separate avenue worth further investment.

---

# Cycle 2 Follow-Up Authorisations

**STACKING EXPERIMENT AUTHORISED: H6 (SwiGLU) + H4 (PD Residual Scaling).** Reason: Both are individually CONFIRMED with the two strongest deltas in the programme (H6: -0.005, H4: -0.004). They modify orthogonal subsystems (MLP activation vs residual stream dynamics). The expected stacked delta is -0.006 to -0.009 depending on interaction effects. This is the highest-value experiment available. Write as a new sprint contract with status PROPOSED. Success criterion: val_bpb delta < -0.006 (must exceed best individual result by at least 0.001 to demonstrate genuine stacking). VRAM estimate: ~69-70 GB (SwiGLU reduces activation memory from narrower hidden dim; PD adds one residual-sized tensor).

**NO FOLLOW-UP: Hypothesis 7 (Leaky Integral).** The delta (-0.0007) is far from the threshold and the mechanistic explanation is clear: retrospective accumulation is not useful at depth=10. This is now the second retrospective failure alongside H3. The residuals/retrospective subcategory is BLOCKED. No parameter tweak will fix this.

**NO FOLLOW-UP: Hypothesis 8 (Attention Temperature).** The regression (+0.005) is severe enough to warrant caution. However, the Evaluator flags an implementation concern (head_temps possibly mis-grouped with Muon). The Researcher should VERIFY the implementation before proposing any further attention hypotheses. If the implementation was correct, the attention/temperature subcategory should be considered exhausted for scalar-parameter modifications. If the implementation was buggy, a corrected re-run could be proposed -- but this is NOT authorised until verification is complete.

---

# Cycle 3 Follow-Up Authorisations

**NEW BASELINE DIRECTIVE:** H10 (SwiGLU + PD stacking, val_bpb=0.953648) should become the new baseline for future experiments, despite its INCONCLUSIVE verdict on the stacking threshold. It is the best configuration found in the programme. Future experiments should be measured against val_bpb=0.953648, not the original 0.959340. The success threshold remains delta < -0.003 from the new baseline (i.e., val_bpb < 0.950648).

**FOLLOW-UP AUTHORISED: Hypothesis 14 (Factored Embeddings).** Reason: HIGH SURPRISAL result -- both agents predicted failure, got confirmation. The mechanistic model is wrong in an interesting way. Authorised variation: stack H14 (factored embeddings, 256-dim) with H10 (SwiGLU + PD) as a triple-stack experiment. The rationale is that H14 reduces parameters (regularization benefit) while H10 adds complexity (stacking benefit), and these may interact positively -- the regularization from factored embeddings may help the stacked model. Success criterion: val_bpb < 0.950648 (delta < -0.003 from the H10 baseline). Write as a new sprint contract with status PROPOSED.

**FOLLOW-UP AUTHORISED: Hypothesis 14 bottleneck variation.** Reason: HIGH SURPRISAL -- the bottleneck_dim=256 was arbitrary and the result was borderline (delta=-0.003215, just above threshold). A different bottleneck_dim (128 or 384) could be substantially better or worse. Test bottleneck_dim=192 (halfway between 128 and 256) against the ORIGINAL baseline to map the bottleneck-performance curve. Success criterion: val_bpb delta < -0.003 from original baseline. Write as a new sprint contract with status PROPOSED. Note: this is a LOW PRIORITY follow-up -- the triple-stack experiment takes precedence.

**NO FOLLOW-UP: Hypothesis 10 (SwiGLU + PD stacking).** The INCONCLUSIVE verdict is accepted. The stacking is subadditive at 37% discount. No parameter tweak will change this -- the subadditivity is structural. The configuration becomes the new baseline, not a follow-up target.

---

# Cycle 6 Evaluations

## Evaluation: Hypothesis 22 -- PD Residual + Factored Embeddings Stack (H4+H14, no SwiGLU)

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.003336)
- VRAM < 76 GB: PASS (actual: 69.5 GB)
- Wall-clock < 1200s: PASS (actual: ~975s)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

The delta of -0.003336 clears the -0.003 threshold. All four criteria pass. However, the mechanical CONFIRMED verdict conceals a critical finding: H22 (PD+factored, 0.956004) is WORSE than H4 alone (PD only, 0.955143) by +0.000861. Adding factored embeddings to PD caused a regression of 0.86 mBPB relative to PD alone. The factored embeddings did not provide additive benefit on top of PD -- they degraded it.

The stacking arithmetic:
- H4 alone (PD): 0.955143 (delta -0.004197)
- H14 alone (factored): 0.956125 (delta -0.003215)
- Additive expectation: delta -0.007412
- Actual H22 (PD+factored): delta -0.003336
- Subadditive discount: 55% (compared to 37% for H10 SwiGLU+PD)
- Net effect of adding factored to PD: +0.000861 REGRESSION vs PD alone

This is the third experiment demonstrating that factored embeddings degrade stacked configurations:
- H15 (SwiGLU+PD+factored): worse than H10 (SwiGLU+PD) by +0.002605
- H19 (SwiGLU+factored): worse than H6 (SwiGLU) by +0.001889
- H22 (PD+factored): worse than H4 (PD) by +0.000861

The negative interaction is NOT SwiGLU-specific. Factored embeddings degrade every stacking partner tested.

**Prediction calibration:**
- Researcher predicted delta=-0.004 at medium confidence, P(success)=0.45. Evaluator independently predicted P(success)=0.35. Belief divergence was 0.10.
- Actual outcome: delta=-0.003336. Directionally correct for both (both expected improvement). Researcher was 1.20x too optimistic on magnitude. Both agents correctly predicted this would clear the threshold, but the Researcher's P(success)=0.45 was closer to the outcome than the Evaluator's 0.35. The Evaluator was too sceptical.

**Surprisal score:** medium. The mechanical CONFIRMED verdict was predicted by the Researcher (P=0.45) and plausible per the Evaluator (P=0.35). The surprise is in the INTERPRETATION: H22 passes the absolute threshold but factored embeddings degrade PD performance. The Evaluator's sprint contract note ("the truly informative comparison is against H4 alone") was prescient -- the absolute threshold masks a negative interaction. The Researcher's mechanistic hypothesis ("the negative SwiGLU-factored interaction is specific to gated activations") was WRONG. This is the key model-updating finding: factored embeddings are a universal stacking poison, not a SwiGLU-specific one.

**Evaluator notes:** This result resolves the central question of cycle 6: is the negative factored-embeddings interaction SwiGLU-specific? The answer is definitively NO. Factored embeddings degrade PD alone (+0.000861), SwiGLU alone (+0.001889), and SwiGLU+PD (+0.002605). The degradation is smaller for PD (+0.86 mBPB) than for SwiGLU (+1.89 mBPB), which is consistent with the original hypothesis that SwiGLU is more sensitive to input rank. But the direction is the same: factored embeddings hurt stacking universally.

The mechanism: the 256-dim embedding bottleneck constrains the rank of the initial residual stream representation. This affects ALL downstream components that benefit from high-rank features. PD's velocity signal (x - x_prev) is computed from the residual stream, which starts lower-rank due to the bottleneck. SwiGLU's gate discriminates based on input features, which are lower-rank. The bottleneck's regularization benefit (seen when factored embeddings are used alone) is overwhelmed by the information loss when combined with other improvements that exploit the full-rank residual stream.

Factored embeddings (H14) should be understood as a standalone regularizer that is incompatible with stacking. It works alone because the baseline model is over-parameterized in the embedding layer. But once other improvements are added (PD, SwiGLU), the model is no longer over-parameterized in the same way, and the bottleneck becomes a binding constraint.

---

## Calibration Notes

**As of cycle 6:** Cycle 6 results: 1 CONFIRMED (H22, delta=-0.003336). Out of 1 prediction:
- H22: Researcher P=0.45, Evaluator P=0.35. Result: CONFIRMED. Both directionally correct. Researcher was 1.20x too optimistic on magnitude (predicted -0.004, got -0.003336). The Evaluator was too sceptical (P=0.35 for a result that passed cleanly).

**Cumulative calibration (17 experiments, cycles 1-6):**
- Directional accuracy: 12/17 experiments had directional agreement between predictions and outcome. The 5 misses remain H2, H8, H12 (regressions neither predicted), H14 (improvement both predicted failure), and the H15/H19 negative interaction (improvement predicted but degraded by interaction).
- Magnitude calibration: The Researcher's predicted deltas are still systematically too optimistic. For H22, the prediction was 1.20x too optimistic -- the tightest prediction accuracy in the programme for a positive result. The Researcher's stacking predictions specifically: H10 1.23x, H15 1.94x, H19 1.63x, H22 1.20x. The Researcher is getting MORE accurate on stacking predictions over time (the 1.20x for H22 is the best stacking prediction to date). However, the Researcher's mechanistic model of WHY stacking works/fails was wrong for H22 -- predicting no negative interaction without SwiGLU.
- The Evaluator was correctly sceptical about stacking (P=0.35 for H22, lower than Researcher's 0.45) but the scepticism was insufficiently targeted: the Evaluator worried about PD velocity noise from lower-rank inputs, which was the right concern, but still predicted a plausible PASS.

**Surprisal summary (cycle 6):** The key finding is NOT the mechanical CONFIRMED verdict but the pattern it completes: factored embeddings are universally incompatible with stacking. The Researcher's hypothesis that the negative interaction was SwiGLU-specific was falsified. This is a MEDIUM surprisal result -- the absolute outcome was predictable (both agents expected improvement) but the interpretation is model-updating. The stacking/with-factored-embeddings subcategory should be permanently BLOCKED regardless of the activation function used.

## Pivot Directives

**BLOCKED: stacking/with-factored-embeddings** -- 3 failures (H15 INCONCLUSIVE, H19 INCONCLUSIVE, H22 CONFIRMED-but-negative-interaction), 0 stacking improvements. UPDATED from cycle 5. Rationale: Factored embeddings degrade EVERY stacking partner tested: SwiGLU (+1.89 mBPB), PD (+0.86 mBPB), SwiGLU+PD (+2.61 mBPB). The negative interaction is NOT SwiGLU-specific -- it is universal. The 256-dim bottleneck constrains residual stream rank, which degrades all components that exploit high-rank features. Do not stack factored embeddings with any other improvement. Factored embeddings (H14) remain valid as a STANDALONE improvement only.

All other pivot directives from cycle 5 remain unchanged:
- **BLOCKED: residuals/retrospective** -- 2 failures, 0 confirmations (unchanged)
- **BLOCKED: activation/normalization** -- 1 failure, 0 confirmations (unchanged)
- **BLOCKED: training-loop/weight-averaging** -- 2 failures, 0 confirmations (unchanged)
- **OPEN despite 2 failures: attention/all** -- H8 and H13 both REFUTED; one more failure = BLOCKED (unchanged)

## Exploration Directives

**HIGH SURPRISAL (explore further):** stacking/interaction-mechanisms -- H22 completes the picture. The stacking/with-factored-embeddings subcategory is now exhaustively tested (3 experiments, 3 negative interactions). The open question shifts: what CAN stack with H10 (SwiGLU+PD)? The only untested subsystems are positional and attention/structural. With 0 remaining engineering runs, this is a directive for future budget cycles.

**MEDIUM SURPRISAL (continue exploring):** positional -- 0 experiments. This is the largest gap in our coverage. Learnable RoPE frequencies (H20) was rejected only due to budget constraints, not on merit. This should be the first experiment in any future cycle.

**LOW SURPRISAL (diminishing returns):** All BLOCKED subsystems. embeddings/bottleneck -- the rate-distortion curve is mapped (192-dim too aggressive, 256-dim optimal standalone, stacking incompatible). No further embedding experiments warranted.

## Follow-Up Authorisations

**NO FOLLOW-UP: Hypothesis 22.** The result is mechanically CONFIRMED but the stacking interaction is negative. No parameter variation (e.g., different bottleneck_dim, different PD learning rate) will fix a fundamental rank incompatibility. The stacking/with-factored-embeddings subcategory is BLOCKED. Engineering run budget is exhausted (17/17).

**NO FOLLOW-UP: Hypothesis 12 (LAWA/EMA).** The failure is clean: beta=0.999 is wrong for this regime. A lower beta (0.995, 0.99) might work, but the expected gain is small given warmdown already provides implicit averaging. The training-loop subsystem remains OPEN but is LOW PRIORITY.

**NO FOLLOW-UP: Hypothesis 13 (QK-Norm Linf).** The regression (+0.007) is the worst in the programme. The attention subsystem's rigidity to norm changes is now established.

---

# Cycle 4 Follow-Up Authorisations

**NO FOLLOW-UP: Hypothesis 11 (Peri-LN).** The delta (-0.000522) is far from the threshold (-0.003), only 17% of the way there. Surprisal is low -- both agents predicted failure and got failure. No mechanistic surprise warrants a variation. The normalisation/output-norm subsystem is tested and unproductive at depth=10. No parameter tweak or implementation variation will overcome the fundamental issue that variance cascade is not a problem at 10 layers.

---

# Cycle 5 Follow-Up Authorisations

**NO FOLLOW-UP: Hypothesis 15 (Triple Stack).** The INCONCLUSIVE verdict is accepted. The key finding is not borderline performance but negative interaction: factored embeddings degrade SwiGLU+PD. No parameter tweak addresses this -- the interaction is structural. H10 (SwiGLU+PD, 0.953648) remains the best configuration.

**NO FOLLOW-UP: Hypothesis 19 (SwiGLU+Factored).** Same negative interaction as H15. Factored embeddings neutralize SwiGLU's gating benefit. No follow-up warranted.

**NO FOLLOW-UP: Hypothesis 17 (Tail-Phase EMA).** Baseline-contaminated experiment confirmed the null effect of EMA. Training-loop/weight-averaging is now BLOCKED.

**NO FOLLOW-UP: Hypothesis 16 (192-dim Factored Embeddings).** The rate-distortion knee is located between 192 and 256 dims. This is useful information but does not warrant a 224-dim follow-up -- H14 at 256-dim is already CONFIRMED and near-optimal. Further bottleneck sweeps have low expected information gain.

**FOLLOW-UP AUTHORISED: H4+H14 stacking (PD Residual + Factored Embeddings, WITHOUT SwiGLU).** Reason: HIGH SURPRISAL from the negative SwiGLU-factored interaction. The question is whether the negative interaction is SwiGLU-specific (factored embeddings degrade gated activations) or general (factored embeddings degrade all stacked configurations). Testing PD+factored without SwiGLU isolates the mechanism. If PD+factored works: the problem is specific to gated activations needing high-rank input. If PD+factored also fails: factored embeddings are fundamentally incompatible with stacking. Use ReluSquared (baseline activation) + PD derivative + factored embeddings (256-dim). Success criterion: val_bpb < 0.952 (delta < -0.003 from H4 baseline of 0.955143, or equivalently delta < -0.007 from original baseline). Write as a new sprint contract with status PROPOSED.

**STANDING AUTHORISATIONS FROM CYCLE 3 REMAIN:** The triple-stack experiment (H10 + H14) and the bottleneck_dim variation remain the highest-priority experiments. These were authorised in cycle 3 and have not yet been executed. The Researcher should prioritise these over proposing new hypotheses.

---

# Cycle 2 Evaluations

# Cycle 3 Evaluations

## Evaluation: Hypothesis 10 -- SwiGLU + PD Residual Stacking (H6+H4)

**Sprint contract thresholds:**
- val_bpb delta < -0.006: FAIL (actual: -0.005692)
- VRAM < 76 GB: PASS (actual: 72.3 GB)
- Wall-clock < 1200s: PASS (actual: 1121.5s)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

The contract specifies INCONCLUSIVE if delta is between -0.006 and -0.003. The actual delta of -0.005692 falls in this range. The stacked result is the best val_bpb in the programme (0.953648) but does not exceed the best individual result (H6: -0.004964) by the required 0.001 margin. The stacking produced a delta that is only 0.000728 better than H6 alone, not the required 0.001. Stacking is subadditive: -0.005692 vs the sum of individual deltas (-0.005 + -0.004 = -0.009). The interaction discount is 37%.

**Prediction calibration:**
- Researcher predicted delta=-0.007 at medium confidence, P(success)=0.60. Evaluator independently predicted P(success) = 0.55. Belief divergence was 0.05.
- Actual outcome: -0.005692. Both agents predicted this would clear the threshold; it did not. The Researcher overestimated the delta by 1.23x. Both agents were directionally correct (predicted improvement, got improvement) but magnitude-wrong about clearing the stacking threshold.

**Surprisal score:** medium. Both agents predicted success (P=0.55-0.60) and the result was INCONCLUSIVE -- falling just 0.000308 short of the -0.006 threshold. The direction was correct but the stacking was more subadditive than either agent expected. The 37% interaction discount is informative: SwiGLU and PD are not fully orthogonal. They likely compete for gradient signal at the residual stream interface (SwiGLU changes gradient magnitudes flowing through the MLP, which affects the deriv_lambdas learning).

**Evaluator notes:** This is the best absolute val_bpb in the programme (0.953648) despite being INCONCLUSIVE by the stacking contract. The stacking threshold was deliberately stringent (must beat best individual by 0.001). The result demonstrates that both improvements are partially compatible but significantly subadditive. The 37% discount suggests the two subsystems are not fully orthogonal -- the MLP activation function and residual stream derivative term share gradient flow. This should become the new baseline for future experiments regardless of the INCONCLUSIVE verdict, because it is the best configuration found. However, future stacking experiments should use a weaker threshold (delta < -0.003 from this new baseline) rather than requiring superadditive gains.

---

## Evaluation: Hypothesis 12 -- LAWA/EMA Weight Averaging

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: +0.003935 last-iterate; EMA val_bpb = 1.142475)
- VRAM < 76 GB: PASS (actual: 68.1 GB)
- Wall-clock < 1200s: PASS (actual: 1015.6s)
- No NaN/Inf: PASS

**Verdict: REFUTED**

Both the last-iterate (+0.003935) and EMA (delta = +0.183135) regressed from baseline. The EMA weights are catastrophically bad (1.142 val_bpb, ~19% worse than baseline). The last-iterate regression is real but smaller (+0.004). The contract stated P(regression)=0.05, claiming "EMA cannot regress" because "the training run is identical to baseline." The last-iterate regression suggests the EMA shadow parameter maintenance (lerp_ calls) introduced subtle numerical perturbation, or this is seed variance amplified by the EMA bookkeeping overhead.

**Prediction calibration:**
- Researcher predicted delta=-0.002 at low confidence, P(success)=0.30, P(regression)=0.05. Evaluator independently predicted P(success) = 0.15. Belief divergence was 0.15.
- Actual outcome: +0.003935 (last-iterate REFUTED). Directionally wrong for the Researcher (predicted improvement, got regression). The Evaluator was more sceptical but also did not predict a regression -- both agents expected EMA to be neutral-to-positive.

**Surprisal score:** medium-high. The EMA catastrophic failure (1.142 val_bpb) is the most dramatic regression in the programme. The last-iterate regression is more surprising: the training procedure was supposedly identical to baseline, yet the last-iterate weights degraded by +0.004. This suggests either (a) the EMA lerp_ calls introduced numerical side effects (unlikely but possible with bf16 precision), or (b) this is within the seed variance range and the "regression" is noise. The EMA failure itself is less surprising given the Evaluator's pre-run concern about beta=0.999 being too wide (1000-step window incorporating unconverged early weights). The mechanism is clear: at 1800 steps with 70% warmdown, EMA(0.999) averages weights from step 800 onward, which includes the entire warmdown phase -- but also the pre-warmdown phase where weights are far from converged.

**Evaluator notes:** The EMA catastrophe is explained by the beta=0.999 choice. The EMA's effective window is ~1000 steps, meaning it incorporates weights from step ~800 onward. But training only runs 1800 steps total, and the first 540 steps are warmup. The EMA thus averages weights from across the entire learning rate lifecycle, including the high-LR phase where weights are rapidly changing. This is exactly the failure mode the Evaluator predicted. The last-iterate regression (+0.004) is more puzzling. If the training dynamics are truly unchanged, this should be zero (same seed, same computation graph). Possible explanations: (1) the lerp_ calls on shadow params cause torch.compile to generate a slightly different graph; (2) bf16 accumulation of the lerp_ intermediate affects the training params through aliased memory; (3) genuine seed variance. This is worth noting but not worth investigating further. The training-loop subsystem is now tested with one REFUTED result. LAWA/EMA with beta=0.999 is definitively wrong for this regime.

---

## Evaluation: Hypothesis 13 -- QK-Norm Linf Relaxation

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: +0.007378)
- VRAM < 76 GB: PASS (actual: 67.8 GB)
- Wall-clock < 1200s: PASS (actual: 1017.4s)
- No NaN/Inf: PASS

**Verdict: REFUTED**

The +0.007378 regression is the worst single-experiment delta in the programme. The loss curve was consistently ~0.02-0.03 higher than baseline from step ~300 onward, indicating the regression is not a late-training artifact but a fundamental representation quality problem.

**Prediction calibration:**
- Researcher predicted delta=-0.001 at low confidence, P(success)=0.15, P(regression)=0.30. Evaluator independently predicted P(success) = 0.10. Belief divergence was 0.05.
- Actual outcome: +0.007378. Both agents correctly anticipated this was risky (low P(success)), and the Researcher explicitly assigned P(regression)=0.30 -- the highest regression probability in any cycle 3 contract. The Researcher was directionally correct in predicting regression risk. The magnitude (+0.007) is larger than either agent expected even in the regression scenario.

**Surprisal score:** low-medium. Both agents predicted this was unlikely to succeed (P=0.10-0.15) and the Researcher assigned P(regression)=0.30. The regression itself is not surprising. The magnitude (+0.007, worst in programme) is mildly surprising but falls within the range of "attention modifications that fail hard" established by H8 (+0.005). This is the second attention modification to produce a large regression, confirming the pattern.

**Evaluator notes:** This is the second attention subsystem failure, after H8 (+0.005). Combined pattern: both scalar temperature modification (H8) and geometric norm change (H13) produced regressions. The regression magnitude escalated: H8 was +0.005, H13 is +0.007. The Linf norm changes the constraint geometry from hypersphere (L2) to hypercube (Linf), and despite the 1/sqrt(head_dim) scaling to maintain logit range, the attention patterns degrade. The likely mechanism: L2 normalization provides a smooth, rotationally symmetric constraint surface. Linf normalization produces a non-smooth constraint surface with corners at the hypercube vertices. The gradient of the Linf norm is sparse (only the argmax dimension receives gradient), which creates training instability -- not NaN-level instability, but sufficient to degrade the quality of learned attention patterns. Additionally, the 1/sqrt(128) = 0.0884 scaling on Q creates a very compressed logit range, reducing the effective resolution of attention scores. This explains the consistent +0.02-0.03 train loss elevation from step 300 onward: the attention mechanism is fundamentally less expressive with Linf normalization.

---

## Evaluation: Hypothesis 14 -- Factored Embeddings (256-dim Bottleneck)

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.003215)
- VRAM < 76 GB: PASS (actual: 67.9 GB)
- Wall-clock < 1200s: PASS (actual: 1019.2s)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

All thresholds met. The delta of -0.003215 just clears the -0.003 threshold by 0.000215. This is a borderline confirmation -- the margin is thin but the threshold is met. The run saved 19M parameters (from ~84M to ~65M) while improving val_bpb, suggesting the embedding was over-parameterized and the bottleneck acts as regularization.

**Prediction calibration:**
- Researcher predicted delta=-0.001 at low confidence, P(success)=0.15, P(regression)=0.25. Evaluator independently predicted P(success) = 0.08. Belief divergence was 0.07.
- Actual outcome: -0.003215. Both agents significantly underestimated the improvement. The actual delta is 3.2x the Researcher's prediction and far exceeded both agents' expectations. This is a genuine surprise.

**Surprisal score:** HIGH. Both agents predicted failure (P=0.08-0.15). The Researcher assigned P(regression)=0.25, meaning regression was considered more likely than success. Yet the result is a clear CONFIRMED. This is the highest-surprisal result since H4 (PD residual scaling in cycle 1), where both agents also predicted failure and got confirmation. The mechanism -- parameter reduction as regularization at 1800 steps -- was dismissed by both agents. The Evaluator specifically wrote "the model does not benefit from having fewer parameters (this is not a regularization-limited regime at 1800 steps)" -- this was wrong.

**Evaluator notes:** This is the most informative result of cycle 3. Both agents' models of the architecture were wrong about the embedding subsystem. The 60% parameter reduction (32.2M to 13.1M in embeddings) improved val_bpb, directly contradicting the Evaluator's pre-run assessment that "the freed capacity is not reinvested anywhere" and that 1800 steps is too short for the regularization benefit. The research_log notes that the loss curve was consistently 0.02-0.04 lower than baseline from step 300 onward, indicating the benefit manifests early and persists. The wte_proj matrix (256x640) being optimized by Muon may be critical -- Muon's Newton-Schulz orthogonalization on the projection could produce a better linear map than AdamW-trained full embeddings. This opens the embeddings subsystem as a productive direction. Follow-up: test whether a different bottleneck_dim (128, 192, 384) performs better or worse. Also test stacking with H10 (SwiGLU + PD).

---

## Evaluation: Hypothesis 6 -- SwiGLU Activation Follow-Up (hidden_dim=1792)

**Sprint contract thresholds:**
- val_bpb delta < -0.003: PASS (actual: -0.004964)
- VRAM < 76 GB: PASS (actual: 67.5 GB)
- Wall-clock < 1200s: PASS (completed without timeout; timing not preserved in run.log)
- No NaN/Inf: PASS

**Verdict: CONFIRMED**

**Prediction calibration:**
- Researcher predicted delta=-0.003 at medium confidence, P(success)=0.45. Evaluator independently predicted P(success) = 0.35. Belief divergence was 0.10.
- Actual outcome: -0.004964. Exceeded both predictions. Both agents were directionally correct but underestimated magnitude. The actual delta is 1.65x the predicted delta.

**Surprisal score:** medium. Both agents predicted borderline success (0.35-0.45 P), and the result was a clear confirmation that exceeded the threshold by 65%. The outcome resolved the H1 follow-up question cleanly: width was indeed the bottleneck. Not high surprisal because both agents thought this was plausible; but the margin of success (0.005 delta vs 0.003 threshold) is larger than either expected, which is mildly surprising.

**Evaluator notes:** This confirms that SwiGLU with hidden_dim=1792 beats ReluSquared at this scale/optimizer combination. The delta (-0.005) is the strongest single-intervention result in the programme so far, surpassing H4's PD residual scaling (-0.004). The progression from H1 (hidden_dim=1664, delta=-0.0025) to H6 (hidden_dim=1792, delta=-0.005) shows a strongly superlinear response to width: a 7.7% increase in hidden dim nearly doubled the delta. This suggests H1 was operating in a regime where the width bottleneck was severe -- the gating benefit was being throttled by insufficient capacity. The ~5% parameter count increase over baseline is a minor cost. SwiGLU is now a confirmed improvement and should be included in any stacking experiments.

---

# Cycle 4 Evaluations

## Evaluation: Hypothesis 11 -- Peri-LN Output Normalization on Sublayer Outputs

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: -0.000522)
- VRAM < 76 GB: PASS (actual: 72.6 GB)
- Wall-clock < 1200s: PASS (actual: 1017.9s)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

The delta of -0.000522 falls in the INCONCLUSIVE range (between -0.003 and +0.001). The direction is correct (improvement over baseline) but the magnitude is negligible -- only 26% of the seed variance threshold. The intervention produced a real but trivially small benefit.

**Prediction calibration:**
- Researcher predicted delta=-0.002 at low confidence, P(success)=0.25. Evaluator independently predicted P(success) = 0.20. Belief divergence was 0.05.
- Actual outcome: -0.000522. Both agents were directionally correct (predicted improvement, got improvement), but the magnitude was 3.8x less than the Researcher's prediction. Neither agent predicted success, and neither got it. Both models were roughly correct.

**Surprisal score:** low. Both agents predicted this was unlikely to succeed (P=0.20-0.25) and it did not succeed. The delta is far from the threshold (-0.000522 vs -0.003). The outcome matches the pre-run expectation: at depth=10, variance accumulation is not severe enough for Peri-LN to materially help. Neither agent's model is challenged by this result.

**Evaluator notes:** The trivial delta (-0.000522) confirms that Peri-LN's benefits require deeper networks. At 10 layers, the variance cascade that Peri-LN addresses is not the binding constraint. The Friis analogy predicted disproportionate impact from normalizing early-layer outputs, but with only 10 stages in the cascade, the cumulative variance growth is modest. The 72.6 GB VRAM is notably higher than baseline (67.8 GB) despite zero new parameters -- the two extra RMSNorm calls per block increase activation memory for torch.compile's backward graph. This VRAM cost (4.8 GB above baseline, consuming half the remaining headroom) makes Peri-LN unattractive even if a marginal benefit existed.

---

## Evaluation: Hypothesis 7 -- Leaky Integral Residual Accumulator

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: -0.000701)
- VRAM < 76 GB: PASS (actual: 69.3 GB)
- Wall-clock < 1200s: PASS (completed without timeout; timing not preserved in run.log)
- No NaN/Inf: PASS

**Verdict: INCONCLUSIVE**

**Prediction calibration:**
- Researcher predicted delta=-0.002 at low confidence, P(success)=0.20. Evaluator independently predicted P(success) = 0.12. Belief divergence was 0.08.
- Actual outcome: -0.000701. Both agents were directionally correct (predicted improvement, got improvement), but the magnitude was 2.9x less than the Researcher's prediction. The Evaluator's scepticism was better calibrated.

**Surprisal score:** low. Both agents predicted this would likely fail (P=0.12-0.20), and it did fail. The delta is weak (-0.0007) and far from the threshold (-0.003). The outcome is consistent with the Evaluator's pre-run concern: retrospective accumulation of stale layer outputs is not useful at depth=10. This is in the same category as H3's near-zero result (-0.0006).

**Evaluator notes:** H7 (integral, delta=-0.0007) is nearly identical to H3 (retrospective momentum, delta=-0.0006). Both are retrospective residual modifications and both produced negligible improvements. Meanwhile H4 (derivative/anticipatory, delta=-0.004) clearly worked. The pattern is now stark: in the residual stream at depth=10, forward-looking (velocity/derivative) information is useful; backward-looking (accumulated history/momentum) information is not. The integral term's decay=0.9 means it carries substantial signal from early layers, and this signal is evidently stale noise rather than useful correction. The integ_lambdas likely stayed near zero throughout training.

---

## Evaluation: Hypothesis 8 -- Per-Head Learnable Attention Temperature

**Sprint contract thresholds:**
- val_bpb delta < -0.003: FAIL (actual: +0.004604)
- VRAM < 76 GB: PASS (actual: 66.2 GB)
- Wall-clock < 1200s: PASS (actual: 1003.2s)
- No NaN/Inf: PASS

**Verdict: REFUTED**

**Prediction calibration:**
- Researcher predicted delta=-0.002 at low confidence, P(success)=0.25. Evaluator independently predicted P(success) = 0.20. Belief divergence was 0.05.
- Actual outcome: +0.004604. Directionally wrong for both agents -- neither predicted a regression. Both expected a small improvement or no change; the actual result is a clear degradation.

**Surprisal score:** medium-high. Both agents expected directional improvement or neutrality, not a +0.005 regression. The magnitude of harm is comparable to H2's divisive normalization failure (+0.004629). A simple 50-scalar multiplicative intervention on Q producing this level of regression is unexpected. The mechanism needs explanation: the per-head temperature either (a) destabilized QK-norm's carefully balanced logit distribution, (b) created a redundant/competing degree of freedom with the softcap that confused optimisation, or (c) was implemented with a bug that placed it in the wrong optimizer group. Implementation quality should be verified.

**Evaluator notes:** This is the first test of the attention subsystem and it produced a clear regression. The +0.005 delta is alarming for such a minimal intervention (50 scalars). Possible explanations: (1) QK-norm + softcap already provides an optimal temperature regime, and adding learnable temperature scalars creates a degenerate optimisation landscape where the temperature fights the softcap; (2) with only 5 heads, the temperature scalars may have diverged during training (some going very high, some very low), creating pathological attention patterns; (3) implementation may have placed head_temps in the wrong optimizer group (Muon vs AdamW). The Evaluator's pre-run note specifically warned about Muon grouping by shape -- if head_temps (shape (5,)) was grouped with Muon, Newton-Schulz would fail or produce garbage. Must verify implementation before drawing strong conclusions about the attention subsystem.
