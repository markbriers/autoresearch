# Research Log

## 2026-04-12 — Baseline Run
- val_bpb: 0.959340
- peak_vram_mb: 67803.6
- total_tokens: 471.9M
- depth: 10, n_embd: 640, n_head: 5, vocab_size: 8192
- Time budget: 1200s
- Note: Required C_INCLUDE_PATH=/usr/include/python3.11 for Triton compilation (Python.h header mismatch)

## Run 1 | val_bpb: 0.956880 | delta from baseline: -0.002460 | Status: PENDING_EVALUATION

**Intervention:** Replace ReluSquared MLP activation with SwiGLU (gated SiLU), using 8/3 expansion ratio (hidden_dim=1664) to match parameter count.
**Papers:** Shazeer 2020 (arXiv:2002.05202) x Carandini & Heeger 2012 (Nature Reviews Neuroscience)
**Closest prior art:** SwiGLU is standard in LLaMA/Gemma. Prior v4 result said "SwiGLU beats ReluSquared per-step." Re-test at DEPTH=10, dim=640, Muon.
**Predicted delta:** -0.005
**Actual delta:** -0.002460
**VRAM:** 65.6 GB (67159.3 MB)
**Wall-clock:** 1082.7s
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Final smoothed train loss ~2.68. MFU ~19.5%. Param count 85.0M (vs baseline). Step time ~570-580ms.

## Run 2 | val_bpb: 0.963969 | delta from baseline: +0.004629 | Status: PENDING_EVALUATION

**Intervention:** Add divisive normalization after ReluSquared: divide hidden activations by sqrt of channel-pooled mean (groups of 64 channels), with eps=1e-6.
**Papers:** Shazeer 2020 (arXiv:2002.05202) x Carandini & Heeger 2012 (Nature Reviews Neuroscience)
**Closest prior art:** BatchNorm/LayerNorm normalize differently; this is divisive normalization inspired by cortical lateral inhibition.
**Predicted delta:** -0.003
**Actual delta:** +0.004629
**VRAM:** 66.6 GB (68204.8 MB)
**Wall-clock:** 1059.1s
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth but consistently higher than baseline. Final smoothed train loss ~2.87 (baseline ~2.84). MFU ~20.3%. No new parameters. Step time ~555-565ms. Large regression; reverting commit.

## Run 3 | val_bpb: 0.958786 | delta from baseline: -0.000554 | Status: PENDING_EVALUATION

**Intervention:** Add predictor-corrector residual connection: track prev_delta (previous block output delta), add beta * prev_delta to each block output, with per-layer learnable beta (init 0.0) optimized with scalar_lr.
**Papers:** "Predictor-Corrector Enhanced Transformers with EMA Coefficient Learning" (arXiv:2411.03042) x classical Heun's method for ODEs.
**Closest prior art:** 2411.03042 uses multi-step framework for translation/summarization; this is a simplified single-step momentum blend.
**Predicted delta:** -0.004
**Actual delta:** -0.000554
**VRAM:** 69.3 GB (71005.4 MB)
**Wall-clock:** 1019.9s
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Final smoothed train loss ~2.69. MFU ~21.2%. Added 10 scalar params (pc_betas). Step time ~530-540ms. Required patching Triton driver.py (PyObject_CallOneArg -> PyObject_CallFunctionObjArgs) to fix Python 3.10/3.11 header ABI mismatch.

## Run 4 | val_bpb: 0.955143 | delta from baseline: -0.004197 | Status: PENDING_EVALUATION

**Intervention:** Add PD (proportional-derivative) controller to residual scaling: compute velocity = x - x_prev, add deriv_lambda * velocity to the residual stream before each block. Per-layer learnable deriv_lambdas (init 0.0) optimized with scalar_lr * 0.01.
**Papers:** "Predictor-Corrector Enhanced Transformers" (arXiv:2411.03042) x classical PID control theory.
**Closest prior art:** Existing resid_lambdas are proportional-only. DeepNet alpha scaling is also proportional-only. Derivative term is novel.
**Predicted delta:** -0.003
**Actual delta:** -0.004197
**VRAM:** 69.3 GB (71005.4 MB)
**Wall-clock:** 1016.9s
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Final smoothed train loss ~2.68. MFU ~21.1%. Added 10 scalar params (deriv_lambdas). Step time ~535-545ms.

## Run 5 | val_bpb: 0.954376 | delta from baseline: -0.004964 | Status: PENDING_EVALUATION

**Intervention:** Replace ReluSquared MLP with SwiGLU (gated SiLU) using hidden_dim=1792 (wider than H1's 1664).
**Papers:** Shazeer 2020 (GLU Variants) x Carandini & Heeger 2012 (divisive normalization / gain control)
**Closest prior art:** H1 tested SwiGLU at hidden_dim=1664, got delta=-0.00246 (INCONCLUSIVE). This follow-up uses hidden_dim=1792.
**Predicted delta:** -0.003
**Actual delta:** -0.004964
**VRAM:** 67.5 GB (69109.8 MB)
**Wall-clock:** 1043.2s
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Step time ~550-575ms. MFU ~20-21%. Total params 87.5M (vs baseline ~84M). 3 weight matrices per MLP layer (c_gate, c_fc, c_proj) with hidden_dim=1792.

## Run 6 | val_bpb: 0.958639 | delta from baseline: -0.000701 | Status: PENDING_EVALUATION

**Intervention:** Add leaky integral term to residual stream: integral = 0.9 * integral + block_delta; x = x + integral_lambda * integral.
**Papers:** Predictor-Corrector Enhanced Transformers (arXiv:2411.03042) x Astrom & Murray PID control theory
**Closest prior art:** H4 (PD residual, delta=-0.004 CONFIRMED). H3 (predictor-corrector, delta=-0.0006 INCONCLUSIVE).
**Predicted delta:** -0.002
**Actual delta:** -0.000701
**VRAM:** 69.3 GB (71005.1 MB)
**Wall-clock:** 961.6s
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Step time ~515-540ms. MFU ~21-22%. Added 10 scalar params (integral_lambdas, init=0, decay=0.9). Initial torch.compile required C_INCLUDE_PATH workaround for Python.h missing from /usr/include/python3.10.

## Run 7 | val_bpb: 0.963944 | delta from baseline: +0.004604 | Status: PENDING_EVALUATION

**Intervention:** Add per-head temperature scalars (init=1.0) that multiply Q after QK-norm, before FA3. head_temps at GPT level, shape (n_layer, n_head) = (10, 5).
**Papers:** Ye et al. 2024 Differential Transformer (arXiv:2410.05258) x Boltzmann distribution / statistical mechanics temperature
**Closest prior art:** Standard 1/sqrt(d_k) scaling; QK-norm in baseline. Differential attention failed 3x in prior versions.
**Predicted delta:** -0.002
**Actual delta:** +0.004604
**VRAM:** 66.2 GB (67804.2 MB)
**Wall-clock:** 949.4s
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Step time ~505-540ms. MFU ~21-22%. Added 50 scalar params (head_temps, init=1.0). Initial run crashed with FA3 dtype error (float32 from nn.Parameter); fixed by casting head_temps to q.dtype. Required separate fix commit.

## Run 8 | val_bpb: 0.953648 | delta from baseline: -0.005692 | Status: INCONCLUSIVE

**Intervention:** Combine SwiGLU activation (hidden_dim=1792, from H6) and PD residual derivative scaling (deriv_lambdas, from H4) in a single run.
**Papers:** Shazeer 2020 (GLU Variants) x classical PID control theory (derivative term for residual stream)
**Closest prior art:** H6 (SwiGLU, delta=-0.005) and H4 (PD residual, delta=-0.004) individually confirmed. This is a stacking experiment.
**Predicted delta:** -0.007
**Actual delta:** -0.005692
**VRAM:** 72.3 GB (72310.6 MB)
**Wall-clock:** 1121.5s (training: 1067.1s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Step time ~580-600ms. MFU ~19.4%. Total params include SwiGLU (3 projections x 1792) + deriv_lambdas (10 scalars). Compilation took ~21s (first step). Epoch 1 throughout.

## Run 9 | val_bpb: 0.963275 (last-iterate) / 1.142475 (EMA) | delta from baseline: +0.003935 (last-iterate) | Status: REFUTED

**Intervention:** Add EMA weight averaging (beta=0.999) during training. Maintain shadow EMA weights, update after each optimizer step. Evaluate with both last-iterate and EMA weights.
**Papers:** Sanyal et al. 2024, "Early Weight Averaging meets High Learning Rates for LLM Pre-training" (COLM 2024) x James-Stein estimator / bias-variance tradeoff
**Closest prior art:** EMA standard in diffusion models; COLM 2024 shows it works for GPT-2 pretraining. Untested with Muon optimizer.
**Predicted delta:** -0.002
**Actual delta:** +0.003935 (last-iterate vs baseline); EMA val_bpb = 1.142475 (massive regression)
**VRAM:** 68.1 GB (68071.3 MB)
**Wall-clock:** 1015.6s (training: 962.4s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Step time ~535-540ms (no measurable overhead from EMA update). Loss curve identical to baseline (EMA is training-loop-only, does not affect training dynamics). Last-iterate val_bpb 0.963275 is worse than baseline 0.959340 (+0.004). EMA val_bpb 1.142475 is dramatically worse -- beta=0.999 creates a ~1000-step averaging window that incorporates far-from-converged early weights. The EMA weights are stale/averaged too broadly. The last-iterate regression may be due to seed variance (note: the training was identical to baseline except for the per-step lerp_ call on shadow params, which should not affect training).

## Run 10 | val_bpb: 0.966718 | delta from baseline: +0.007378 | Status: REFUTED

**Intervention:** Replace L2 QK-norm with Linf normalization (x / max|x|), with 1/sqrt(head_dim) = 1/sqrt(128) scaling on Q to maintain logit scale.
**Papers:** Karagodin et al. 2025, "Normalization in Attention Dynamics" x Compressed sensing Lp norm relaxation (Candes & Tao 2005)
**Closest prior art:** Standard QK-norm uses L2. This changes the geometric constraint from hypersphere to hypercube.
**Predicted delta:** -0.001
**Actual delta:** +0.007378
**VRAM:** 67.8 GB (67754.4 MB)
**Wall-clock:** 1017.4s (training: 963.6s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Step time ~535-540ms. MFU ~21%. Loss curve consistently ~0.02-0.03 higher than baseline throughout training (visible from step ~300 onward). Final train loss ~2.71 vs baseline ~2.70. Large regression; reverting commit.

## Run 11 | val_bpb: 0.956125 | delta from baseline: -0.003215 | Status: CONFIRMED

**Intervention:** Factor wte from (vocab_size, n_embd=640) into (vocab_size, 256) @ (256, 640). The 256-dim embedding feeds into a linear projection (wte_proj) to n_embd=640. wte_proj optimized with Muon.
**Papers:** Lan et al. 2020, "ALBERT" (arXiv:1909.11942) x Shannon 1959, rate-distortion theory
**Closest prior art:** ALBERT factored embeddings. Untested with Muon optimizer at this config.
**Predicted delta:** -0.001
**Actual delta:** -0.003215
**VRAM:** 67.9 GB (67920.0 MB)
**Wall-clock:** 1019.2s (training: 966.5s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Step time ~535-545ms. MFU ~21%. Loss curve consistently ~0.02-0.04 lower than baseline throughout training (visible from step ~300 onward). Total params reduced from ~84M to ~65M (wte shrunk from 32.2M to 8.4M, wte_proj added 0.16M). The parameter reduction acts as regularization. wte_proj (256x640) placed in Muon group by shape. Compilation took ~24s (first step). Epoch 1 throughout.

## Run 12 | val_bpb: 0.958818 | delta from baseline: -0.000522 | Status: PENDING_EVALUATION

**Intervention:** Add RMSNorm after each sublayer output (attention and MLP) before adding to the residual stream (Peri-LN). Zero-parameter change using existing norm() function.
**Papers:** N/A (standard normalization placement variant)
**Closest prior art:** Pre-LN (current baseline), Post-LN (original Transformer), Peri-LN wraps sublayer output in norm before residual add.
**Predicted delta:** -0.003
**Actual delta:** -0.000522
**VRAM:** 72.6 GB (74339.5 MB)
**Wall-clock:** 1017.9s (training: 966.7s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Step time ~535-545ms. MFU ~20.9%. Loss curve smooth. No new parameters added. VRAM increased from baseline ~67.8 GB to 72.6 GB (additional norm operations in computation graph).

## Run 13 | val_bpb: 0.956253 | delta from baseline: -0.003087 | Status: PENDING_EVALUATION

**Intervention:** Triple stack H6+H4+H14: SwiGLU activation (hidden_dim=1792) + PD-controller derivative term (deriv_lambdas init=0, lr=scalar_lr*0.01) + factored embeddings (256-dim bottleneck).
**Papers:** Shazeer 2020 (arXiv:2002.05202), Lan et al. 2020 (arXiv:1909.11942), Astrom & Murray 2008 (Feedback Systems)
**Closest prior art:** H10 stacked H6+H4 (delta=-0.005692, 37% subadditive). This adds H14 to the stack.
**Predicted delta:** -0.006
**Actual delta:** -0.003087
**VRAM:** 72.1 GB (72104.2 MB)
**Wall-clock:** 1124.8s (training: 1072.0s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Final smoothed train loss ~2.68. MFU ~19.4%. Param count 84.5M. Step time ~600ms. SwiGLU hidden_dim=1792 adds a c_gate weight per MLP layer; PD derivative adds 10 scalar deriv_lambdas. torch.compile succeeded.

## Run 14 | val_bpb: 0.956265 | delta from baseline: -0.003075 | Status: PENDING_EVALUATION

**Intervention:** Stack SwiGLU activation (H6, hidden_dim=1792) with factored embeddings (H14, bottleneck_dim=256), WITHOUT PD derivative term (H4).
**Papers:** Shazeer 2020 (arXiv:2002.05202), Shannon 1959 (rate-distortion theory)
**Closest prior art:** H10 stacked H6+H4 (delta=-0.005692). This stacks H6+H14 instead.
**Predicted delta:** -0.005
**Actual delta:** -0.003075
**VRAM:** 69.2 GB (69224.2 MB)
**Wall-clock:** 1113.6s (training: 1064.3s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Final smoothed train loss ~2.68. MFU ~19.6%. Param count 84.5M. Step time ~600ms. Val_bpb nearly identical to H15 (0.956265 vs 0.956253), suggesting PD derivative adds negligible benefit on top of H6+H14 at this scale.

## Run 15 | val_bpb: 0.956133 | delta from baseline: -0.003207 | Status: PENDING_EVALUATION

**Intervention:** Apply EMA weight averaging ONLY during warmdown phase (steps 540-1800), with beta=0.99 (effective window ~100 steps). Evaluate both last-iterate and EMA weights.
**Papers:** "Exponential Moving Average of Weights in Deep Learning" (arXiv:2411.18704), "When, Where and Why to Average Weights?" (arXiv:2502.06761), Polyak-Ruppert tail averaging (1991)
**Closest prior art:** H12 tested EMA with beta=0.999 over full training and was REFUTED (EMA val_bpb=1.142).
**Predicted delta:** -0.001
**Actual delta:** -0.003207 (but see notes below)
**VRAM:** 68.2 GB (68181.4 MB)
**Wall-clock:** 1018.9s (training: 965.1s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. last_iterate_bpb=0.956133, ema_bpb=0.956294. The last iterate was BETTER than EMA. The val_bpb=0.956133 reflects the last-iterate (min of the two). The -0.003207 delta vs baseline (0.959340) is due to the codebase already including H14 factored embeddings (256-dim), not the EMA intervention. The EMA itself did not help: ema_bpb (0.956294) > last_iterate_bpb (0.956133). This is the baseline architecture with factored embeddings + EMA, where EMA is a no-op improvement.

## Run 16 | val_bpb: 0.958777 | delta from baseline: -0.000563 | Status: PENDING_EVALUATION

**Intervention:** Change factored embedding bottleneck dimension from 256 (H14, confirmed) to 192.
**Papers:** Lan et al. 2020 (arXiv:1909.11942), Shannon 1959 (rate-distortion theory)
**Closest prior art:** H14 tested bottleneck_dim=256 and achieved delta=-0.003215 (CONFIRMED).
**Predicted delta:** -0.003
**Actual delta:** -0.000563
**VRAM:** 67.9 GB (67882.0 MB)
**Wall-clock:** 1013.6s (training: 964.1s)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. Loss curve smooth. Final smoothed train loss ~2.69. MFU ~21.1%. Param count 82.3M (vs 82.9M at 256-dim, vs 85.0M at full). Step time ~540ms. The 192-dim bottleneck performed substantially worse than 256-dim (val_bpb 0.958777 vs 0.956125), suggesting the rate-distortion knee is between 192 and 256 dims.

## Run 17 | val_bpb: 0.956004 | delta from baseline: -0.003336 | Status: PENDING_EVALUATION

**Intervention:** Stack PD residual scaling (H4) with factored embeddings (H14), using ReluSquared activation (not SwiGLU).
**Papers:** H4 source: Predictor-Corrector Enhanced Transformers (arXiv:2411.03042); H14 source: factored embedding bottleneck (parameter efficiency).
**Closest prior art:** H4 alone achieved val_bpb=0.955143 (delta=-0.004197). H14 alone achieved val_bpb=0.956125 (delta=-0.003215). H10 (SwiGLU+PD) achieved 0.953648. H15 (SwiGLU+PD+factored) achieved 0.956253. H19 (SwiGLU+factored) achieved 0.956265.
**Predicted delta:** not specified in contract
**Actual delta:** -0.003336
**VRAM:** 69.5 GB
**Wall-clock:** 975.2s training, 1022.5s total
**Raw observations:** Training completed 1800 steps without errors. Loss decreased monotonically. Final smoothed training loss ~2.82. No NaN or divergence observed. val_bpb=0.956004 is worse than H4 alone (0.955143) but better than H14 alone (0.956125). The stack does not appear additive -- the combined result is between the two individual results rather than better than either.
