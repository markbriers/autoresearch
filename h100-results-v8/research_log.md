# Research Log -- v8 (Cross-Domain Enforcement)

## Run 1 | val_bpb: 0.945117 | delta from baseline: -0.014435 | Status: PENDING_EVALUATION

**Intervention:** Replace ReluSquared MLP with SwiGLU (gate = Silu(Wx) * Vx), keeping hidden dim at 4*n_embd for both projections.
**Papers:** Shazeer 2020 "GLU Variants Improve Transformer" x Donoho & Johnstone 1994 "Ideal Spatial Adaptation by Wavelet Shrinkage"
**Closest prior art:** SwiGLU tested in combination with attn stacking (H6, cycle 2) which showed destructive interference. Never tested in isolation.
**Predicted delta:** -0.006
**Actual delta:** -0.014435
**VRAM:** 79.1 GB (peak_vram_mb: 80762.8)
**Wall-clock:** 1261.2s (training_seconds: 1208.1)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. VRAM exceeded 76 GB threshold (79.1 GB vs 76 GB limit). Wall-clock exceeded 1200s budget (1261.2s total, 1208.1s training). Per-step time ~650ms vs baseline ~470ms (~38% slower due to extra gate projection). Loss curve appeared smooth with no anomalies.

## Run 2 | val_bpb: 0.953754 | delta from baseline: -0.005798 | Status: PENDING_EVALUATION

**Intervention:** Replace ReluSquared with ShrinkReLU: tau = softplus(raw_tau); x = relu(x - tau).square(). Learned per-layer threshold initialized at raw_tau=-2.0 (tau~0.13).
**Papers:** Zhao et al. 2019 "Deep Residual Shrinkage Networks" x Donoho & Johnstone 1994 "Ideal Spatial Adaptation by Wavelet Shrinkage"
**Closest prior art:** xIELU (v6) modifies activation with learned params but different mechanism. ShrinkReLU preserves ReluSquared structure with adaptive threshold shift.
**Predicted delta:** -0.004
**Actual delta:** -0.005798
**VRAM:** 66.2 GB (peak_vram_mb: 67806.1)
**Wall-clock:** 988.3s (training_seconds: 939.8)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. VRAM well within budget (66.2 GB < 76 GB). Wall-clock well within budget (988.3s < 1200s). Per-step time comparable to baseline. No anomalies observed.

## Run 3 | val_bpb: 0.956652 | delta from baseline: -0.002900 | Status: PENDING_EVALUATION

**Intervention:** Apply divisive normalization to attention output after FA3: y = y / (1 + softplus(sigma) * ||y||_dim), with per-head learnable sigma initialized to zero (identity passthrough).
**Papers:** Gemma-2 softcapping (Team et al. 2024) x Carandini & Heeger 2012 "Normalization as a canonical neural computation"
**Closest prior art:** Current code uses tanh softcapping at 15 for output logits. Divisive normalization is context-dependent unlike fixed softcap. Different from QK-norm and attention temperature.
**Predicted delta:** -0.003
**Actual delta:** -0.002900
**VRAM:** 69.4 GB (peak_vram_mb: 71055.3)
**Wall-clock:** 1003.0s (training_seconds: 953.6)
**Raw observations:** Training completed all 1800 steps. No NaN/Inf. VRAM within budget (69.4 GB < 76 GB). Wall-clock within budget (1003.0s < 1200s). Per-step time slightly higher than baseline due to norm computation in attention. No anomalies observed.

## Run 4 | val_bpb: 1.095295 | delta from baseline: +0.135743 | Status: PENDING_EVALUATION

**Intervention:** Add learnable per-dimension gain vector (init ones) to each RMSNorm, transforming parameter-free rms_norm(x) into gamma * rms_norm(x). Stacked with ShrinkReLU (H12 confirmed).
**Papers:** Zhang & Sennrich 2019 "Root Mean Square Layer Normalization" (NeurIPS 2019) x Perez 1984 "Automatic Gain Control" (telecommunications)
**Closest prior art:** Current codebase uses parameter-free RMSNorm. LLaMA/Gemma-2 use RMSNorm WITH learnable gain. Not previously tested here.
**Predicted delta:** -0.005
**Actual delta:** +0.135743
**VRAM:** 67.9 GB
**Wall-clock:** 733.2s training, 777.9s total
**Raw observations:** Training loss converged to ~3.07 (vs expected ~2.7-2.8 for baseline). val_bpb 1.095295 is a large regression from baseline 0.959552. No NaN/Inf observed. Loss curve appeared normal in shape but converged to a much higher value. The norm gain parameters (21 x 640 = 13,440 params) were added to an Adam group with lr=scalar_lr (0.5) and betas=(0.8, 0.95). VRAM was 67.9 GB (essentially unchanged). Wall-clock well within budget.

## Run 5 | val_bpb: 0.953771 | delta from baseline: -0.005781 | Status: PENDING_EVALUATION

**Intervention:** Replace ReluSquared+ShrinkReLU MLP with SwiGLU using hidden_dim=1728 (2.7x n_embd), matching current parameter budget (3 matrices at 1728 vs 2 matrices at 2560).
**Papers:** Shazeer 2020 "GLU Variants Improve Transformer" (arXiv:2002.05202) x Candes & Tao 2006 "Near-Optimal Signal Recovery from Random Projections" (IEEE TIT)
**Closest prior art:** H10 tested SwiGLU at full 4x hidden_dim: -0.014435 val_bpb but 79.1 GB VRAM and 1261s wall-clock (both over budget). This is a dimension-reduced variant at 1728 hidden dim.
**Predicted delta:** -0.008
**Actual delta:** -0.005781
**VRAM:** 68.1 GB
**Wall-clock:** 1017.3s training, 1068.2s total
**Raw observations:** Training loss converged normally. Per-step time ~550ms vs ~400ms for baseline (37% slower per step due to 3 matmuls vs 2). Final training loss in the ~2.6 range. val_bpb 0.953771 vs baseline 0.959552. VRAM 68.1 GB (essentially unchanged from baseline 67.8 GB). Wall-clock 1017.3s is within the 1200s budget but notably slower than baseline (~730s). No NaN/Inf observed. This replaces confirmed ShrinkReLU (val_bpb 0.953754, delta -0.005798), so the net improvement over ShrinkReLU is approximately 0.953754 - 0.953771 = -0.000017 (essentially identical).

## Run 6 | val_bpb: 1.830049 | delta from baseline: +0.870497 | Status: PENDING_EVALUATION

**Intervention:** Per-head attention output scaling (HeadScale) -- learnable scalar per attention head scaling the head's contribution after FlashAttention3 output, before c_proj.
**Papers:** Bhojanapalli et al. 2021 (NeurIPS workshop) x Turrigiano 2008 (synaptic scaling)
**Closest prior art:** attn_temperature (scales Q before dot product); HeadScale scales OUTPUT after attention
**Predicted delta:** -0.004
**Actual delta:** +0.870497
**VRAM:** 69.3 GB (peak_vram_mb=71004.2)
**Wall-clock:** 775.0s training, 821.4s total
**Raw observations:** Massive regression. Train loss at step 100 was 5.306 (baseline ShrinkReLU shows ~3.95 at step 100). Final train loss 5.13 at step 1800. Model converged but to a much worse solution. 1800 steps completed, no NaN/Inf. VRAM within budget. The head_scale parameters at scalar_lr=0.5 may have interfered with the already-present attn_temperature parameters at the same LR, creating redundant/conflicting scaling degrees of freedom.

## Run 7 | val_bpb: 1.097629 | delta from baseline: +0.138077 | Status: PENDING_EVALUATION

**Intervention:** Learned per-layer norm scaling (NormScale) -- single learnable scalar per RMSNorm site (21 sites: 2 per block + 1 final) scaling norm output. Init to 1.0.
**Papers:** Zhang & Sennrich 2019 (RMSNorm) x Boltzmann 1877 / Gibbs 1902 (partition function temperature)
**Closest prior art:** H16 (per-dimension gain) was catastrophic (+0.136); NormScale uses scalar-per-site instead
**Predicted delta:** -0.003
**Actual delta:** +0.138077
**VRAM:** 66.3 GB (peak_vram_mb=67891.0)
**Wall-clock:** 732.2s training, 776.9s total
**Raw observations:** Large regression similar in magnitude to H16 (+0.136). Train loss at step 100 was 4.630 (baseline ~3.95). Final train loss remained elevated. 1800 steps completed, no NaN/Inf. VRAM within budget. Despite using only 21 scalar params (vs H16's 13,440), the norm gain at scalar_lr=0.5 still caused significant regression. Both H16 and H20 show similar ~+0.14 delta despite very different param counts.

## Run 8 | val_bpb: 1.830455 | delta from baseline: +0.870903 | Status: PENDING_EVALUATION

**Intervention:** Learned sublayer output scaling (SubScale) -- two learnable scalars per layer (attn_scale, mlp_scale) scaling sublayer output before residual addition. Init to 1.0. lr=scalar_lr*0.01.
**Papers:** Wang et al. 2022 (DeepNet) x Turrigiano & Nelson 2004 (homeostatic plasticity)
**Closest prior art:** H9 (sublayer scaling, REFUTED in cycle 2)
**Predicted delta:** -0.004
**Actual delta:** +0.870903
**VRAM:** 78.8 GB (peak_vram_mb=80689.0)
**Wall-clock:** 737.4s training, 786.0s total
**Raw observations:** Massive regression identical in magnitude to H17 (+0.870). Train loss at step 100 was 5.310. Final val_bpb=1.830455 nearly identical to H17's 1.830049. 1800 steps completed, no NaN/Inf. VRAM near limit at 78.8 GB. This pattern of val_bpb ~1.83 for both H17 and H18 is suspicious -- may indicate a systematic issue with the ShrinkReLU baseline setup rather than experiment-specific problems.

## IMPORTANT NOTE: Runs 6-8 (H17, H20, H18) are INVALID

The torch.compile inductor cache from the prior SwiGLU (H15) experiment was stale. When train.py reverted to ShrinkReLU, the cached compiled kernels were incompatible with the new model architecture. Additionally, clearing the cache exposed a missing Python.h issue (python3.10-dev not installed), requiring a patch to triton's build.py to use extracted headers from libpython3.10-dev.

All three experiments (H17 val_bpb=1.830, H20 val_bpb=1.098, H18 val_bpb=1.830) ran with corrupted compiled code and their results are meaningless. These must be re-run after fixing the environment.

## Run 9 | val_bpb: 0.953048 | delta from baseline: -0.006504 | Status: PENDING_EVALUATION

**Intervention:** Learned output softcap temperature -- replace fixed softcap=15 with learnable `softcap = 15 * softplus(raw_softcap) / ln(2)`, where raw_softcap=0 gives softcap=15 (identity init). Optimized with lr=scalar_lr, betas=(0.96, 0.95).
**Papers:** Team Gemma 2024 (softcapping) x Jaynes 1957 (maximum entropy principle)
**Closest prior art:** Fixed softcap=15 in current codebase; no prior art making softcap learnable
**Predicted delta:** -0.003
**Actual delta:** -0.006504
**VRAM:** 70.2 GB (peak_vram_mb=71894.2)
**Wall-clock:** 928.1s training, 985.6s total
**Raw observations:** Healthy training curve. Step 100 loss=3.966 (baseline ~3.949). val_bpb=0.953048, delta=-0.006504. 1800 steps completed, no NaN/Inf. VRAM within budget. Wall-clock within budget. This is the first run after fixing the triton compilation environment (Python.h headers). Clean cache, valid compilation.
