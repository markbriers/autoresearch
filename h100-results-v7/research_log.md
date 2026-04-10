# Research Log

## 2026-04-09: Baseline Run
- val_bpb: 0.959552
- peak_vram_mb: 67803.6
- Steps: 1800, Time budget: 1200s
- Hardware: H100 NVL (95.8 GB)
- Note: Required C_INCLUDE_PATH=/usr/include/python3.11 for Triton compilation (Python.h missing for 3.10)

## Run 1 | val_bpb: 0.955919 | delta from baseline: -0.003633 | Status: PENDING_EVALUATION

**Intervention:** Replace ReluSquared with xIELU activation (learnable per-layer alpha_p, alpha_n via inverse softplus of 0.8)
**Papers:** Huang et al., 2024, "Deriving Activation Functions via Integration"
**Closest prior art:** autoresearch v6 confirmed xIELU outperforms ReluSquared
**Predicted delta:** < -0.003
**Actual delta:** -0.003633
**VRAM:** 67.8 GB (67805.5 MB)
**Wall-clock:** 1008.3s training, 1059.8s total
**Raw observations:** Training completed all 1800 steps without issues. Loss curve appeared normal with smooth convergence. No NaN or instability detected. 20 xIELU scalar params added to scalar_lr AdamW group. VRAM essentially unchanged from baseline.

## Run 2 | val_bpb: 0.958368 | delta from baseline: -0.001184 | Status: PENDING_EVALUATION

**Intervention:** Add RMS normalization after each sublayer output before residual addition (Peri-LN)
**Papers:** Kim et al., 2025, ICML "Peri-LN: Revisiting Normalization Layer in the Transformer Architecture"
**Closest prior art:** Used in Gemma-3; no prior autoresearch test
**Predicted delta:** < -0.003
**Actual delta:** -0.001184
**VRAM:** 74.2 GB (74224.2 MB)
**Wall-clock:** 945.0s training, 993.8s total
**Raw observations:** Training completed all 1800 steps without issues. No new parameters added. VRAM increased by ~6.4 GB from baseline (67.8 to 74.2 GB), likely due to additional normalization activations stored for backward pass. No NaN or instability detected.

## Run 3 | val_bpb: 0.956189 | delta from baseline: -0.003363 | Status: PENDING_EVALUATION

**Intervention:** Replace ReluSquared MLP with SwiGLU gated MLP (gate/up: 640->1664, down: 1664->640)
**Papers:** Shazeer 2020, "GLU Variants Improve Transformer"
**Closest prior art:** autoresearch v4 confirmed SwiGLU beats ReluSquared per-step
**Predicted delta:** < -0.003
**Actual delta:** -0.003363
**VRAM:** 67.2 GB (67159.3 MB)
**Wall-clock:** 1006.2s training, 1056.0s total
**Raw observations:** Training completed all 1800 steps without issues. hidden_dim=1664 (parameter-matched). VRAM slightly decreased from baseline (67.8 to 67.2 GB). No NaN or instability detected. Three Muon groups for new matrix shapes (640x1664, 1664x640) created automatically.

## Run 4 | val_bpb: 0.954713 | delta from baseline: -0.004839 | Status: PENDING_EVALUATION

**Intervention:** Add learnable per-head attention temperature scalar (tau=1.0 init) scaling q after QK-norm
**Papers:** Zhang et al., NeurIPS 2024, "Selective Attention: Enhancing Transformer through Principled Context Control"
**Closest prior art:** No prior autoresearch test
**Predicted delta:** < -0.003
**Actual delta:** -0.004839
**VRAM:** 67.8 GB (67804.2 MB)
**Wall-clock:** 930.1s training, 978.0s total
**Raw observations:** First attempt crashed with RuntimeError (FlashAttention dtype mismatch) because float32 temperature parameter upcast q from bf16 to float32. Fixed by adding .to(q.dtype) cast on temperature. Second attempt completed all 1800 steps. 60 temperature scalar params (6 heads x 10 layers) added to scalar_lr AdamW group. VRAM unchanged from baseline.

## Run 5 | val_bpb: 0.955944 | delta from baseline: -0.003608 | Status: PENDING_EVALUATION

**Intervention:** Combine xIELU activation (replacing ReluSquared in MLP) with learnable per-head attention temperature in a single model
**Papers:** Huang et al. 2024 ("Deriving Activation Functions via Integration") x Zhang et al. NeurIPS 2024 ("Selective Attention")
**Closest prior art:** F1 (xIELU alone, delta=-0.003633) and F3 (attn_temp alone, delta=-0.004839) confirmed independently
**Predicted delta:** -0.007 +/- 0.002
**Actual delta:** -0.003608
**VRAM:** 67.8 GB (67806.1 MB)
**Wall-clock:** 1014.0s training, 1065.3s total
**Raw observations:** Training completed all 1800 steps without issues. No NaN or instability. Model has n_head=5 (not 6 as contract stated). 20 xIELU scalar params + 50 attn_temperature params = 70 new scalars total. VRAM unchanged from baseline. The combined delta (-0.003608) is LESS than attn_temperature alone (-0.004839) and roughly equal to xIELU alone (-0.003633), suggesting the two interventions are not additive and may interfere.
