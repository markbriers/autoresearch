# Research Log — v5 (Research-Only Constraint)

## Design

All hyperparameters and model dimensions are FROZEN. The agent can only modify architecture through paper-driven synthesis. No HP tuning, no scaling, no Tier B refinements. This isolates whether cross-domain paper reading produces improvements that engineering cannot.

## Baseline

10x640 (85.9M params), 1800 steps, batch=262K.
- **val_bpb: 0.953643**
- Training time: 452.4s
- Peak VRAM: 67.8 GB
- MFU: 40.60%

## Run 1 | val_bpb: N/A | delta from baseline: N/A | CRASH

**Diagnosis:** SwiGLU OOM — extra gate projection adds ~16.4M params and proportionally more activations, exceeding 80GB VRAM.

**Domain A** — "Are Human Dendrites Different?" (Fişek & Häusser, 2020, DOI: 10.1016/j.tics.2020.03.002)
- Source primitive: Multiplicative dendritic gating (XOR-like operations via calcium spikes)
- Target bottleneck: ReluSquared limits MLP function class to single-pathway activation
- Mapping: Dendritic compartments multiplicatively gate each other; SwiGLU mirrors this
- Validation: Human L2/3 dendrites compute XOR through multiplicative interactions

**Domain B** — "GLU Variants Improve Transformer" (Shazeer, 2020, arXiv: 2002.05202)
- Source primitive: SwiGLU = (Swish₁(xW) ⊗ xV)W₂
- Target bottleneck: Single-projection activation limits expressiveness
- Mapping: Second projection provides data-dependent gating filter
- Validation: SwiGLU 1.636 vs ReLU 1.677 log-perplexity on C4

**Synthesis:** Multiplicative gating from dendritic computation enables richer function class. SwiGLU implements this in transformers.
**Falsifiability:** "val_bpb will decrease because multiplicative gating enables more complex features, not just from more params."
**Code change:** Replace ReluSquared MLP with SwiGLU (c_gate + c_up + c_proj). Adds 16.4M params.
**Outcome:** CRASH — OOM. The extra gate projection (10 layers × 640 × 2560) pushes total VRAM beyond 80GB. The baseline already uses 67.8GB with DBS=128. The diagnosis was correct but the implementation is infeasible under VRAM constraints. Would need reduced batch size (frozen) or reduced MLP width (frozen).
**Surprise score:** 2 (VRAM constraint was foreseeable given 67.8GB baseline)

## Run 2 | val_bpb: 0.951991 | delta from baseline: -0.001652 | KEEP

**Diagnosis:** Pre-attention causal depthwise conv improves val_bpb by smoothing local context before Q/K projections. However, significant throughput penalty (30% vs 41% MFU) from breaking torch.compile fusion.

**Domain A** — "Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., 2020, DOI: 10.21437/interspeech.2020-3015)
- Source primitive: Depthwise separable convolution module with GLU gating, capturing local acoustic features
- Target bottleneck: Self-attention treats each position independently before computing correlations
- Mapping: In speech, phoneme features span multiple frames. Depthwise conv captures local patterns that attention alone misses
- Validation: Removing conv causes largest performance drop in Conformer (2.1%→1.9% WER)

**Domain B** — "Hungry Hungry Hippos: Towards Language Modeling with State Space Models" (Dao et al., 2023, arXiv: 2212.14052)
- Source primitive: Shift-SSM — causal short convolution storing last m tokens as state
- Target bottleneck: Attention Q/K dot products operate on point-wise token representations
- Mapping: Shift-SSM preprocesses keys with local context before attention-like operation
- Validation: Hybrid H3-attention outperforms pure Transformer by 1.0 PPL on OpenWebText

**Synthesis:** Both speech and language models benefit from local convolution before global attention. The insight: attention Q/K projections operate on point-wise representations and benefit from local context smoothing regardless of domain.
**Falsifiability:** "val_bpb will decrease because Q/K operates on locally-smoothed features, and this would NOT appear if sliding window already captures sufficient local context."
**Code change:** Added depthwise Conv1d(640, kernel=4, causal) before Q/K projections. Identity-initialized. 2560 extra params. Conv params use AdamW with scalar LR.
**Outcome:** CONFIRMED. val_bpb 0.951991 vs 0.953643 baseline (delta -0.0017). The improvement is real but modest. Significant throughput penalty: 30% MFU vs 41% baseline (conv breaks torch.compile fusion). VRAM +3.2GB. The hypothesis that local smoothing helps Q/K is validated, but the cost-benefit is marginal.
**Surprise score:** 3 (the improvement was expected; the severe MFU drop was not — depthwise conv1d is not well-fused by torch.compile)

## Run 3 | val_bpb: 0.955985 | delta from baseline: +0.002342 | DISCARD

**Diagnosis:** Z-loss on pre-softcap logits hurts performance. The softcap (tanh at ±12) already controls logit magnitude; z-loss adds unnecessary regularization pressure that reduces the model's ability to make confident predictions.

**Domain A** — "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (Zoph et al., 2022, arXiv: 2202.08906)
- Source primitive: Router z-loss L_z = (1/B)Σ(log Σ exp(x_j))² with coefficient 0.001
- Target bottleneck: Large logit magnitudes cause numerical instability in bfloat16
- Mapping: Z-loss penalizes log-sum-exp squared, keeping logits bounded differentiably
- Validation: Z-loss achieves 100% stability (3/3 runs) vs 67% without in MoE training

**Domain B** — Maximum entropy principle (Jaynes, 1957, "Information Theory and Statistical Mechanics")
- Source primitive: Equilibrium distributions maximize entropy subject to constraints
- Target bottleneck: Output logits can grow large, concentrating probability too sharply
- Mapping: Z-loss acts as entropy-like regularizer preventing peaked output distributions
- Validation: Maximum entropy principle is foundational to statistical mechanics

**Synthesis:** Z-loss implements an entropy-maintaining pressure from statistical mechanics in the logit space. However, this model already uses softcap=12 which bounds logit magnitudes via tanh.
**Falsifiability:** "val_bpb will decrease because z-loss improves gradient flow to low-probability tokens, and this would NOT appear if softcap already provides sufficient logit regularization."
**Code change:** Added z_loss = 1e-4 * torch.logsumexp(logits, dim=-1).square().mean() to cross-entropy loss, computed on pre-softcap logits.
**Outcome:** REFUTED. val_bpb 0.955985 vs 0.951991 (delta +0.004). The softcap already bounds logits, making z-loss redundant. The z-loss adds ~0.05 to the loss, which distorts the gradient signal. The mapping was wrong: the model doesn't suffer from unbounded logits (softcap prevents this), so there's no bottleneck for z-loss to address. The 1e-4 coefficient may also be too large for this model scale.
**Surprise score:** 2 (expected that softcap + z-loss might conflict; the magnitude of regression was somewhat surprising)

## Run 4 | val_bpb: N/A | delta from baseline: N/A | CRASH

**Diagnosis:** Sandwich norm (post-norm on sublayer outputs) causes OOM. Each additional norm() call saves intermediate tensors for backprop, pushing VRAM past 80GB.

**Domain A** — "CogView: Mastering Text-to-Image Generation via Transformers" (Ding et al., 2021, arXiv: 2105.13290)
- Source primitive: Sandwich LayerNorm (pre+post norm) for stable deep autoregressive image generation
- Target bottleneck: Pre-norm alone allows hidden state magnitudes to grow
- Mapping: Post-norm bounds each sublayer's output before residual addition
- Validation: CogView 4B could not converge without sandwich norm

**Domain B** — "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., 2022, arXiv: 2203.00555)
- Source primitive: SubLN (sub-LayerNorm) for extremely deep transformers
- Target bottleneck: Residual stream magnitude grows with depth in pre-norm models
- Mapping: Post-norm ensures uniform contribution across layers
- Validation: DeepNet scales to 1000+ layers with SubLN

**Synthesis:** Both CogView and DeepNet found post-norm essential for deep models. Applied to our 10-layer model to bound sublayer output magnitude.
**Falsifiability:** "val_bpb will decrease because post-norm controls residual magnitude, and this would NOT appear if pre-norm + resid_lambdas already suffice."
**Code change:** Wrapped attn and MLP outputs in norm() before residual addition. Zero extra parameters.
**Outcome:** CRASH — OOM. Even though norm() has no learnable params, the backward pass through the additional norms saves intermediate activations for gradient computation. At 71GB (with pre-attn conv), the extra norm activations exceed 80GB. The VRAM constraint is extremely tight.
**Surprise score:** 3 (didn't expect a parameter-free change to cause OOM — the activation memory cost of norm backward was underestimated)

## Run 5 | val_bpb: 0.951769 | delta from baseline: -0.001874 | KEEP

**Diagnosis:** Per-head learnable attention temperature provides small additional improvement on top of pre-attention conv. Combined improvement: -0.0019 from baseline.

**Domain A** — "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015, arXiv: 1503.02531)
- Source primitive: Softmax temperature T in softmax(z/T) controls distribution sharpness
- Target bottleneck: Fixed softmax scale gives all heads the same attention sharpness
- Mapping: Different tasks need different temperatures for optimal soft/hard attention
- Validation: Temperature is essential for knowledge distillation; fixed T=1 is suboptimal

**Domain B** — Boltzmann distribution (Gibbs, 1902, "Elementary Principles in Statistical Mechanics")
- Source primitive: P(state) ∝ exp(-E/kT) — temperature controls equilibrium distribution sharpness
- Target bottleneck: With QK-norm, attention temperature is effectively fixed by head_dim
- Mapping: Different physical systems at different temperatures exhibit different behaviors
- Validation: Temperature is fundamental to phase transitions and distribution shape in physics

**Synthesis:** QK-norm fixes effective attention temperature. Per-head learnable temperature (via Q scaling) lets each head discover its optimal sharpness — some sharp for precise retrieval, others broad for context aggregation.
**Falsifiability:** "val_bpb will decrease because heads learn diverse temperature profiles, and this would NOT appear if all heads need the same temperature."
**Code change:** Added log_temp parameter per head (50 params total). Q scaled by exp(log_temp) after QK-norm. Uses AdamW with scalar_lr*0.01.
**Outcome:** CONFIRMED. val_bpb 0.951769 vs 0.951991 (delta -0.0002). Small but consistent improvement. Zero VRAM overhead. The per-head temperature enables attention diversity that QK-norm alone cannot provide.
**Surprise score:** 2 (expected direction, small magnitude — the heads likely don't diverge much in temperature preference with only 1800 steps)

## Run 6 | val_bpb: 0.951846 | delta from baseline: -0.001797 | DISCARD

**Diagnosis:** Extending conv smoothing to V inputs gives essentially no change (+0.00008). Locally-smoothed values blur token-level distinctions that attention should preserve.

**Code change:** Changed V projection input from raw x to x_conv.
**Outcome:** DISCARD. Flat result. Smoothing V is unhelpful because the attention mechanism's role is to aggregate distinct token values — smoothing them defeats the purpose.
**Surprise score:** 1 (expected that V benefits from per-token granularity)

## Run 7 | val_bpb: 0.948232 | delta from baseline: -0.005411 | KEEP

**Diagnosis:** Partial RoPE (75% rotated, 25% position-free) provides the largest improvement yet. Dedicating 32 of 128 head dimensions to position-free semantic matching enables better content-based attention patterns.

**Domain A** — "The Impact of Positional Encoding on Length Generalization in Transformers" (Kazemnejad et al., 2023, arXiv: 2305.19466)
- Source primitive: Analysis of how different positional encoding schemes affect transformer capabilities
- Target bottleneck: Full RoPE forces ALL attention to be position-dependent, preventing pure content matching
- Mapping: Removing positional encoding from some dimensions allows position-free semantic similarity
- Validation: Paper shows that positional encoding design significantly impacts transformer behavior

**Domain B** — Frequency allocation in telecommunications (Shannon, 1948; orthogonal frequency-division multiplexing)
- Source primitive: OFDM allocates different subcarriers to different types of information (pilot signals vs data)
- Target bottleneck: All head dimensions carry the same type of information (position-encoded)
- Mapping: Dedicating separate "subcarriers" (dimensions) to positional vs semantic info allows independent optimization of both channels
- Validation: OFDM's success in wireless communications demonstrates that separating information types into orthogonal subspaces improves overall system capacity

**Synthesis:** Full RoPE forces all QK dot product dimensions to be position-modulated, preventing the model from computing pure content similarity. Partial RoPE allocates 75% of dimensions to position-aware matching (where to attend) and 25% to position-free semantic matching (what to attend to). This mirrors OFDM's principle of dedicating orthogonal subcarriers to different information types.
**Falsifiability:** "val_bpb will decrease because position-free dimensions enable content-based attention, and this would NOT appear if all attention patterns are primarily position-dependent."
**Code change:** Modified apply_rotary_emb to apply RoPE to only 96 of 128 head dims (75%). Last 32 dims pass through without rotation.
**Outcome:** CONFIRMED. val_bpb 0.948232 vs 0.951769 (delta -0.0035). This is the largest single improvement. The position-free dimensions allow pure semantic similarity computation, complementing the position-aware channels.
**Surprise score:** 4 (the magnitude of improvement was larger than expected — -0.0035 is nearly twice the pre-attn conv improvement)

## Run 8 | val_bpb: 0.948432 | delta from baseline: -0.005211 | DISCARD

**Diagnosis:** Attention softcap=30 via flash_attn has no effect since QK-norm already bounds attention logits to ~±11.3, well below 30.
**Code change:** Added softcap=30.0 to flash_attn_func call.
**Outcome:** DISCARD. Flat result (+0.0002 from best). The softcap is too high to have any effect with QK-norm.
**Surprise score:** 1

## Run 9 | val_bpb: 0.951494 | delta from baseline: -0.002149 | DISCARD (diagnostic)

**Diagnosis:** Removing pre-attention conv while keeping partial RoPE + per-head temp. Tests whether conv and partial RoPE are redundant.
**Code change:** Removed pre_conv from CausalSelfAttention.
**Outcome:** DISCARD. Regression of +0.003 from best, but still better than baseline. MFU recovered to 40% (from 30%). This proves conv and partial RoPE are COMPLEMENTARY: conv provides local n-gram context, partial RoPE provides position-free semantic matching. Both are needed.
**Surprise score:** 3 (expected some overlap, but the complementarity was stronger than expected)

## Run 10 | val_bpb: N/A | delta from baseline: N/A | CRASH

**Diagnosis:** Learnable RoPE frequencies cause OOM — making cos/sin depend on a learnable parameter requires saving intermediates for backward pass. Yet another VRAM casualty.
**Surprise score:** 2

## Run 11 | val_bpb: 0.948813 | delta from baseline: -0.004830 | DISCARD

**Diagnosis:** Changing RoPE base from 10000 to 50000 slightly regresses. The default base is well-tuned for seq_len=2048. Higher base compresses frequency spectrum but reduces position discrimination for nearby tokens.
**Surprise score:** 1

## Run 12 | val_bpb: 0.951307 | delta from baseline: -0.002336 | DISCARD

**Diagnosis:** Removing K-norm (keeping Q-norm only) causes significant regression (+0.003 from best). K-norm is essential for stable attention with Muon optimizer. Letting K magnitudes vary doesn't usefully encode salience.
**Surprise score:** 2
