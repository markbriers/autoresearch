# Hypotheses

## Engineering Run Counter: 5/5 | Phase: RESEARCH (return to Phase 1, Round 3)

### H5: Per-Head Learnable Attention Temperature (PRIORITY 5) | Status: CONFIRMED (val_bpb 0.951769, delta -0.0002 from H2)

**Domain Pairing: Knowledge Distillation × Statistical Mechanics**

**Domain A** — "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015, arXiv: 1503.02531)
- Source primitive: Softmax temperature T in softmax(z/T) controls distribution sharpness
- Target bottleneck: Fixed softmax scale in attention gives all heads the same sharpness
- Mapping: Different tasks (broad context vs precise retrieval) need different temperatures
- Validation: Temperature is essential for effective knowledge distillation; T=1 (no scaling) is suboptimal

**Domain B** — Boltzmann distribution in statistical mechanics (Gibbs, 1902)
- Source primitive: P(state) ∝ exp(-E/kT) — temperature controls equilibrium distribution sharpness
- Target bottleneck: With QK-norm, attention temperature is effectively fixed by head_dim
- Mapping: Different physical systems require different temperatures for optimal behavior
- Validation: Temperature is a fundamental parameter governing phase transitions and distribution shape

**Synthesis:** With QK-norm (RMSNorm on Q, K), the effective attention temperature is fixed. Different heads may need different sharpness — some should be sharp (precise token matching) while others should be broad (context aggregation). A learnable per-head temperature allows this diversity. This is analogous to how physical systems at different temperatures exhibit different behaviors (sharp crystalline order vs fluid mixing). Only 50 extra scalar parameters.

**Falsifiability:** "val_bpb will decrease because heads learn diverse temperature profiles, and this would NOT appear if all heads need the same temperature (uniform profile after training)."

**Implementation:** Scale Q by exp(log_temp) where log_temp is a learnable per-head parameter initialized to 0.

---

### H1: SwiGLU Activation (PRIORITY 1) | Status: CRASH (OOM)

**Domain Pairing: Neuroscience × NLP**

**Domain A** — "Are Human Dendrites Different?" (Fişek & Häusser, 2020, DOI: 10.1016/j.tics.2020.03.002)
- Source primitive: Multiplicative dendritic gating — human cortical dendrites use calcium-mediated multiplicative interactions between dendritic branches to implement XOR-like operations (building on Gidon et al. 2020). This enables richer input-output functions than simple summation.
- Target bottleneck: ReluSquared's elementwise squaring limits the function class of the MLP — it cannot selectively gate one pathway based on another.
- Mapping: Dendritic multiplicative gating allows biological neurons to selectively amplify or suppress inputs depending on co-occurring signals. SwiGLU mirrors this with component-wise product of two projections, where one pathway gates the other.
- Validation: Human L2/3 dendrites with multiplicative interactions compute XOR (impossible for single-compartment neurons), analogous to how GLU variants enable more complex feature interactions than pointwise activations.

**Domain B** — "GLU Variants Improve Transformer" (Shazeer, 2020, arXiv: 2002.05202)
- Source primitive: SwiGLU = (Swish₁(xW) ⊗ xV)W₂, a gated linear unit with Swish activation.
- Target bottleneck: Standard ReLU/ReluSquared FFN layers have limited expressiveness — single-pathway activation.
- Mapping: Adding a second projection (V) that multiplicatively gates the first (W) creates a data-dependent filter that can selectively pass or suppress features.
- Validation: SwiGLU achieves 1.636 log-perplexity vs 1.677 for ReLU on C4 (-2.4% relative improvement). GEGLU achieves 1.633.

**Synthesis:** Biological dendrites achieve computational richness through multiplicative gating between compartments — one branch modulates another's output. SwiGLU implements an analogous mechanism: xV gates Swish(xW) through element-wise multiplication. This cannot be derived from either paper alone: the neuroscience shows WHY multiplicative gating is computationally superior (it expands the function class beyond what additive interactions achieve), while the NLP paper shows HOW to implement it efficiently in transformers.

**Falsifiability:** "val_bpb will decrease because the multiplicative gating enables the MLP to learn more complex feature interactions, and this would NOT appear if the improvement were simply from having more parameters (since ReluSquared with matched parameter count would perform comparably)."

**Mechanistic prediction:** Loss should decrease throughout training, not just early. The improvement should be more pronounced in later layers where features are more abstract and benefit more from selective gating.

**Architectural constraint check:** ✅ Only changes activation function and MLP structure. Adds a gate projection (nn.Linear(n_embd, 4*n_embd)) per layer — new parameters use same optimizer group as existing matrix params (Muon). Does not change any frozen HP or dimension.

**Implementation sketch:**
- Replace MLP.c_fc with two projections: c_gate (for Swish) and c_up (for the gating value)
- Forward: x = SiLU(c_gate(x)) * c_up(x); x = c_proj(x)
- New gate params added to Muon optimizer group with MATRIX_LR

---

### H2: Pre-Attention Causal Depthwise Convolution (PRIORITY 2) | Status: CONFIRMED (val_bpb 0.951991, delta -0.0017)

**Domain Pairing: Speech Recognition × State Space Models**

**Domain A** — "Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., 2020, DOI: 10.21437/interspeech.2020-3015)
- Source primitive: Depthwise separable convolution module within a transformer block, capturing local acoustic features alongside global self-attention.
- Target bottleneck: Self-attention treats each position independently before computing correlations — no local context smoothing.
- Mapping: In speech, phoneme features span multiple frames. Depthwise conv captures these local patterns (formant transitions, co-articulation). Removing conv causes the largest performance drop (2.1%→1.9% WER).
- Validation: Conformer (conv+attention) achieves 1.9%/4.3% WER vs Transformer-only 2.4%/5.6% on LibriSpeech.

**Domain B** — "Hungry Hungry Hippos: Towards Language Modeling with State Space Models" (Dao et al., 2023, arXiv: 2212.14052)
- Source primitive: Shift-SSM — a causal short convolution (shift matrix A) that stores the last m input tokens as state, providing local context before long-range processing via diagonal-SSM.
- Target bottleneck: Attention computes Q/K/V from single token representations, missing local n-gram context.
- Mapping: The shift-SSM preprocesses keys with local context before the outer-product attention-like operation. This allows the model to compare locally-smoothed features rather than point-wise token representations.
- Validation: Hybrid H3-attention (125M) outperforms pure Transformer by 1.0 PPL on OpenWebText. Shift-SSM is critical — removing it eliminates the recall capability.

**Synthesis:** Both speech (Conformer) and language (H3) models benefit from local convolution, but for different reasons: Conformer captures co-articulation in acoustic features, while H3 captures n-gram patterns for associative recall. The cross-domain insight is that attention Q/K projections operate on point-wise representations and benefit from local context smoothing regardless of domain. In our sliding-window architecture (SSSL), short windows (256 tokens) already capture some local context, but Q/K projections still lack sub-token-level local smoothing. A causal depthwise conv of kernel=4 on the input to attention would provide this.

**Falsifiability:** "val_bpb will decrease because Q/K dot products will operate on locally-smoothed features, and this would NOT appear if the sliding window already captures sufficient local context (in which case the conv would be redundant)."

**Mechanistic prediction:** Improvement should be visible from early training (local patterns are learned first). Larger improvement in short-window layers (S) than long-window layers (L).

**Architectural constraint check:** ✅ Adds a small depthwise conv1d (640 channels, kernel=4, causal) before the QKV projections. Very few extra parameters (~2560). Does not change any frozen value.

**Implementation sketch:**
- Add nn.Conv1d(n_embd, n_embd, kernel_size=4, groups=n_embd, padding=3) to CausalSelfAttention
- In forward: x_conv = conv(x.transpose(1,2)).transpose(1,2)[:, :T, :] (causal padding)
- Feed x_conv into Q, K, V projections instead of x
- Conv params added to Muon optimizer (or AdamW with SCALAR_LR since they're small)

---

### H3: Z-Loss on Output Logits (PRIORITY 3) | Status: REFUTED (val_bpb 0.955985, delta +0.004 REGRESSION)

**Domain Pairing: Sparse Expert Training × Statistical Mechanics**

**Domain A** — "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (Zoph et al., 2022, arXiv: 2202.08906)
- Source primitive: Router z-loss: L_z = (1/B) Σ (log Σ exp(x_j))² with coefficient 0.001. Penalizes large logit magnitudes entering softmax to stabilize training.
- Target bottleneck: Large logits cause numerical instability in exponential functions, especially in bfloat16.
- Mapping: By penalizing log-sum-exp squared, z-loss keeps logit magnitudes bounded without hard clipping, maintaining differentiability.
- Validation: Z-loss achieves 100% training stability (3/3 runs) vs 67% without (4/6 runs), while slightly improving quality (-1.741 vs -1.755).

**Domain B** — Free energy minimization in statistical mechanics (Jaynes, 1957; "Information Theory and Statistical Mechanics", Physical Review)
- Source primitive: Maximum entropy principle — the equilibrium distribution maximizes entropy subject to constraints. Systems that concentrate probability too sharply (low entropy) are thermodynamically unstable.
- Target bottleneck: During training, output logits can grow large, concentrating probability mass on few tokens (low-entropy regime), causing poor gradient flow to non-predicted tokens.
- Mapping: Z-loss acts as an entropy-like regularizer that prevents the output distribution from becoming too peaked, analogous to how physical systems resist entropy decrease.
- Validation: The maximum entropy principle underlies all of statistical mechanics — distributions that maximize entropy are most robust to perturbation.

**Synthesis:** Statistical mechanics tells us that sharply peaked distributions are fragile (low entropy = high free energy = thermodynamic instability). Z-loss implements this insight in neural network training: it penalizes large logits that would create peaked output distributions, keeping the model in a "high-entropy regime" where gradient information flows to all tokens. This is distinct from softcap (which hard-clips logits) — z-loss provides a smooth gradient signal that encourages the model to distribute probability more broadly. The model already uses softcap=12, but z-loss addresses the gradient signal (encouraging exploration) rather than just bounding magnitude.

**Falsifiability:** "val_bpb will decrease because z-loss improves gradient flow to low-probability tokens, and this would NOT appear if the softcap already provides sufficient logit regularization (in which case z-loss would be redundant)."

**Mechanistic prediction:** Improvement should appear gradually during warmdown phase when LR drops and the model refines its predictions. Early training loss might be slightly higher (regularization cost), but final val_bpb should be lower.

**Architectural constraint check:** ✅ Adds an auxiliary loss term to the training loop. No new parameters. Does not change any frozen HP. Coefficient 1e-4 is part of the architectural design, not an optimizer HP.

**Implementation sketch:**
- After computing logits and cross-entropy loss, add: z_loss = 1e-4 * torch.logsumexp(logits, dim=-1).square().mean()
- Total loss = ce_loss + z_loss
- Backpropagate through total loss

---

### H4: Sandwich Normalization (Pre+Post Norm) (PRIORITY 4) | Status: CRASH (OOM)

**Domain Pairing: Vision Generation × Deep Network Training**

**Domain A** — "CogView: Mastering Text-to-Image Generation via Transformers" (Ding et al., 2021, arXiv: 2105.13290)
- Source primitive: Sandwich LayerNorm — adding LayerNorm both before AND after each sublayer (attention, MLP). Prevents representation collapse in very deep autoregressive image generation models.
- Target bottleneck: Pre-norm alone allows hidden state magnitudes to grow unboundedly through the residual stream, causing training instability in deep models.
- Mapping: The post-norm bounds the output of each sublayer before it enters the residual stream, preventing any single layer from dominating.
- Validation: CogView (4B params) could not converge without sandwich norm. With it, stable training of 48-layer autoregressive image generation.

**Domain B** — "DeepNet: Scaling Transformers to 1,000 Layers" (Wang et al., 2022, arXiv: 2203.00555)
- Source primitive: SubLN (sub-LayerNorm) — residual scaling + post-norm. Specifically: x + α * PostNorm(Sublayer(PreNorm(x))).
- Target bottleneck: In deep pre-norm transformers, the residual stream grows in magnitude through the layers, causing later layers to have disproportionately small relative contributions.
- Mapping: Post-normalization bounds each sublayer's output, ensuring uniform contribution magnitude across layers.
- Validation: DeepNet scales to 1000+ layers with SubLN, vs <100 layers without. Significant quality improvements at depth 18-50.

**Synthesis:** Both vision generation (CogView) and deep NLP (DeepNet) independently discovered that pre-norm alone is insufficient — adding post-normalization on sublayer outputs is critical for deep models. Our model has only 10 layers but uses aggressive residual scaling (resid_lambdas, x0_lambdas). The post-norm ensures that regardless of the learned residual scales, each sublayer's contribution has bounded magnitude. This is a different mechanism from pre-norm (which normalizes inputs) — post-norm normalizes outputs, controlling what enters the residual stream.

**Falsifiability:** "val_bpb will decrease because post-norm bounds sublayer output magnitude, improving gradient flow through the residual stream, and this would NOT appear if the current pre-norm + residual scaling already achieves sufficient magnitude control (in which case post-norm would be redundant or harmful due to over-normalization)."

**Mechanistic prediction:** Training loss may be slightly higher initially (double norm reduces effective learning rate), but should converge to a lower final value. The improvement should be more visible in the warmdown phase.

**Architectural constraint check:** ✅ Adds RMSNorm calls on sublayer outputs. No new parameters (RMSNorm has no learnable params in this codebase). Does not change any frozen value.

**Implementation sketch:**
- In Block.forward: x = x + norm(self.attn(norm(x), ve, cos_sin, window_size))
- And: x = x + norm(self.mlp(norm(x)))
- Just wrap each sublayer output in an additional norm() call
