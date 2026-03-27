# Hypotheses

## Engineering Run Counter: 3/5 | Phase: ENGINEERING

---

## Hypothesis 5 | Status: REFUTED | Priority: 1

**Diagnosis:** The model has separate wte (5.2M params) and lm_head (5.2M params) embeddings. The lm_head is initialized to small random values (std=0.001) while wte is standard normal. These 10.4M params (12% of total) are not sharing any structure, meaning the model must independently learn compatible encoding and decoding representations.

**Domain A:** Information theory — channel coding and matched filters. In communications, a matched filter (decoder) that is the transpose of the encoder achieves maximum SNR for signal recovery. The encoder and decoder share structure, which eliminates the need to independently optimize both ends of the channel. This is the fundamental principle behind matched encoding/decoding.

**Domain B:** Weight tying in transformers (Press & Wolf 2017, "Using the Output Embedding to Improve Language Models"). Sharing the input embedding (wte) and output projection (lm_head) reduces parameters and forces the embedding space to be useful for both token encoding and prediction. GPT-2, T5, and many modern models use this.

**Synthesis:** Apply weight tying to our model, where wte and lm_head share weights. The non-obvious interaction: our optimizer uses different LRs for embeddings (0.6 * scale) vs unembedding (0.004 * scale). With tying, the shared matrix receives gradients from BOTH paths. We should use the EMBEDDING LR (higher) since the embedding gradient dominates. Additionally, value embeddings (26.2M params) that index by token ID create a "virtual embedding" pathway — tying wte/lm_head may improve these too by making the token embedding space more coherent. The 5.2M freed params reduce optimizer state by ~20MB.

**Obviousness check:** Weight tying is well-known. The non-obvious aspect is the interaction with our dual-LR optimizer (embeddings at 0.6 vs unembedding at 0.004) and the downstream effect on value embeddings through the shared token space.

**Falsifiability:** "val_bpb will decrease because the shared embedding provides a matched encoding/decoding codebook that's more sample-efficient, and this improvement would NOT appear if the input and output embedding spaces were already well-aligned (testable: if the cosine similarity between wte and lm_head rows is already high, tying won't help)."

**Novelty check:** Information theory alone suggests matched filters. Weight tying alone is standard. The specific hypothesis about how the dual-LR conflict resolves (and whether the value embedding pathway benefits) requires reading both.

**Mechanistic prediction:** Early training should improve faster (the shared embedding learns from prediction loss immediately, instead of through indirect gradient flow). The improvement should be 0.002-0.008 val_bpb. If it hurts, it's because the embedding and unembedding tasks genuinely need different representations at this model size.

**Implementation sketch:** In GPT.__init__, set `self.lm_head.weight = self.transformer.wte.weight` (parameter sharing). In setup_optimizer, merge lm_head_params into embedding_params group (use embedding LR). Update init_weights to not double-initialize. Update parameter count assertion.

---

## Hypothesis 6 | Status: REFUTED | Priority: 2

**Diagnosis:** The SSSL window pattern uses 256-token short windows for 3/4 of layers. FA3 with window_size=256 processes more attention FLOPs than necessary for local syntactic patterns, which typically span <64 tokens.

**Domain A:** Signal processing — Nyquist sampling theorem and bandwidth allocation. A signal with bandwidth B can be perfectly reconstructed from samples taken at rate 2B. Over-sampling wastes resources; the optimal sampling rate is exactly 2×bandwidth. Similarly, an attention window wider than the pattern it captures wastes compute.

**Domain B:** Sliding window attention in transformers (Longformer, BigBird). Short-window layers handle local patterns (syntax, word-level dependencies) while long-window layers handle global context (topic, long-range coreference). The optimal window size depends on the pattern bandwidth.

**Synthesis:** Reduce short-window from 256 to 128 tokens. For a 10-layer model with SSSL pattern, layers 0-2 (short window) handle local patterns. Natural language syntax rarely spans >64 tokens; 128 is already generous. Reducing from 256→128 cuts attention FLOPs for these layers by ~2×, increasing throughput and getting more training steps. The key insight from Nyquist: 128 tokens is >2× the typical syntactic dependency span, so no information is lost.

**Obviousness check:** Changing window size is straightforward. The Nyquist framing (128 > 2× syntactic bandwidth ≈ 50 tokens) provides a principled rationale rather than arbitrary choice.

**Falsifiability:** "val_bpb will decrease because the throughput gain (more steps) outweighs the reduced attention context, and this would NOT appear if the short-window layers need >128 tokens of context (i.e., if medium-range dependencies in layers 0-2 are load-bearing)."

**Novelty check:** The Nyquist analogy alone doesn't apply to attention windows. Sliding window attention alone doesn't invoke bandwidth arguments. The combination provides a principled window size choice.

**Mechanistic prediction:** Throughput should increase ~5-10% (attention is ~35% of FLOPs, 3/4 of layers use short window, and those windows get 2× cheaper). This gives ~90-180 more steps. If val_bpb improves, the extra steps matter more than the lost context. Expected: 0.001-0.005 improvement.

**Implementation sketch:** Change `short_window = 256` to `short_window = 128` in `_compute_window_sizes`. One line change.

---

## Hypothesis 7 | Status: CONFIRMED (marginal) | Priority: 3

**Diagnosis:** The Muon LR of 0.03 was optimized at 10×512 (60M params, v2). At 10×640 (85.9M params), the dmodel_lr_scale adjusts AdamW LRs by 1/√(640/768) ≈ 1.095, but the Muon LR is NOT scaled. Muon operates on orthogonal manifolds where the update magnitude depends on matrix dimensions differently.

**Domain A:** Dynamical systems theory — critical slowing down near phase transitions. In dynamical systems approaching a bifurcation point, the relaxation time diverges (critical slowing down). The system needs stronger driving forces to escape the slow-convergence regime near the critical point.

**Domain B:** Muon optimizer with Newton-Schulz orthogonalization. The orthogonalized gradient preserves direction but the LR controls step size on the manifold. With more parameters (640 vs 512 width), the loss landscape is higher-dimensional, and the model may be in a "critically slow" convergence regime where a slightly higher LR would explore more efficiently.

**Synthesis:** Increase Muon LR from 0.03 to 0.04. The dynamical systems analogy suggests that wider models (more dimensions) need stronger driving to avoid critical slowing down. The Newton-Schulz orthogonalization ensures the update direction remains well-conditioned regardless of LR, so higher LR primarily affects exploration speed on the orthogonal manifold. At 0.03, the model may be under-exploring — small LR causes it to follow overly conservative paths on the manifold.

**Obviousness check:** LR tuning is standard. The dynamical systems framing (critical slowing down at higher dimensions) provides a specific prediction about the direction of change.

**Falsifiability:** "val_bpb will decrease because higher Muon LR escapes slow-convergence regions faster, and this would NOT appear if 0.03 is already at the stability boundary (in which case 0.04 would cause divergence or instability)."

**Mechanistic prediction:** If the model is under-exploring, higher LR should improve mid-training loss (faster descent). If it's near the stability boundary, loss will spike early. Expected: 0.001-0.005 improvement OR divergence (informative either way).

**Implementation sketch:** Change `MATRIX_LR = 0.03` to `MATRIX_LR = 0.04`.

---

## Hypothesis 8 | Status: UNTESTED | Priority: 4 | [EXPLORATORY]

**Diagnosis:** The softcap of 12 was optimized for batch=8 regime (v1, MPS). At batch=262K (H100), gradient noise is dramatically lower, and the model's logit distribution may be different. The softcap acts as a regularizer — too tight loses information, too loose allows overconfident predictions.

**Domain A:** Thermodynamics — temperature and entropy. In statistical mechanics, the temperature T controls the Boltzmann distribution's entropy. Higher temperature = more uniform distribution = higher entropy. The softcap acts analogously to an inverse temperature — tighter cap = higher "temperature" = more entropy in the output distribution. At a given model capability, there's an optimal temperature that balances confidence (low entropy) with calibration (not over-committing).

**Domain B:** Logit soft-capping in transformers (Gemma, PaLM). Softcap = C means logits are bounded by [-C, C], preventing the model from making infinitely confident predictions. This acts as implicit label smoothing with a learnable smooth boundary.

**Synthesis:** Test softcap=10 (tighter) for the H100 262K batch regime. The thermodynamic analogy: with 262K batch (very low gradient noise), the model can afford to be more confident — but the softcap prevents this. Counterintuitively, a TIGHTER softcap might help because it forces the model to spread probability mass more evenly across plausible continuations, which is measured by BPB. At batch=8 (high noise), softcap=12 was optimal because tighter caps amplified the noisy gradients. At batch=262K (low noise), tighter caps provide cleaner regularization without the noise amplification.

**Obviousness check:** Softcap tuning is straightforward. The thermodynamic argument about batch-size-dependent optimal temperature is non-obvious.

**Falsifiability:** "val_bpb will decrease because tighter softcap at low gradient noise provides better implicit regularization, and this would NOT appear if softcap=12 is already optimal regardless of batch size."

**Mechanistic prediction:** The improvement, if any, should be concentrated in late training when the model approaches its final calibration. Expected: 0.001-0.003 improvement or neutral.

**Implementation sketch:** Change `softcap = 12` to `softcap = 10` in the forward method.

---

## Hypothesis 1 | Status: INCONCLUSIVE (OOM) | Priority: 1

**Diagnosis:** Attention maps in shallow 10-layer model likely overallocate to irrelevant context; with only 5 heads and 128-dim head, each head must cover too many attention patterns, mixing signal with noise.

**Domain A:** Differential amplifiers in analog electronics — source primitive: Common-mode rejection. A differential amplifier takes two inputs and outputs their difference, rejecting any signal component common to both (noise/interference) while amplifying the differential signal. The common-mode rejection ratio (CMRR) quantifies noise suppression quality.

**Domain B:** Differential Transformer (arXiv 2410.05258) — source primitive: Differential attention. Splits each head's Q,K into two halves (Q1,Q2,K1,K2 each with head_dim/2), computes two separate softmax attention maps, and subtracts: `DiffAttn = (softmax(Q1K1^T/√d) - λ·softmax(Q2K2^T/√d))V`. Lambda is learned per-layer with init `λ_init = 0.8 - 0.6·exp(-0.3·(l-1))`. Post-subtraction GroupNorm normalizes each head. Result: sparse, noise-cancelled attention patterns.

**Synthesis:** Apply differential attention to our 10×640 model, but with a key architectural adaptation: our model already applies QK-norm (`norm(q), norm(k)`) before attention, which normalizes both Q1K1 and Q2K2 attention maps to similar scales *before* the softmax. This acts as "matched impedance" in the differential amplifier analogy — ensuring both subtracted signals are on the same scale, which should improve the common-mode rejection ratio compared to the standard Diff Transformer that relies on post-hoc GroupNorm. Neither the electronics CMRR principle nor the Diff Transformer paper proposes pre-normalized differential attention — this combination emerges from reading them together.

**Obviousness check:** A competent ML engineer would know about Diff Transformer, but the specific interaction with pre-existing QK-norm as an analog of impedance matching is non-obvious. The standard Diff Transformer uses post-subtraction normalization; ours would use pre-subtraction normalization via QK-norm, which changes the optimization dynamics.

**Falsifiability:** "val_bpb will decrease because differential attention cancels the noise floor in attention maps (common-mode component), and this improvement would NOT appear if the model's attention patterns were already sparse and noise-free (i.e., if the bottleneck is MLP capacity rather than attention quality)."

**Novelty check:** The electronics CMRR principle alone doesn't suggest applying it to softmax maps. The Diff Transformer paper alone doesn't consider pre-normalized QK as a substitute for post-subtraction GroupNorm. The combination — leveraging existing QK-norm as the "matching" mechanism — requires reading both.

**Mechanistic prediction:** Early training loss should drop faster because the noise-cancelled attention provides cleaner gradient signal from the start. The improvement should be most visible in the first 30% of training when attention patterns are still forming. Final val_bpb improvement of 0.003-0.008. If the improvement is only in late training, that would suggest a regularization effect rather than noise cancellation.

**Implementation sketch:** In `CausalSelfAttention.forward`: after computing q, k, v and applying RoPE and QK-norm, split q and k along the last dimension into halves (q1, q2, k1, k2). Run two FA3 calls: `out1 = FA3(q1, k1, v)` and `out2 = FA3(q2, k2, v)`. Compute `y = out1 - lambda * out2` where lambda is a learned per-layer parameter. If FA3 doesn't support different Q/K dim vs V dim, fall back to splitting V too and doubling heads. Add `self.diff_lambda` as a learnable parameter initialized per the paper's formula. Skip the post-subtraction GroupNorm since QK-norm provides pre-normalization.

---

## Hypothesis 2 | Status: REFUTED | Priority: 2

**Diagnosis:** ReluSquared MLP creates extreme sparsity (~50%+ of intermediate activations are zero), meaning the effective dimension of the 2560-dim intermediate layer is ~1200-1400. The model has 12.2 GB of VRAM headroom (67.8 used of 80 GB), leaving room for wider MLPs.

**Domain A:** Compressive sensing / sparse signal recovery (Candès, Romberg, Tao 2006; Donoho 2006) — source primitive: The Restricted Isometry Property (RIP). For sparse signals, recovery from compressed measurements requires the measurement matrix to have enough columns (measurements) relative to the signal's sparsity level. Specifically, if a signal has sparsity s, we need O(s·log(n/s)) measurements for reliable recovery. Insufficient width leads to information loss that no algorithm can recover.

**Domain B:** ReluSquared MLP in transformers — source primitive: The MLP intermediate layer acts as a "measurement matrix" that projects tokens into a higher-dimensional space where sparse feature selection (ReluSquared) identifies relevant features. The c_proj then "recovers" the output from this sparse representation.

**Synthesis:** The compressive sensing analogy reveals that 4× expansion may be insufficient given ReluSquared's heavy sparsification. If ~50% of activations are zero, the effective measurement count is ~2×model_dim, barely above the information-theoretic minimum. Increasing to 6× (3840-dim intermediate) provides ~3× effective measurements, moving well above the recovery threshold. This should allow the model to represent more token-specific features without information loss. The non-obvious part: with standard GELU/SiLU (which are not truly sparse), wider MLPs just add parameters. With ReluSquared's hard sparsity, wider MLPs qualitatively change the recovery guarantee.

**Obviousness check:** "Make the MLP wider" is obvious. The insight that ReluSquared's binary sparsity makes this fundamentally different from widening a GELU MLP — because it changes the information-theoretic recovery bound — is not obvious. A wider GELU MLP is just more compute; a wider ReluSquared MLP crosses a phase transition in representational capacity.

**Falsifiability:** "val_bpb will decrease because wider MLPs increase the effective number of non-zero features per token above the sparse recovery threshold, and this improvement would NOT appear if the bottleneck is attention capacity rather than MLP representational capacity (in which case we'd see no change or regression due to throughput loss)."

**Novelty check:** Compressive sensing alone doesn't suggest MLP widths. The MLP architecture alone doesn't invoke information-theoretic recovery bounds. The combination — treating ReluSquared's sparsity as a compressive sensing problem — is a genuine cross-domain transfer.

**Mechanistic prediction:** Training loss should improve uniformly across all training (not just early or late), because the improvement is in per-token representational capacity. The throughput will drop ~15-20% (more MLP FLOPs) but if the hypothesis is correct, the quality improvement per step will more than compensate. Expected val_bpb improvement: 0.005-0.015. Parameter count increases from 85.9M to ~102M.

**Implementation sketch:** Change `MLP.__init__` to use `6 * config.n_embd` instead of `4 * config.n_embd` for the intermediate dimension. Monitor VRAM — should be within 80GB. If OOM, try 5× instead. No other changes needed.

---

## Hypothesis 3 | Status: REFUTED | Priority: 3

**Diagnosis:** The model trains for 1790 steps with 70% warmdown (cosine decay from full LR to 0.01× LR). The final model snapshot captures a single point on the parameter trajectory, not the trajectory's average. With Muon's orthogonalized updates, late-stage parameters oscillate around the minimum in a rotation-dominated regime (norm-preserving), meaning averaging would capture the "center" of these rotations.

**Domain A:** Polyak-Ruppert averaging in stochastic approximation (Polyak & Juditsky 1992) — source primitive: For a stochastic gradient method with step size η, averaging the iterates x̄_n = (1/n)Σx_i achieves the optimal O(1/n) convergence rate regardless of η, as long as η doesn't vanish too fast. The key insight: individual iterates oscillate, but their average converges at the optimal rate.

**Domain B:** Schedule-Free Learning (arXiv 2405.15682) — source primitive: Unifies LR scheduling and iterate averaging. Maintains three sequences: z (optimizer iterates), y (gradient evaluation point), x (weighted average for evaluation). The weighted average x achieves performance matching or exceeding cosine schedules without requiring knowledge of stopping time T. Won MLCommons 2024 AlgoPerf challenge.

**Synthesis:** Rather than replacing our entire optimizer with Schedule-Free (which would require rewriting Muon), apply the core insight — iterate averaging — as a lightweight post-hoc mechanism. Maintain an EMA shadow copy of model weights during training, and evaluate using the EMA model. The non-obvious interaction: Muon's orthogonalized updates preserve weight matrix norms (they act as rotations in matrix space). EMA of rotation matrices averages over the Lie group SO(n), and for small perturbations this approximates the Fréchet mean. This means EMA is particularly well-suited to Muon because the averaging operates in a geometrically meaningful way — unlike Adam where EMA averages over both scale and direction.

**Obviousness check:** EMA is well-known. The insight that Muon's norm-preserving updates make EMA geometrically meaningful (averaging rotations vs averaging arbitrary vectors) is non-obvious and changes the expected EMA decay rate.

**Falsifiability:** "val_bpb will decrease because EMA captures the center of Muon's rotational oscillations in late training, and this improvement would NOT appear if the model converges to a sharp minimum (where the trajectory doesn't oscillate) or if Muon's updates are not rotation-dominated."

**Novelty check:** Polyak averaging alone doesn't consider orthogonalized optimizers. Schedule-Free alone doesn't consider the geometry of Muon updates. The combination — EMA as Fréchet mean approximation for Muon's SO(n) trajectories — requires understanding both.

**Mechanistic prediction:** The EMA model should primarily improve over the final snapshot during the warmdown phase, when the LR is dropping and individual iterates oscillate at a scale proportional to LR. The improvement should be 0.001-0.005 val_bpb, concentrated entirely in late training. If the improvement is early-training dominant, it would indicate averaging is compensating for LR noise rather than late-stage oscillation.

**Implementation sketch:** After the optimizer step, update shadow parameters: `ema_param = decay * ema_param + (1-decay) * param`. Use decay=0.995 (average over ~200 steps ≈ last ~11% of training). Before final evaluation, copy EMA params into the model. Memory cost: ~170MB extra for shadow weights (85.9M params × 2 bytes bf16). Implement as a simple loop after `optimizer.step()`.

---

## Hypothesis 4 | Status: REFUTED | Priority: 4 | [EXPLORATORY]

**Diagnosis:** Each token position receives gradient signal only from its own next-token prediction loss. With 85.9M parameters and 1790 steps of 262K tokens, the model sees ~469M tokens total. Richer gradient signal per token could improve sample efficiency.

**Domain A:** Multi-task learning theory (Caruana 1997; Ruder 2017) — source primitive: Auxiliary tasks act as an inductive bias, sharing representations across related objectives. The gradient from auxiliary tasks provides implicit regularization by constraining the shared representation to be useful for multiple purposes. The key mechanism: auxiliary gradients from related tasks smooth the loss landscape and reduce the effective dimensionality of the optimization problem.

**Domain B:** Multi-token Prediction (arXiv 2404.19737, Meta) — source primitive: At each position, predict the next n tokens using n independent output heads on the shared trunk. Each head is a separate unembedding projection. The trunk receives gradients from all n heads, providing n× richer gradient signal per forward pass. Memory-efficient implementation computes head losses sequentially. Benefits are "increasingly useful for larger model sizes" and especially strong for "development of induction heads."

**Synthesis:** For our 85.9M model with a 7.5-minute time budget, even n=2 (predict next 2 tokens) would double the gradient signal per position. The non-obvious interaction with Muon: Muon orthogonalizes the gradient before applying it. With multi-token prediction, the gradient has contributions from two different prediction targets at each position. Muon's Newton-Schulz orthogonalization of this richer gradient should produce updates that are orthogonal in the space of *multi-task useful directions*, not just single-task directions. This could be qualitatively different from simply training longer — it changes the *manifold* that Muon orthogonalizes onto.

**Obviousness check:** Multi-token prediction is known. The interaction with Muon's orthogonalization — that the richer gradient changes the subspace Muon projects onto — is non-obvious and has not been studied.

**Falsifiability:** "val_bpb will decrease because the auxiliary second-token prediction provides gradient signal about future context that improves the shared representations, and this improvement would NOT appear if the shared trunk already captures sufficient future context (testable: if single-token loss doesn't improve while only the auxiliary head improves, the synthesis failed)."

**Novelty check:** Multi-task learning theory alone suggests auxiliary objectives. Multi-token prediction alone doesn't consider orthogonalized optimizers. The specific hypothesis — that Muon+multi-token changes the effective optimization manifold — requires both.

**Mechanistic prediction:** Training loss on the primary (next-token) objective should decrease faster in early-to-mid training, when the auxiliary signal provides the most new information. Late training benefit may be smaller as both tasks converge. The overhead is one extra unembedding projection (640×32768 = ~21M params) plus one extra forward pass through it per step. Expected val_bpb improvement: 0.002-0.010, but high uncertainty — this is exploratory. Throughput will decrease ~10-15%.

**Implementation sketch:** Add a second `lm_head_2` (nn.Linear, same shape as lm_head). In the forward pass, compute `logits_2 = lm_head_2(x)` with targets shifted by 2 (instead of 1). Total loss = primary_loss + 0.5 * aux_loss. The auxiliary head uses the same softcap. Only the primary loss is used for val_bpb evaluation. Init lm_head_2 the same way as lm_head. Add lm_head_2 to the AdamW optimizer group (unembedding LR).
