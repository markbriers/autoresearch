# Hypotheses

## Hypothesis 1 | Status: REFUTED | Priority: 1

**Diagnosis:** The additive residual stream in deeper models (10+ layers) suffers from norm growth, requiring resid_lambdas and x0_lambdas as band-aids. The optimization landscape is curved (non-Euclidean), but additive residuals treat it as flat.

**Domain A:** nGPT: Normalized Transformer with Representation Learning on the Hypersphere (2410.01131) — source primitive: LERP residual connections with per-dimension learnable rates (eigen learning rates). Each block proposes a direction; `h <- Norm(h + alpha * (h_block - h))` interpolates instead of adding. Removes need for LayerNorm. Claimed 4-20x training speedup.

**Domain B:** ReluSquared activation sparsity in our MLP — source primitive: ~50% of MLP neurons produce exact zeros after ReLU, and squaring amplifies surviving activations. MLP output vectors are inherently sparse and directional (pointing toward specific feature subsets).

**Synthesis:** Apply LERP residuals ONLY to the MLP branch, keeping additive residuals for attention. This is non-obvious because nGPT applies LERP uniformly to both branches. Our insight: ReluSquared MLP outputs are sparse directional vectors (they "point toward" detected features), making LERP interpolation natural — each MLP block rotates the hidden state toward its detected features, with alpha controlling rotation magnitude. Attention outputs are dense superpositions of many positions' values, better modeled by additive accumulation (maintaining superposition). This asymmetric treatment exploits the structural difference between sparse MLP and dense attention outputs that nGPT's uniform approach ignores.

**Falsifiability:** "val_bpb will decrease because MLP-branch LERP prevents residual norm growth while preserving the directional sparsity signal, and this improvement would NOT appear if the benefit came solely from norm control (in which case symmetric LERP on both branches would be equally good or better)."

**Prediction:** Loss curve should show faster early convergence (LERP stabilizes the residual stream immediately) with a steeper descent in the first 30% of training. Final val_bpb improvement of 0.002-0.005. If attention-branch LERP helps equally, the mechanistic story about ReluSquared sparsity is wrong.

**Implementation sketch:** In `Block.forward`, change the MLP residual from `x = x + self.mlp(norm(x))` to `mlp_out = norm(self.mlp(norm(x))); x = norm(x + alpha_m * (mlp_out - x))` where `alpha_m` is a learnable per-dimension parameter initialized to `1/n_layer` (~0.1 for 10 layers). Keep attention as additive: `x = x + self.attn(...)`. Remove resid_lambdas and x0_lambdas (subsumed). Add norm after each LERP step.

**RESULT: REFUTED** — val_bpb 1.080 (+0.103 regression). The per-dimension scaling fundamentally conflicts with Muon's Newton-Schulz orthogonalization. Muon normalizes gradient direction; element-wise alpha scaling re-introduces non-uniform magnitude that breaks the orthogonalization invariant. The mapping was wrong: nGPT uses standard optimizers (Adam) where per-dimension scaling is compatible; Muon's orthogonal update structure makes per-dimension output scaling destructive.

---

## Hypothesis 2 | Status: REFUTED | Priority: 2

**Diagnosis:** VE (Value Embedding) parameters consume ~50% of total model params at depth 10 (5 VE layers × vocab_size × kv_dim). This inflates the model, reducing throughput and limiting depth scaling. Prior runs showed depth 20 failed because VE params reached 73M.

**Domain A:** DeepSeek-V2 Multi-head Latent Attention (2405.04434) — source primitive: Low-rank KV compression via latent vector. Projects K,V through a compressed bottleneck `c = W_down * x`, then `K = W_up_k * c`, `V = W_up_v * c`. Reduces KV cache by 93% while maintaining or improving quality vs MHA.

**Domain B:** Information bottleneck principle (Tishby) — source primitive: Optimal representations compress input while preserving task-relevant information. Over-parameterized representations capture noise; bottleneck forces learning of sufficient statistics.

**Synthesis:** Apply MLA-style low-rank factorization to VE embeddings. Replace `nn.Embedding(vocab_size, kv_dim)` with `nn.Embedding(vocab_size, d_c)` + `nn.Linear(d_c, kv_dim)` where d_c << kv_dim. This is non-obvious because MLA compresses across the TIME axis (reducing KV cache per sequence position), while our synthesis compresses across the VOCABULARY axis (reducing the dimensionality of token-to-value mappings). The mathematical structure (low-rank factorization as information bottleneck) is the same, but applied to a completely different data axis. Neither paper proposes this specific combination.

**Falsifiability:** "val_bpb will decrease because the bottleneck forces VE to learn a compressed token-identity representation that generalizes better, and this improvement would NOT appear if VE's value comes from having independent full-rank embeddings per token (in which case compression would always hurt)."

**Prediction:** With d_c = 128 (4× compression for kv_dim=512), VE params drop from ~84M to ~21M. This frees compute for more depth. At iso-params: test depth 12 with low-rank VE vs depth 10 with full VE. Loss curve should show similar early behavior but better final convergence due to bottleneck regularization. Expected improvement: 0.001-0.003.

**Implementation sketch:** Create a `LowRankEmbedding` class wrapping `nn.Embedding(vocab_size, d_c)` + `nn.Linear(d_c, kv_dim, bias=False)`. Replace `self.value_embeds` entries with `LowRankEmbedding`. Set d_c = 128. Optionally increase DEPTH to 12 to use freed param budget.

**RESULT: REFUTED** — val_bpb 0.988 (+0.010 regression). The bottleneck compressed effectively (50.3M→38M params with only +0.010 loss) but the prediction that compression would IMPROVE quality through regularization was wrong. The target bottleneck diagnosis was correct (VE is over-parameterized) but the mapping was wrong: the information bottleneck benefits of MLA arise from compressing across the time axis (redundancy in sequential KV cache), while VE compresses across the vocabulary axis where tokens genuinely need distinct default values. Per-token VE diversity is load-bearing, not redundant.

---

## Hypothesis 3 | Status: REFUTED | Priority: 3

**Diagnosis:** The softcap mechanism (`tanh(logits/cap) * cap`) is a blunt sigmoid compression that treats all logit dimensions equally. It was tuned to cap=12 but this value is fragile across configurations. A more principled logit bounding mechanism could both simplify the architecture and improve robustness.

**Domain A:** nGPT (2410.01131) — source primitive: Cosine similarity logits. Normalize both the final hidden state and the unembedding weight matrix, producing logits in [-1, 1]. A learnable per-vocabulary temperature scaling vector `s_z` restores expressivity.

**Domain B:** Softcap mechanism in current architecture — source primitive: Logit soft-capping via tanh provides implicit regularization that prevents overconfident predictions during noisy early training, critical for stable convergence with Muon optimizer.

**Synthesis:** Replace `softcap * tanh(logits/softcap)` with normalized dot product + learnable temperature. Normalize the last hidden state and lm_head weights (both to unit norm), compute cosine similarity logits in [-1, 1], multiply by a single learnable scalar temperature `tau` initialized to 30. This is non-obvious because nGPT normalizes ALL layers (full architecture change), while we selectively apply normalization ONLY at the output layer, combining with the existing RMS-normed residual stream. The synthesis is: use nGPT's output mechanism without its input/residual mechanism, treating the output normalization as a principled replacement for softcap rather than part of a full architectural rewrite.

**Falsifiability:** "val_bpb will decrease because learned temperature provides per-training-phase-optimal logit scaling, and this would NOT appear if the fixed softcap=12 is already optimal (in which case learned temperature would converge to ~12 and show no improvement)."

**Prediction:** Loss curve should show smoother early training (no abrupt gradient from tanh saturation). The learned temperature should start near 30 and adjust during training. If it converges to ~12, the hypothesis is uninformative. Expected improvement: 0.001-0.003, or the learned temperature reveals the true optimum is far from 12.

**Implementation sketch:** In `GPT.forward`: (1) normalize `x` before lm_head, (2) normalize lm_head weight rows at init (and re-normalize periodically or use F.normalize in forward), (3) replace softcap with `logits = tau * F.cosine_similarity(x.unsqueeze(-2), lm_head.weight.unsqueeze(0), dim=-1)` where tau is a learnable scalar. Or simpler: `logits = tau * F.linear(F.normalize(x, dim=-1), F.normalize(self.lm_head.weight, dim=-1))`.

---

## Hypothesis 4 | Status: UNTESTED | Priority: 4 | [EXPLORATORY] | DEFERRED

**Diagnosis:** In time-budgeted training, FLOPs per step directly trade off with step count. Current architecture spends equal compute on all tokens at all layers, but token difficulty varies enormously — function words and repeated patterns need less processing than rare/novel content.

**Domain A:** Mixture-of-Depths (2404.02258) — source primitive: Top-k token routing per layer. A linear router selects which tokens participate in block computation; others skip via residual identity. 12.5% capacity (87.5% of tokens skip) on alternating layers matches baseline quality at ~50% fewer FLOPs per step.

**Domain B:** Time-budgeted training throughput constraint — source primitive: In our 7.5-min regime, any mechanism that reduces FLOPs per forward pass translates directly to more optimization steps. A 30% FLOPs reduction → ~43% more steps (steps ∝ 1/(1-reduction)).

**Synthesis:** Apply MoD with 50% capacity on alternating layers. The non-obvious element: in time-budgeted training, the optimal capacity ratio shifts from MoD's recommended 12.5% toward higher capacity (50%) because we need quality per step, not just throughput. MoD was optimized for fixed-step training where total compute is constant; in time-budgeted training, we get FREE extra steps from the throughput gain, changing the quality-throughput tradeoff. We predict 50% capacity is better than 12.5% in our regime because each step must count more.

**Falsifiability:** "val_bpb will decrease because the throughput gain (more steps) outweighs the per-step quality loss from token routing, and this would NOT appear if the router's learned token selection is random-quality (in which case stochastic depth with the same skip rate would perform equally)."

**Prediction:** With 50% capacity on 5 alternating layers (of 10), expect ~20-30% more steps. Loss curve should show slightly higher per-step loss but lower final val_bpb due to more total optimization. If the router doesn't learn meaningful token selection (verified by checking if router weights correlate with token loss), the mechanism is just stochastic depth with extra overhead.

**Implementation sketch:** Add `self.router = nn.Linear(config.n_embd, 1, bias=False)` to Block. In forward: compute router scores, select top-50% tokens, route selected through attention+MLP, skip others via identity. Apply on alternating layers (even indices). Use straight-through estimator for gradients through the top-k selection.

---

# Cycle 2 Hypotheses (post Muon-compatibility discovery)

## Hypothesis 5 | Status: REFUTED | Priority: 1

**Diagnosis:** With 2401 steps at 10×512, the architecture is compute-bound. Attention and MLP are computed sequentially within each block, creating a pipeline bottleneck. MLP's input depends on attention output (serial dependency), but in practice the attention-to-MLP information flow may be weak enough that running them in parallel preserves quality while improving throughput.

**Domain A:** PaLM: Scaling Language Modeling with Pathways (2204.02311) — source primitive: Parallel attention+MLP formulation. Instead of sequential `x' = x + attn(norm(x)); y = x' + mlp(norm(x'))`, use parallel `y = x + attn(norm(x)) + mlp(norm(x))`. Both branches see the same normalized input. Eliminates one sequential dependency per layer.

**Domain B:** Amdahl's Law (computer science) — source primitive: Overall speedup from parallelizing a fraction of sequential work. In time-budgeted training, any throughput gain translates directly to more steps. The serial attention→MLP dependency is the sequential fraction limiting speedup.

**Synthesis:** Apply PaLM-style parallel blocks to our architecture. The non-obvious element: our architecture uses Muon optimizer which orthogonalizes gradients. With sequential blocks, MLP gradients are computed through the attention output. With parallel blocks, MLP gradients are independent of attention — this changes the gradient landscape in a way that might be MORE compatible with Muon's per-parameter-group orthogonalization (each group gets cleaner gradients from its own branch). Neither PaLM nor Muon papers consider this interaction.

**Falsifiability:** "val_bpb will decrease because parallel formulation increases throughput (more steps) while the quality loss from losing attention→MLP information flow is small, and this would NOT appear if attention→MLP information flow is critical for this architecture (in which case sequential would be strictly better regardless of step count)."

**Prediction:** ~10-15% throughput increase → ~240-360 more steps. Quality per step should be slightly worse (PaLM reports 0-1% quality loss at large scale, might be larger at our scale). Net effect: improvement if throughput gain > quality loss. Loss curve should show parallel model behind sequential in early steps but catching up due to more total steps.

**Implementation sketch:** In `Block.forward`, change from sequential to parallel: `x = x + self.attn(norm(x), ve, cos_sin, window_size) + self.mlp(norm(x))`. This is a one-line change. The pre-norm is computed once and shared.

---

## Hypothesis 6 | Status: UNTESTED | Priority: 2

**Diagnosis:** With 2401 steps and 10 layers, training at this duration is well beyond the "short training" regime. The warmdown scaling (0.7) already accounts for step count. The remaining bottleneck is model quality per step — specifically, the allocation of representational capacity across layers. All 10 layers have identical structure, but lower layers learn simpler patterns (local syntax) while upper layers learn complex patterns (semantics). Uniform capacity is suboptimal.

**Domain A:** Mixture-of-Depths (2404.02258) — source primitive: Top-k token routing per layer with identity skip for non-selected tokens. 12.5% capacity on alternating layers matches baseline quality.

**Domain B:** Stochastic depth (Huang et al., 2016) — source primitive: During training, randomly drop layers with probability increasing from 0 (first layer) to p_max (last layer). At inference, use all layers with outputs scaled by survival probability. Provides implicit ensemble regularization.

**Synthesis:** Use STOCHASTIC depth (not learned routing) with linearly increasing drop probability. This is simpler than MoD (no router parameters, no top-k), preserves Muon compatibility (just masks gradients, doesn't change their structure), and provides both regularization AND throughput improvement. The non-obvious element: combine stochastic depth with our x0_lambdas skip connections. When a layer is dropped, the x0_lambda still connects the input directly to the output — the model naturally learns to rely more on x0 connections for later layers, creating an implicit depth-adaptive architecture that gracefully degrades when layers are missing.

**Falsifiability:** "val_bpb will decrease because stochastic depth provides ensemble regularization + throughput (more steps), and this would NOT appear if the model is already well-regularized (in which case the throughput gain would be offset by quality loss from missing layers)."

**Prediction:** With p_max=0.2 (last layer drops 20% of the time), expect ~10% throughput gain and mild regularization benefit. Loss curve should show slightly higher variance (due to stochasticity) but lower final val_bpb.

**Implementation sketch:** In `GPT.forward`, before each block: compute drop probability `p = (i / (n_layer-1)) * p_max` where p_max=0.2. During training, skip block with probability p. Scale output by `1/(1-p)` when not skipped. At eval, use all blocks (no scaling needed since init accounts for scaling). Keep resid_lambdas/x0_lambdas applied regardless of skip.

---

## Hypothesis 7 | Status: UNTESTED | Priority: 3

**Diagnosis:** Softcap=12 was tuned with ~500-1000 steps on H100. With 2401 steps, the model has more training to overcome early instability, potentially benefiting from LESS regularization (higher softcap) to achieve lower final loss.

**Domain A:** Minimum Description Length principle (information theory) — source primitive: Optimal model complexity depends on data size. More data/training → more complex model is justified. Softcap limits logit complexity; optimal cap should scale with training duration.

**Domain B:** Prior H100 softcap tuning (runs 31-33, H22) — source primitive: Softcap=12 beat both 10 and 15 at ~500-1000 steps. At 2401 steps, the optimum may shift toward less restriction.

**Synthesis:** Test softcap=15 (the original value) at 2401 steps. The non-obvious prediction: what was "too loose" at 500 steps may be "just right" at 2401 steps, because longer training provides enough gradient signal to learn precise logit magnitudes that the tighter cap was restricting.

**Falsifiability:** "val_bpb will decrease because more training steps justify weaker regularization, and this would NOT appear if softcap=12's benefit is about logit dynamics rather than regularization strength (in which case it would be optimal regardless of step count)."

**Prediction:** Small improvement (0.001-0.003) or no change. If softcap=15 is worse, softcap's benefit is structural (dynamics) not regulatory (complexity control).

**Implementation sketch:** Change `softcap = 12` to `softcap = 15` in GPT.forward.

---

## Hypothesis 8 | Status: UNTESTED | Priority: 4 | [EXPLORATORY]

**Diagnosis:** With 25 GB VRAM headroom (55 GB used, 80 GB available) and 2401 steps already, the bottleneck may be model capacity rather than step count. The prior best depth was 10; with 7.5 min, DEPTH=12 at 384-dim (smaller model, more depth) might give more compositional power without excessive step reduction.

**Domain A:** MobileLLM depth-width scaling (2402.17764) — source primitive: For small models, depth is more valuable than width. Going deeper-thinner improves quality at constant param budget. The compositional power of each additional layer outweighs the reduced per-layer capacity.

**Domain B:** VE parameter overhead constraint — source primitive: VE inflates params proportional to depth × vocab_size × kv_dim. At 12×384: kv_dim = 384, VE layers = 6, VE params = 6 × 32K × 384 = 75.5M. Manageable but significant.

**Synthesis:** Test 12×384 (DEPTH=12, dim=384). With 7.5 min budget, expect ~3000+ steps (384-dim is faster than 512-dim). The additional depth provides more compositional layers, while higher step count provides more optimization. The non-obvious element: prior H100 runs found 10×512 beat 12×384 at 5 min, but at 7.5 min the step-count advantage of 384-dim may tip the balance.

**Falsifiability:** "val_bpb will decrease because deeper architecture provides more compositional power and the 7.5-min budget provides enough steps, and this would NOT appear if 512-dim width is needed for sufficient per-layer capacity (in which case depth alone can't compensate)."

**Prediction:** 12×384 should give ~3000-3500 steps. Expect val_bpb within ±0.005 of current best. If it's better, depth beats width at 7.5 min. If worse, 512-dim per-layer capacity is more important.

**Implementation sketch:** DEPTH=12, ASPECT_RATIO=30 (12×30=360, rounds to 384). Adjust window pattern to 9S+3L for 12 layers.
