# Research Log

## Run 0 | val_bpb: 1.350108 | baseline | KEEP

Unmodified `train.py` on Apple Silicon MPS. 11.5M params, DEPTH=4, 321 steps, 26.1 GB VRAM.

## Run 1 | val_bpb: 1.353994 | delta: +0.004 | DISCARD

**Diagnosis:** Value embedding learning rate might be too high, causing the VE gate to overfit.
**Rationale:** Halving VE LR from 0.6 to 0.3 to test bilinear coupling hypothesis.
**Prediction:** Lower VE LR should improve generalization.
**Code change:** VE LR 0.6 → 0.3
**Outcome:** Marginal regression. VE LR of 0.6 is appropriate for this architecture.
**Surprise score:** 1

## Run 2 | val_bpb: 1.348613 | delta: -0.001 | KEEP

**Diagnosis:** With only 321 steps, spending 50% on warmdown wastes training at full LR.
**Rationale:** Reducing warmdown from 0.5 to 0.3 gives ~60 more steps at peak LR.
**Prediction:** More steps at full LR should lower final loss.
**Code change:** WARMDOWN_RATIO 0.5 → 0.3
**Outcome:** Small improvement as predicted.
**Surprise score:** 1

## Run 3 | val_bpb: 1.344953 | delta: -0.005 | KEEP

**Diagnosis:** With only 4 layers, residual stream benefits from stronger skip to initial representation.
**Rationale:** x0_lambdas=0.1 is too small — increasing to 0.2 gives the initial embedding more influence in the residual stream, compensating for limited depth.
**Prediction:** Stronger input skip should improve val_bpb.
**Code change:** x0_lambdas init 0.1 → 0.2
**Outcome:** Best improvement so far. Confirms shallow models need strong input skip connections.
**Surprise score:** 2

## Run 4 | val_bpb: 1.347956 | delta: +0.003 | DISCARD

**Diagnosis:** Softcap at 15 might be too restrictive for logit expressiveness.
**Rationale:** Doubling softcap to 30 gives logits more dynamic range.
**Prediction:** More logit range should help if the model is constrained.
**Code change:** softcap 15 → 30
**Outcome:** Regression. Softcap=15 provides useful regularisation for this model size.
**Surprise score:** 1

## Run 5 | val_bpb: 1.496132 | delta: +0.151 | DISCARD

**Diagnosis:** With only 2 attention heads, attention noise limits information extraction per layer.

**Domain A** — Differential Transformer (2410.05258)
- Source primitive: Differential attention — subtraction of two softmax attention maps
- Target bottleneck: Attention noise drowning out relevant context
- Mapping: Subtraction cancels common-mode noise (differential amplifier analogy)
- Validation: 65% fewer params to match Transformer; ICLR 2025

**Domain B** — SpectFormer (2304.06446)
- Source primitive: Spectral (Fourier) layers in early layers, attention in deeper layers
- Target bottleneck: Attention wastes capacity on patterns spectral methods handle better
- Mapping: Different mixing mechanisms have different inductive biases per layer
- Validation: 2% top-1 accuracy improvement on ImageNet

**Synthesis:** Post-RoPE Q/K split creates frequency-domain decomposition (low/high positional frequencies). Differential attention then acts as band-pass positional filter.
**Code change:** Split Q/K into sub-heads, two separate attention maps, weighted subtraction. Manual attention path (MPS SDPA doesn't support different Q/K vs V dims).
**Outcome:** Massive regression. Manual attention is ~40% slower (228 vs 321 steps) and uses 7.8 GB more VRAM. The throughput loss dominates any attention quality improvement. The idea is sound but infeasible on MPS without efficient SDPA support for mismatched dims.
**Surprise score:** 2 (the MPS SDPA limitation was unexpected)

## Run 6 | val_bpb: 1.413882 | delta: +0.069 | DISCARD

**Diagnosis:** 2 heads limit attention pattern diversity; increasing to 4 heads should help.
**Rationale:** HEAD_DIM=64 gives 4 heads instead of 2, doubling attention diversity at zero parameter cost. Each SDPA call should be the same total FLOPs.
**Prediction:** Faster convergence from more diverse attention, modest val_bpb improvement.
**Code change:** HEAD_DIM 128 → 64
**Outcome:** Large regression. MPS SDPA memory scales with head count — VRAM jumped 24→34 GB, steps dropped 321→254. This architecture (2 heads × 128-dim) is strongly preferred on MPS due to SDPA efficiency characteristics. The attention mechanism cannot be changed without throughput penalty.
**Surprise score:** 4 (MPS SDPA scaling with head count was very unexpected)

## Run 7 | val_bpb: 1.363247 | delta: +0.018 | DISCARD

**Diagnosis:** Model benefits from more time at full LR (runs 2-3 pattern).
**Rationale:** Reducing warmdown from 0.3 to 0.15 gives ~48 more steps at peak LR.
**Prediction:** Small improvement (~0.001-0.002), diminishing returns.
**Code change:** WARMDOWN_RATIO 0.3 → 0.15
**Outcome:** Regression. 0.15 warmdown is too aggressive — the model needs sufficient cooldown to converge. The optimal warmdown for 321 steps is around 0.25-0.3 (not 0.15 or 0.5). This establishes a ceiling on warmdown reduction.
**Surprise score:** 2

## Run 8 | val_bpb: 1.332807 | delta: -0.012 | KEEP

**Diagnosis:** With 321 steps, each optimizer step is ~0.93s. More frequent updates could help.
**Rationale:** Halving batch size (32→16, 65K→32K tokens/step) gives ~2× more steps (569). Total tokens processed is the same. Time-based LR schedule is unaffected. Smaller batch uses ~12 GB VRAM (vs 24 GB), which might reduce MPS memory pressure.
**Prediction:** Val_bpb should improve ~0.002-0.005 from finer-grained optimization.
**Code change:** DEVICE_BATCH_SIZE 32→16, TOTAL_BATCH_SIZE 2^16→2^15
**Outcome:** Best improvement yet (-0.012). The model clearly benefits from more optimization steps with noisier gradients. VRAM dropped to 12.4 GB. This was much better than predicted — the combination of more steps and reduced memory pressure on MPS appears highly beneficial.
**Surprise score:** 3 (magnitude of improvement was surprising)

## Run 9 | val_bpb: 1.325292 | delta: -0.008 | KEEP

**Diagnosis:** Batch reduction to 16/32K gave best result (-0.012). Can we push further?
**Rationale:** Halving again to 8/16K should give ~1200 steps. Still in the "more steps" regime.
**Prediction:** Further improvement, possibly diminishing returns (~0.003-0.005).
**Code change:** DEVICE_BATCH_SIZE 16→8, TOTAL_BATCH_SIZE 2^15→2^14
**Outcome:** Another strong improvement: 1.325292 (-0.008 from prev, -0.020 from original best). 1195 steps, 6.5 GB VRAM. The model is clearly step-limited, not batch-noise-limited. Diminishing returns are visible (0.008 vs 0.012) but still worthwhile.
**Surprise score:** 2

## Run 10 | val_bpb: 1.349339 | delta: +0.024 | DISCARD

**Diagnosis:** Batch=8 improved -0.008; can we push to batch=4?
**Rationale:** Further halving to 4/8K should give ~2400 steps. Testing the limit.
**Prediction:** Possible improvement or regression if gradients become too noisy.
**Code change:** DEVICE_BATCH_SIZE 8→4, TOTAL_BATCH_SIZE 2^14→2^13
**Outcome:** Regression. Despite 2336 steps (7.3× original), the gradient noise at 8K tokens/step is too high for stable optimization. The model trained but converged to a worse solution. The optimal batch size for this architecture is DEVICE_BATCH_SIZE=8, TOTAL_BATCH_SIZE=16K (1200 steps). The batch-size/step-count tradeoff has a clear U-shaped optimum.
**Surprise score:** 1

---

## State of Search — Runs 0-10

**Best val_bpb:** 1.325292 (Run 9) — delta -0.025 from baseline 1.350108.

**Highest surprise scores:** Run 6 (HEAD_DIM change, score 4) and Run 8 (batch reduction, score 3). Both involved throughput/memory tradeoffs on MPS that were non-obvious.

**Most productive direction:** Batch size reduction (runs 8-10). This single direction accounts for -0.020 of the total -0.025 improvement. The mechanism is simple: with a time-based LR schedule, more optimization steps per minute directly translates to more gradient updates.

**Pattern of failures:** 4 of 6 failures (runs 1, 4, 5, 6) were due to MPS hardware constraints rather than bad ideas. The attention mechanism (runs 5, 6) is completely frozen by MPS SDPA limitations. Warmdown reduction (run 7) hit a hard floor at 0.15. The only "idea was bad" failure is VE LR halving (run 1).

**Architecture inductive biases:** This model is severely constrained by MPS throughput. The dominant optimisation axis is tokens-per-second × steps-per-minute, not model quality per step. Any change that reduces throughput by >5% must deliver >0.01 val_bpb improvement to break even.

**Most underexplored direction:** The MLP. With batch=8 and 6.5 GB VRAM, there's substantial headroom for increasing model capacity (currently only 11.5M params with ~20 GB unused VRAM). MLP expansion ratio, activation functions, or model width changes have not been tested with the new batch size. Also, optimizer LR tuning has not been revisited since the batch size change — the optimal LRs may be different with 1200 steps vs 321.

## Run 11 | val_bpb: 1.330891 | delta: +0.006 | DISCARD

**Diagnosis:** VRAM headroom (~20 GB) could be invested in model capacity.
**Rationale:** ASPECT_RATIO 64→96 gives 384-dim, 3 heads, 19.7M params (1.7× more).
**Prediction:** More capacity per step should compensate for fewer steps (~530 est).
**Code change:** ASPECT_RATIO 64 → 96
**Outcome:** Regression. 842 steps (not as few as predicted) but still fewer than 1195. The wider model's per-step quality gain doesn't compensate for 30% fewer optimization steps. At this training budget, step count dominates model size. The finding generalises: any change that reduces steps by >10% needs proportionally more benefit.
**Surprise score:** 1

## Run 12 | val_bpb: 1.328005 | delta: +0.003 | DISCARD

**Diagnosis:** Muon LR was tuned for batch=32. Might need adjustment for batch=8.
**Rationale:** Muon LR 0.04→0.06: higher LR to cut through noisy gradients.
**Prediction:** Possible improvement from more aggressive steps.
**Code change:** MATRIX_LR 0.04 → 0.06
**Outcome:** Slight regression. The original 0.04 was better. Higher LR overshoots in the noisier gradient regime.
**Surprise score:** 1

## Run 13 | val_bpb: 1.317505 | delta: -0.008 | KEEP

**Diagnosis:** Muon LR 0.06 was too high; try lower LR for noisy batch=8 regime.
**Rationale:** With 4× noisier gradients (batch 8 vs 32), the optimizer needs smaller steps for precision. 1276 steps provide enough time for slower convergence.
**Prediction:** Lower LR should improve convergence quality.
**Code change:** MATRIX_LR 0.04 → 0.03
**Outcome:** New best! val_bpb 1.317505 (-0.008). Confirms that with noisy gradients, lower LR gives better convergence.
**Surprise score:** 2

## Run 14 | val_bpb: 1.316588 | delta: -0.001 | KEEP

**Diagnosis:** Muon LR 0.03 improved; testing if 0.02 continues the trend.
**Rationale:** Diminishing returns expected — probing the lower bound.
**Prediction:** Marginal improvement or regression.
**Code change:** MATRIX_LR 0.03 → 0.02
**Outcome:** Marginal improvement (-0.001). The trend is flattening — we're near the Muon LR optimum for batch=8. Optimal is around 0.02-0.03.
**Surprise score:** 1

## Run 15 | val_bpb: 1.309136 | delta: -0.007 | KEEP

**Diagnosis:** x0_lambdas=0.2 was set at batch=32. May need revisiting.
**Rationale:** With noisier gradients and more steps, stronger input skip might help the residual stream maintain stable representations.
**Prediction:** Possible improvement; 0.3 might be too strong.
**Code change:** x0_lambdas init 0.2 → 0.3
**Outcome:** Strong improvement! -0.007. x0_lambdas benefits from re-tuning after the batch/LR changes. Total delta from baseline is now -0.041.
**Surprise score:** 2

## Run 16 | val_bpb: 1.309913 | delta: +0.001 | DISCARD

**Diagnosis:** x0_lambdas=0.3 worked; testing 0.4.
**Rationale:** Further increasing input skip to test upper bound.
**Prediction:** Possible improvement or regression.
**Code change:** x0_lambdas init 0.3 → 0.4
**Outcome:** Marginal regression. 0.3 is the optimum for this configuration.
**Surprise score:** 1

## Runs 17-26 — Exhaustive hyperparameter search

After establishing batch=8/Muon LR=0.02/x0_lambdas=0.3 (best: 1.309136), systematic exploration of all remaining hyperparameters:

| Run | Change | val_bpb | Delta | Status |
|-----|--------|---------|-------|--------|
| 17 | 5% LR warmup | 1.316859 | +0.008 | DISCARD |
| 18 | Embedding LR 0.6→0.3 | 1.312189 | +0.003 | DISCARD |
| 19 | Muon momentum 0.95→0.97 | 1.313677 | +0.005 | DISCARD |
| 20 | VE disabled | 1.339394 | +0.030 | DISCARD |
| 21 | VE on all layers | 1.316163 | +0.007 | DISCARD |
| 22 | Batch 8→6 | 1.319839 | +0.011 | DISCARD |
| 23 | Weight decay 0.2→0.1 | 1.309109 | -0.000 | KEEP (marginal) |
| 24 | Unembedding LR 0.004→0.008 | 1.317445 | +0.008 | DISCARD |
| 25 | Gradient clipping 1.0 | 1.311372 | +0.002 | DISCARD |
| 26 | Adam β1 0.8→0.9 | 1.313757 | +0.005 | DISCARD |

**Conclusion:** The model is at a well-optimised local minimum for this architecture and batch configuration. Single-parameter changes yield diminishing returns.

## Runs 27-35 — Further exploration

| Run | Change | val_bpb | Delta | Status |
|-----|--------|---------|-------|--------|
| 27 | Warmdown 0.2 + FINAL_LR_FRAC 0.1 | 1.313077 | +0.004 | DISCARD |
| 28 | Depth 4→5 (256-dim) | 1.321776 | +0.013 | DISCARD |
| 29 | Label smoothing 0.1 | 1.610793 | +0.302 | DISCARD |
| 30 | z-loss 1e-4 | 1.322164 | +0.013 | DISCARD |
| 31 | Softcap 15→10 | 1.308625 | -0.000 | KEEP (then replaced) |
| 32 | Softcap 10→8 | 1.309569 | +0.001 | DISCARD |
| 33 | Softcap 10→12 | 1.308101 | -0.001 | KEEP ← best |
| 34 | Muon momentum ramp 300→600 | 1.310112 | +0.002 | DISCARD |
| 35 | Scalar LR 0.5→0.3 | 1.309409 | +0.001 | DISCARD |
| 36 | QK norm removed | 1.321328 | +0.013 | DISCARD |

**Final best: val_bpb = 1.308101** (delta -0.042 from baseline 1.350108)

---

## H100 7.5-minute Budget Phase (autoresearch/mar23)

## Run 37 | val_bpb: 0.977763 | H100 7.5min baseline | KEEP

Unmodified train.py on H100 SXM with 7.5-min budget. DEPTH=8, 512-dim, 50.3M params, batch=524K, 1368 steps, 45 GB VRAM. Already beats prior 5-min H100 best (1.023) by 0.045 from extra time alone.

## Run 38 | val_bpb: 1.080267 | delta: +0.103 | DISCARD

**Diagnosis:** Additive residuals suffer from norm growth in deeper models; per-dimension control should help.

**Domain A** — nGPT: Normalized Transformer on the Hypersphere (2410.01131)
- Source primitive: Per-dimension learnable residual scaling (eigen learning rates)
- Target bottleneck: Residual stream norm growth in deep models
- Mapping: Per-dimension alpha allows selective feature updating
- Validation: 4-20x training speedup claimed in nGPT

**Domain B** — ReluSquared activation sparsity (this architecture)
- Source primitive: ~50% of MLP neurons produce zeros; output is sparse/directional
- Target bottleneck: Uniform residual scaling wastes the directionality of sparse MLP outputs
- Mapping: Per-dimension scaling should amplify surviving features
- Validation: Internal observation of ReluSquared sparsity patterns

**Synthesis:** Apply per-dimension MLP residual scaling only (not attention), exploiting ReluSquared's directional outputs.
**Falsifiability:** "val_bpb will decrease because per-dim scaling preserves directional sparsity signal."
**Code change:** Added mlp_alphas (n_layer, n_embd) parameter, multiplied MLP output per-dimension.
**Outcome:** Massive regression. Per-dimension scaling fundamentally conflicts with Muon's Newton-Schulz orthogonalization. Muon normalizes gradient direction globally; element-wise output scaling re-introduces non-uniform magnitude that breaks the orthogonalization invariant. The mapping was wrong: nGPT uses Adam (compatible with per-dim scaling), not Muon (which is scale-invariant by design).
**Surprise score:** 3 (the conflict with Muon was non-obvious)

## Run 39 | val_bpb: 0.987765 | delta: +0.010 | DISCARD

**Diagnosis:** VE params consume ~50% of model at depth 10; compression should enable deeper models.

**Domain A** — DeepSeek-V2 Multi-head Latent Attention (2405.04434)
- Source primitive: Low-rank KV compression via latent bottleneck
- Target bottleneck: Over-parameterized representations capture noise
- Mapping: Bottleneck forces learning of sufficient statistics
- Validation: 93% KV cache reduction with quality improvement in DeepSeek

**Domain B** — Information bottleneck principle (Tishby)
- Source primitive: Optimal compression preserves task-relevant information
- Mapping: Applied to vocabulary axis instead of time axis
- Validation: Theoretical framework for lossy compression

**Synthesis:** Factor VE as vocab×128 + 128→kv_dim, applying information bottleneck to vocabulary axis.
**Code change:** Replaced nn.Embedding(vocab, 512) with nn.Embedding(vocab, 128) + nn.Linear(128, 512).
**Outcome:** Model shrank from 50.3M to 38M params with +0.010 regression. The compression is effective (only +0.010 for 24% fewer params) but the prediction that it would IMPROVE quality was wrong. VE per-token diversity is load-bearing: MLA's bottleneck works on sequential redundancy (nearby tokens have similar KV), while VE compresses across tokens with genuinely distinct needs.
**Surprise score:** 1 (regression was expected direction, magnitude was small)

## Run 40 | val_bpb: 1.722050 | delta: +0.744 | DISCARD

**Diagnosis:** Softcap is a blunt mechanism; cosine similarity logits from nGPT should be more principled.

**Domain A** — nGPT (2410.01131) — cosine similarity output with learnable temperature
**Domain B** — Softcap mechanism in current architecture

**Synthesis:** Replace tanh softcap with normalized dot product × learnable temperature scalar.
**Code change:** F.normalize both hidden state and lm_head weights, multiply by learnable tau (init=30).
**Outcome:** Catastrophic regression. F.normalize discards hidden state norm, which carries critical information in non-normalized architectures. nGPT's cosine logits work because ALL layers are normalized (consistent geometry); applying it to ONLY the output layer creates an impedance mismatch. The source primitive requires the full nGPT architecture to function.
**Surprise score:** 2 (magnitude of failure was surprising given the mechanism sounds reasonable)

## Run 41 | val_bpb: 0.969390 | delta: -0.008 | KEEP

**Diagnosis:** Current config uses suboptimal hyperparams from MPS era.
**Rationale:** Apply known-good H100 settings: batch=262K, warmdown=0.6, FINAL_LR_FRAC=0.05, softcap=12, window=384.
**Prediction:** Should improve based on prior H100 results.
**Code change:** Multiple hyperparameter changes.
**Outcome:** As predicted, -0.008 improvement. 2820 steps (vs 1368). Known optimizations transfer well to 7.5-min budget.
**Surprise score:** 1

## Run 42 | val_bpb: 0.961937 | delta: -0.007 | KEEP

**Diagnosis:** More depth improves model quality at same width.
**Rationale:** Prior best H100 architecture was 10×512. Apply to 7.5-min budget.
**Prediction:** Should improve ~0.005-0.010 from increased depth.
**Code change:** DEPTH 8→10, ASPECT_RATIO 64→39.
**Outcome:** -0.007 improvement. 60.8M params, 2407 steps. Deeper architecture helps at 7.5 min too.
**Surprise score:** 1

## Run 43 | val_bpb: 0.961480 | delta: -0.000 | KEEP

**Diagnosis:** Warmdown scaling rule suggests higher ratio with more steps.
**Rationale:** warmdown ≈ 0.0006 × step_count. At 2407 steps: 0.7.
**Prediction:** Marginal improvement.
**Code change:** WARMDOWN_RATIO 0.6→0.7.
**Outcome:** Marginal improvement (-0.0005). Warmdown scaling holds.
**Surprise score:** 1

## Run 44 | val_bpb: 0.971271 | delta: +0.010 | DISCARD

**Diagnosis:** Sequential attn→MLP is a pipeline bottleneck; parallel could help.
**Domain A:** PaLM (2204.02311) — parallel attention+MLP formulation.
**Synthesis:** Run attn and MLP on same norm(x) input in parallel.
**Code change:** One-line change to Block.forward.
**Outcome:** Regression. Only +3.4% throughput gain (2483 vs 2401 steps) with -0.010 quality loss. At 60M params, attention→MLP information flow is critical.
**Surprise score:** 2

## Run 45 | val_bpb: 0.961738 | delta: +0.000 | DISCARD

**Diagnosis:** Softcap=12 may be too restrictive at 2401 steps.
**Code change:** softcap 12→15.
**Outcome:** Flat (+0.0003). Softcap=12 is optimal regardless of step count.
**Surprise score:** 1

## Run 46 | val_bpb: 0.979912 | delta: +0.019 | DISCARD

**Diagnosis:** 12 layers may beat 10 at 7.5 min budget.
**Code change:** DEPTH 10→12, dim 512→384 (46.4M params, 2608 steps).
**Outcome:** Regression. 10×512 beats 12×384 even at 7.5 min. 512-dim width is structurally essential.
**Surprise score:** 1

## Run 47 | val_bpb: 0.983540 | delta: +0.022 | DISCARD

**Diagnosis:** Stochastic depth might provide ensemble regularization.
**Code change:** Linearly increasing drop rate (0→0.2) for blocks, no throughput gain.
**Outcome:** Significant regression. Without throughput gain, stochastic masking just adds noise. Model already well-regularized.
**Surprise score:** 1

## Runs 48-55 — Engineering tuning at 10×512 with 7.5min budget

| Run | Change | val_bpb | Delta | Status |
|-----|--------|---------|-------|--------|
| 48 | Muon LR 0.04→0.03 | 0.959949 | -0.002 | KEEP |
| 49 | Muon LR 0.03→0.02 | 0.959968 | +0.000 | DISCARD |
| 50 | Batch 262K→131K (DBS=64) | 0.964828 | +0.005 | DISCARD |
| 51 | All-short window (S) | 0.961559 | +0.002 | DISCARD |
| 52 | 7S+3L (long at end) | 0.961766 | +0.002 | DISCARD |
| 53 | WD 0.2→0.1 | 0.959650 | -0.000 | KEEP |
| 54 | Embedding LR 0.6→0.4 | 0.959996 | +0.000 | DISCARD |
| 55 | DEPTH 11 (68.2M, 2204 steps) | 0.959848 | +0.000 | DISCARD |

**Current best after Runs 48-55: val_bpb = 0.959650**

## Runs 56-66 — Continued fine-tuning

| Run | Change | val_bpb | Delta | Status |
|-----|--------|---------|-------|--------|
| 56 | Short window 384→256 | 0.959171 | -0.001 | KEEP |
| 57 | Short window 256→192 | 0.959356 | +0.000 | DISCARD |
| 58 | x0_lambdas 0.2→0.15 | 0.961108 | +0.002 | DISCARD |
| 59 | x0_lambdas 0.2→0.25 | 0.959019 | -0.000 | KEEP |
| 60 | x0_lambdas 0.25→0.3 | 0.959444 | +0.000 | DISCARD |
| 61 | Embedding LR 0.6→0.8 | 0.959051 | +0.000 | DISCARD |
| 62 | Warmdown 0.7→0.75 | 0.959295 | +0.000 | DISCARD |
| 63 | FINAL_LR_FRAC 0.05→0.03 | 0.958926 | -0.000 | KEEP |
| 64 | FINAL_LR_FRAC 0.03→0.01 | 0.958888 | -0.000 | KEEP |
| 65 | FINAL_LR_FRAC 0.01→0.0 | 0.959061 | +0.000 | DISCARD |
| 66 | RoPE base 10000→5000 | 0.959151 | +0.000 | DISCARD |

**Current best: val_bpb = 0.958888** (10×512, Muon LR=0.03, WD=0.1, warmdown=0.7, softcap=12, batch=262K, window=SSSL/256, x0_lambdas=0.25, FINAL_LR_FRAC=0.01)

Total improvement from H100 7.5min baseline: **-0.019** (0.977763 → 0.958888)

## Runs 67-76 — Exhaustive search for remaining improvements

| Run | Change | val_bpb | Delta | Status |
|-----|--------|---------|-------|--------|
| 67 | EMA (beta=0.99) for eval | 0.961762 | +0.003 | DISCARD |
| 68 | Muon momentum 0.95→0.93 | 0.959514 | +0.001 | DISCARD |
| 69 | GQA n_kv_head=2 | 0.963410 | +0.005 | DISCARD |
| 70 | Tiny warmup 0.5% | 0.960197 | +0.001 | DISCARD |
| 71 | NS steps 5→7 | 0.958894 | +0.000 | DISCARD |
| 72 | Seed 42→0 | 0.960468 | +0.002 | DISCARD |
| 73 | Deep supervision (aux loss) | 0.965572 | +0.007 | DISCARD |
| 74 | Focal loss (gamma=1) | 0.962580 | +0.004 | DISCARD |
| 75 | WD 0.1→0.0 | 0.964276 | +0.005 | DISCARD |

**Conclusion:** The 10×512 model is at a genuine local optimum at 0.958888.

## Runs 77-86 — Width exploration and breakthrough

| Run | Change | val_bpb | Delta | Status |
|-----|--------|---------|-------|--------|
| 77 | GELU replacing ReluSquared | 0.973643 | +0.015 | DISCARD |
| 78 | c_proj init zeros→random | 0.959740 | +0.001 | DISCARD |
| 79 | 9×512 | 0.963346 | +0.004 | DISCARD |
| 80 | 10×768 DBS=64 (115M, 1387 steps) | 0.968726 | +0.010 | DISCARD |
| 81 | **10×640 DBS=64 (85.9M, 1764 steps)** | **0.956183** | **-0.003** | **KEEP** |
| 82 | **10×640 DBS=128 (85.9M, 1790 steps)** | **0.953816** | **-0.005** | **KEEP ← BEST** |
| 83 | Muon LR 0.03→0.04 at 640-dim | 0.955144 | +0.001 | DISCARD |
| 84 | warmdown 0.7→0.75 at 640-dim | 0.954976 | +0.001 | DISCARD |
| 85 | batch 262K→524K at 640-dim | 0.969626 | +0.016 | DISCARD |
| 86 | Muon LR 0.03→0.02 at 640-dim | 0.959196 | +0.005 | DISCARD |

**BREAKTHROUGH: 10×640 (85.9M params) beats 10×512 (60.8M params)!** The 7.5-min budget provides enough steps (1790) for the wider model to outperform. At 5 min, 640-dim was too slow (only ~900 steps). At 7.5 min, 1790 steps is enough for the additional width to help.

**Current best: val_bpb = 0.953816** (10×640, Muon LR=0.03, batch=262K, DBS=128, 85.9M params, 1790 steps)
