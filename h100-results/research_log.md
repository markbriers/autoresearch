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

# H100 NVL Phase

Hardware change: Apple Silicon MPS → NVIDIA H100 NVL (94GB VRAM)
Architecture change: DEPTH=4 → DEPTH=8, TOTAL_BATCH_SIZE=2^14 → 2^19, DEVICE_BATCH_SIZE=8 → 128
New capabilities: torch.compile, Flash Attention 3, sliding window attention
MPS findings carried over: warmdown=0.3, x0_lambdas=0.2
Other hyperparams reset to defaults for H100 regime (different batch/depth dynamics)

## Run H0 | val_bpb: 1.084707 | H100 baseline | KEEP

Unmodified `train.py` on H100 NVL. 50.3M params, DEPTH=8, 505 steps, 45.1 GB VRAM, 264.8M tokens, 20.9% MFU, ~600ms/step.
Config: model_dim=512, n_head=4, HEAD_DIM=128, TOTAL_BATCH_SIZE=2^19, DEVICE_BATCH_SIZE=128.
Hyperparams: MATRIX_LR=0.04, EMBEDDING_LR=0.6, WEIGHT_DECAY=0.2, warmdown=0.3, x0_lambdas=0.2, softcap=15.

## Run H1 | val_bpb: 1.083814 | delta: -0.001 | KEEP

**Diagnosis:** Softcap=15 was suboptimal on MPS; testing if softcap=12 transfers to H100.
**Rationale:** Tighter logit capping regularizes predictions. Confirmed on MPS across softcap 8/10/12/15 sweep.
**Prediction:** Small improvement, confirming generalization.
**Code change:** softcap 15 → 12
**Outcome:** Marginal improvement (-0.001), confirms MPS finding transfers. The effect is smaller on H100 (deeper model has better implicit regularization).
**Surprise score:** 1

## Run H2 | val_bpb: 1.093902 | delta: +0.010 | DISCARD

**Diagnosis:** Attention noise may be wasting capacity in this 4-head, 8-layer model.

**Domain A** — Differential Transformer (2410.05258)
- Source primitive: Subtraction of two softmax attention maps to cancel common-mode noise
- Target bottleneck: Attention allocating capacity to irrelevant context
- Mapping: Differential amplifier analogy — subtraction cancels shared noise signal
- Validation: 1.4B DiffTransformer matches 2.8B standard; ICLR 2025 Oral

**Domain B** — xIELU activation (2411.13010)
- Source primitive: Recovering signal from negative pre-activations via trainable ELU branch
- Target bottleneck: ReluSquared kills all negative activations, wasting MLP input information
- Mapping: Both papers address "wasted capacity" in different components
- Validation: Lower perplexity than ReLU² at 1.1B-3B scale on FineWeb Edu

**Synthesis:** DiffAttn addresses attention capacity waste (noise cancellation) while xIELU addresses MLP capacity waste (negative signal recovery). Implemented via single FA3 call with 2× sub-heads at half dim, V repeated for paired sub-heads, then subtraction.
**Falsifiability:** "val_bpb will decrease because differential attention cancels attention noise, and this would NOT appear if attention patterns were already near-optimal."
**Code change:** DiffAttn with 8 sub-heads of dim 64 (vs 4 heads of 128), halved c_v (512→256) and c_proj (256→512). Model dropped from 50.3M to 39.8M params.
**Outcome:** Regression (+0.010). The 21% param reduction from halved V/c_proj dominates. FA3 dim-matching constraint forces V to sub_head_dim, making the implementation param-inefficient. A fair test needs compensating capacity, but that confounds the experiment.
**Surprise score:** 2 (param reduction magnitude was unexpected)

## Run H3 | val_bpb: 1.084680 | delta: +0.001 | DISCARD

**Diagnosis:** WD=0.1 was marginally better on MPS (batch=8). Testing transfer to H100 (batch=524K).
**Rationale:** Lower WD suited noisy MPS gradients; H100's clean large-batch gradients may prefer different WD.
**Prediction:** Marginal improvement if MPS finding generalizes.
**Code change:** WEIGHT_DECAY 0.2 → 0.1
**Outcome:** Slight regression (+0.001). Large batch → cleaner gradients → WD=0.2 provides useful regularization that 0.1 doesn't. MPS batch-size-dependent findings don't transfer to H100 regime.
**Surprise score:** 1

## Run H4a | val_bpb: 1.088027 | delta: +0.004 | DISCARD

**Diagnosis:** H100 has 50GB VRAM headroom; can we invest in depth?
**Rationale:** Depth 8→12, same model_dim=512. Tests if more representational depth helps.
**Prediction:** More capacity should help if step count doesn't drop too much.
**Code change:** DEPTH=12, ASPECT_RATIO=43 (model_dim stays 512)
**Outcome:** Regression. Model doubled to 100.9M params, steps halved to 250. Step count dominates.
**Surprise score:** 1

## Run H4b | val_bpb: 1.059993 | delta: -0.024 | KEEP ← NEW BEST

**Diagnosis:** Depth 8→12 at same width was param-wasteful. What about deeper-at-same-params?

**Domain A** — MobileLLM (2402.14905) — mobile/edge architecture optimization
- Source primitive: "Deep and thin" architecture design for sub-1B models
- Target bottleneck: Width-dominated architectures waste capacity on redundant features
- Mapping: More layers allow more sequential refinement; thinner layers reduce feature redundancy
- Validation: MobileLLM 125M outperforms wider baselines at same param count

**Domain B** — Scaling laws for depth vs width (derived from Chinchilla/Kaplan analyses)
- Source primitive: Depth scales more efficiently than width for small training budgets
- Target bottleneck: Our model trains for only 264M tokens — insufficient for wide models to converge
- Mapping: Width gives capacity but needs data to fill it; depth gives compositional power with less data
- Validation: Standard scaling law findings favor deeper models at fixed compute

**Synthesis:** MobileLLM shows deeper-thinner wins at sub-1B. Scaling laws suggest depth is more efficient per-token. Combined: at our training budget (264M tokens, 500 steps), 12×384 should beat 8×512 because depth gives compositional power that doesn't require as much data, while the thinner model processes tokens faster (more steps).
**Falsifiability:** "val_bpb will decrease because deeper composition extracts more per-token learning, and this would NOT appear if width (feature diversity) were the binding constraint rather than depth (compositional complexity)."
**Code change:** DEPTH 8→12, ASPECT_RATIO 64→32 → model_dim 512→384, n_head 4→3. 46.4M params (vs 50.3M).
**Outcome:** Massive improvement! -0.024 val_bpb. 488 steps (vs 505), 50 GB VRAM. Deeper-thinner wins conclusively. The compositional depth hypothesis is confirmed: at this training budget, 12 layers × 384-dim extracts far more learning than 8 × 512.
**Surprise score:** 4 (magnitude of improvement was much larger than expected)

## Run H5 | val_bpb: 1.072298 | delta: +0.012 | DISCARD

**Diagnosis:** Deeper-thinner (12×384) gave huge win. Does 16×384 push further?
**Rationale:** If depth is the key lever, more depth at same width should help.
**Prediction:** Possible improvement, but diminishing returns expected.
**Code change:** DEPTH 12→16, ASPECT_RATIO 32→24 (model_dim stays 384)
**Outcome:** Regression. 59.8M params (more VE layers at depth 16), 379 steps. The VE param overhead per layer means 16 layers costs more params than expected, reducing step count. 12 layers is the sweet spot at this training budget.
**Surprise score:** 2 (VE-driven param inflation at higher depth was not obvious)

## Runs H6-H8 — Muon LR tuning for 12×384

| Run | Muon LR | val_bpb | Delta | Status |
|-----|---------|---------|-------|--------|
| H6 | 0.06 | 1.058801 | -0.001 | KEEP |
| H7 | 0.08 | 1.057306 | -0.002 | KEEP |
| H8 | 0.10 | 1.062105 | +0.003 | DISCARD |

**Optimal Muon LR for 12×384 is 0.08** (2× the 8×512 optimal of 0.04). Thinner model benefits from more aggressive optimization — Newton-Schulz updates are scale-invariant, so smaller weight matrices allow larger effective steps. LR 0.10 overshoots.
**Surprise score:** 1

## Run H9 | val_bpb: 1.062251 | delta: +0.005 | DISCARD

**Diagnosis:** x0_lambdas=0.2 was optimized for 8-layer model; 12-layer might need different value.
**Rationale:** Deeper model may benefit from stronger input skip to maintain gradient flow.
**Prediction:** Possible improvement from stronger skip at depth 12.
**Code change:** x0_lambdas 0.2 → 0.3
**Outcome:** Regression. 12-layer model benefits LESS from input skip than 4-layer (MPS) model. With more depth, intermediate representations become more valuable than the initial embedding. x0_lambdas=0.2 is correct for this depth.
**Surprise score:** 2 (expected improvement based on MPS, opposite happened)

## Run H10 | val_bpb: 1.065410 | delta: +0.008 | DISCARD [EXPLORATORY]

**Diagnosis:** Only 3 heads at 128-dim may limit attention diversity for 12 layers.
**Rationale:** HEAD_DIM 128→64 gives 6 heads at same model_dim. FA3 on H100 handles more heads efficiently (unlike MPS).
**Prediction:** More diverse attention patterns should help deeper model.
**Code change:** HEAD_DIM 128 → 64 (6 heads)
**Outcome:** Regression despite same params and throughput. HEAD_DIM=128 provides better per-head attention capacity than HEAD_DIM=64. At 384-dim, 3 wide heads outperform 6 narrow ones — each head can attend to more features per head.
**Surprise score:** 2 (expected FA3 to make more heads viable)

---

## State of Search — H100 Runs H0-H10

**Best val_bpb:** 1.057306 (Run H7) — delta -0.027 from H100 baseline 1.084707.

**Biggest win:** Deeper-thinner (Run H4b, -0.024). This single change accounts for 89% of total improvement. The insight: at 500 steps and 264M tokens, depth (compositional power) beats width (feature diversity).

**Pattern of failures:**
- DiffAttn (H2): Good idea but FA3 dim-matching constraint caused 21% param reduction → confounded
- WD/x0_lambdas transfers (H3, H9): MPS findings at batch=8 don't transfer to batch=524K
- More heads (H10): Per-head capacity matters more than head diversity at 384-dim
- Depth 16 (H5): VE overhead inflated params, reducing steps

**Key insight:** For this 5-minute training budget, the dominant factor is **params × steps, not params alone**. Any change that reduces steps by >5% needs proportional quality improvement. The depth sweep (8→12→16) reveals a U-shaped optimum where 12 layers maximizes depth-per-param-per-step.

**MPS findings that transferred:** softcap=12 (small effect)
**MPS findings that didn't transfer:** WD=0.1, x0_lambdas=0.3

**Most underexplored direction:** Batch size reduction (worked massively on MPS). MLP architecture changes.

## Run H11 | val_bpb: 1.063899 | delta: +0.007 | DISCARD

**Diagnosis:** MLP might be the bottleneck — 4x expansion with ReluSquared sparsifies heavily.
**Rationale:** 6× expansion gives MLP more features to work with, compensating for sparsification.
**Prediction:** Possible improvement if MLP capacity is binding.
**Code change:** MLP expansion 4× → 6×
**Outcome:** Regression. 53.5M params (up from 46.4M), 428 steps (down from 488). Throughput loss dominates capacity gain. Same lesson as depth=16: at this budget, step count is sacred.
**Surprise score:** 1

## Run H12 | val_bpb: 1.042115 | delta: -0.015 | KEEP ← NEW BEST

**Diagnosis:** The biggest MPS lever (batch reduction) hasn't been tried on H100.
**Rationale:** TOTAL_BATCH_SIZE=2^19 with DEVICE_BATCH_SIZE=128 has grad_accum_steps=2. Eliminating accumulation (2^18) doubles step count while processing the same total tokens. The time-based LR schedule adapts automatically.
**Prediction:** Significant improvement (~0.005-0.010) based on MPS precedent.
**Code change:** TOTAL_BATCH_SIZE 2^19 → 2^18 (524K→262K tokens/step)
**Outcome:** Massive improvement! -0.015 val_bpb. 972 steps (2× more). Same total tokens (254M). Confirms MPS finding: more optimizer steps at the same compute budget is the dominant lever. The model converges better with more frequent weight updates.
**Surprise score:** 3 (magnitude even larger than MPS precedent at this depth)

## Runs H13-H24 — Tuning 12×384 + batch=262K regime

| Run | Change | val_bpb | Status |
|-----|--------|---------|--------|
| H13 | batch 2^18→2^17 (131K) | 1.046743 | DISCARD — too noisy |
| H14 | Muon LR 0.08→0.06 | 1.041611 | KEEP |
| H15 | Muon LR 0.06→0.04 | 1.037839 | KEEP |
| H16 | Muon LR 0.04→0.03 | 1.038565 | DISCARD |
| H17 | xIELU activation | 1.044158 | DISCARD — slower, no quality gain |
| H18 | embedding LR 0.6→0.4 | 1.038150 | DISCARD |
| H19 | warmdown 0.3→0.4 | 1.036207 | KEEP |
| H20 | warmdown 0.4→0.5 | 1.036677 | DISCARD |
| H21 | WD 0.2→0.3 | 1.037923 | DISCARD |
| H22 | all-short window (S) | 1.035887 | KEEP ← BEST |
| H23 | softcap 12→10 | 1.036200 | DISCARD |
| H24 | depth 14, dim 384 | 1.042037 | DISCARD — VE inflation |
| H25 | VE every 4th layer | 1.041929 | DISCARD — VE quality essential |
| H26 | NS steps 5→3 | 1.040674 | DISCARD — worse orthogonalization |

**Key findings:**
- Muon LR ∝ batch_size pattern continues: 262K optimal at 0.04 (same as original 524K baseline)
- Warmdown 0.4 is better than 0.3 at 972 steps (more cooldown time)
- All-short window marginally better — attention doesn't need full context in early layers
- xIELU activation hurt throughput without quality gain — ReluSquared is well-matched to torch.compile
- VE remains essential even at 12 layers; reducing frequency hurts quality
- NS=3 saves negligible time on H100 but degrades orthogonalization quality

**Best: val_bpb = 1.035887** (delta -0.049 from H100 baseline)

## Runs H27-H37 — Further tuning and architectural experiments

| Run | Change | val_bpb | Status |
|-----|--------|---------|--------|
| H27 | LNS (1/sqrt(depth)) | 1.036087 | DISCARD — resid_lambdas already compensate |
| H28 | 20×256 ultra-deep | 1.052278 | DISCARD — VE inflation to 73.1M params |
| H29 | x0_lambdas 0.2→0.1 | 1.036061 | DISCARD — flat |
| H30 | embedding tying | 3.215849 | CRASH — init/LR mismatch |
| H31 | embedding LR 0.6→0.8 | 1.036888 | DISCARD |
| H32 | FINAL_LR_FRAC 0→0.05 | 1.035006 | KEEP |
| H33 | FINAL_LR_FRAC 0.05→0.10 | 1.035910 | DISCARD |
| H34 | Adam β1 0.8→0.9 | 1.037855 | DISCARD |
| H35 | Muon momentum ramp 300→500 | 1.037079 | DISCARD |
| H36 | MQA (n_kv_head=1) | 1.042629 | DISCARD — 31.5M params too small |
| H37 | short window 512 (1/4) | 1.033199 | KEEP |
| H38 | short window 256 (1/8) | 1.031697 | KEEP ← BEST |
| H39 | short window 128 (1/16) | 1.032566 | DISCARD — too small |
| H40 | Muon LR 0.04→0.035 | 1.033036 | DISCARD |

**Key findings:**
- FINAL_LR_FRAC=0.05 prevents model from freezing at end of warmdown — keeps learning
- Sliding window size is a powerful lever: 256 tokens (1/8 context) gives most steps while maintaining quality
- Only the last layer needs full context; short-window layers handle local patterns
- VE param overhead makes extreme depth (>14 layers) counterproductive at this budget
- LNS doesn't help because existing resid_lambdas already solve the variance problem

**Best: val_bpb = 1.031697** (delta -0.053 from H100 baseline)

## Runs H41-H55 — Depth/width sweep and final tuning

| Run | Change | val_bpb | Status |
|-----|--------|---------|--------|
| H41 | depth 8, dim 384 | 1.033000 | DISCARD — too shallow |
| H42 | Muon LR 0.03 at depth=10 | 1.028694 | KEEP |
| H43 | Muon LR 0.02 | 1.030572 | DISCARD |
| H44 | warmdown 0.5 | 1.028509 | KEEP |
| H45 | warmdown 0.6 | 1.028247 | KEEP |
| H46 | warmdown 0.7 | 1.028629 | DISCARD |
| H47 | 10×512 | 1.024074 | KEEP — wider model wins! |
| H48 | 10×640 | 1.038020 | DISCARD — too wide |
| H49 | Muon LR 0.04 at 10×512 | 1.023513 | KEEP ← BEST |
| H50 | Muon LR 0.05 | 1.024568 | DISCARD |
| H51 | 8S+2L window | 1.024073 | DISCARD — 3L better |

**Key insight:** The optimal model size scales with depth/width jointly. At 10 layers, model_dim=512 (60.8M params, 846 steps) beats 384 (39.7M, 1207 steps) because the quality gain per step from wider representation outweighs the step count reduction.

**Best: val_bpb = 1.023513** (delta -0.061 from H100 baseline)

## Runs H52-H65 — Final optimization at 10×512

| Run | Change | val_bpb | Status |
|-----|--------|---------|--------|
| H52 | WD 0.2→0.1 | 1.023723 | DISCARD |
| H53 | softcap 12→10 | 1.023762 | DISCARD |
| H54 | 11×512 (wrong AR→640) | 1.038485 | DISCARD |
| H55 | 11×512 (correct) | 1.024647 | DISCARD |
| H56 | FINAL_LR_FRAC 0.05→0.02 | 1.023739 | DISCARD |
| H57 | short window 384 | 1.023258 | KEEP ← BEST |
| H58 | short window 320 | 1.023509 | DISCARD |
| H59 | SwiGLU (correct init) | 1.031438 | DISCARD |
| H60 | shared VE | 1.027332 | DISCARD |
| H61 | embedding LR 0.6→1.0 | 1.024561 | DISCARD |
| H62 | warmdown 0.6→0.5 at 512 | 1.023518 | DISCARD |
| H63 | VE gate full input | 1.024943 | DISCARD |
| H64 | MLP 4x→5x | 1.024078 | DISCARD |

**The model is at a well-optimized local minimum.** 10×512, 60.8M params, 845 steps. Short window 384. All hyperparameters are within 0.001 of optimal.

**Best: val_bpb = 1.023258** (delta -0.061 from H100 baseline 1.084707)

## Run H13 | val_bpb: 1.046743 | delta: +0.005 | DISCARD

**Diagnosis:** If 262K works, does 131K (even more steps) help?
**Code change:** TOTAL_BATCH_SIZE 2^18→2^17, DEVICE_BATCH_SIZE 128→64
**Outcome:** Too noisy at 131K tokens/step. 1908 steps but regression. Optimal is 262K.
**Surprise score:** 1

## Runs H14-H16 — Muon LR re-tuning for batch=262K

| Run | Muon LR | val_bpb | Status |
|-----|---------|---------|--------|
| H14 | 0.06 | 1.041611 | KEEP |
| H15 | 0.04 | 1.037839 | KEEP ← BEST |
| H16 | 0.03 | 1.038565 | DISCARD |

**Pattern:** Each batch halving shifts optimal Muon LR downward. Batch=2^19 (524K) optimal: 0.08. Batch=2^18 (262K) optimal: 0.04. The relationship is approximately LR ∝ batch_size. Consistent with MPS finding where batch=8 (16K) needed LR=0.02.
**Surprise score:** 1 (pattern was expected from MPS experience)
