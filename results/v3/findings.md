# Findings

## Confirmed Mechanisms
Things that work, with mechanistic explanations for *why* they work on this architecture.

- **Stronger input skip connections (x0_lambdas=0.2) help shallow models.** With only 4 layers, each layer's contribution is relatively large, so the initial representation needs more weight in the residual stream to prevent early layers from dominating. (Run 3, -0.005 val_bpb)

- **Shorter warmdown ratio (0.3 vs 0.5) is better with few steps.** With only 321 steps, 50% warmdown means the model spends too long at reduced LR. (Run 2, -0.001 val_bpb)

- **Optimal batch size is DEVICE_BATCH_SIZE=8, TOTAL_BATCH_SIZE=16K (~1200 steps).** The batch-size/step-count tradeoff has a U-shaped optimum. Going from 32→16→8 each improved val_bpb (-0.012, -0.008), but batch=4 (8K tokens/step) regressed badly (+0.024) due to excessive gradient noise. The sweet spot gives ~1200 steps with 6.5 GB VRAM. (Runs 8-10)

## Dead Ends
Approaches that failed, with explanations for *why* they failed. The failure mode matters
more than the fact of failure — it reveals what this architecture cannot absorb.

- **nGPT-style modifications are incompatible with Muon optimizer.** Per-dimension residual scaling (+0.103), cosine similarity logits (+0.744) — both failed catastrophically. Muon's Newton-Schulz orthogonalization makes it scale-invariant; adding per-dimension scaling or removing magnitude information breaks the optimization. nGPT uses Adam, which is compatible with these mechanisms. Any future per-dimension scaling idea must account for Muon's orthogonal update structure. (Runs 38, 40)

- **VE bottleneck compresses well but doesn't improve quality.** Low-rank VE (rank=128, 4× compression) produced only +0.010 regression with 24% fewer params. VE per-token diversity is load-bearing — the bottleneck works for sequential KV (MLA) but not for vocabulary-level embeddings where each token genuinely needs distinct values. (Run 39)

- **Manual attention on MPS is infeasible.** Any technique requiring materialised T×T attention matrices (e.g., differential attention with mismatched Q/K vs V dims) adds ~40% overhead and ~8 GB VRAM. In a 5-minute time-budgeted run, the throughput loss (321→228 steps) dominates any attention quality improvement. All future attention modifications must work within SDPA's constraints: Q, K, V must have matching last dimension on MPS. (Run 5)

- **MPS SDPA limitation: Q/K and V must have the same last dimension.** Passing V with dim 128 when Q/K have dim 64 causes SDPA to silently output dim 64 (matching Q) instead of dim 128 (matching V). This is a PyTorch MPS backend bug/limitation. (Run 5)

- **Label smoothing is catastrophic for LM pretraining BPB.** Redistributing probability mass from the correct token to alternatives directly worsens BPB, which measures exact prediction quality. Label smoothing is for classification robustness, not LM pretraining. (Run 29, +0.301 val_bpb)

- **MPS SDPA memory scales with head count.** Doubling heads from 2 to 4 (same total Q/K/V size) increased VRAM from 24→34 GB and slowed training from 321→254 steps. MPS SDPA likely materialises per-head attention matrices rather than using memory-efficient attention. This means HEAD_DIM=128 with 2 heads is strongly preferred over HEAD_DIM=64 with 4 heads on this hardware. Any architectural change that increases effective head count will pay a heavy throughput/memory tax. (Run 6)

## Architecture Inductive Biases
What this specific architecture (shallow depth, ReluSquared, Muon, value embeddings,
residual lambdas) is and is not good at. These emerge from patterns across multiple runs.

- **Attention mechanism is frozen by MPS constraints.** HEAD_DIM=128 with 2 heads is the only viable configuration. More heads = more memory + slower. Manual attention = much slower. All future improvements must work *within* the existing attention mechanism (same 2 heads, same 128-dim Q/K/V, same SDPA). Focus on MLP, embeddings, optimizer, and training dynamics instead.

- **Value embeddings are essential.** Removing VE (4.2M params, 36% of model) caused a +0.030 regression despite freeing compute. VE provides a "default value" for attention that is disproportionately valuable for this shallow architecture — likely because it gives each layer a strong inductive prior without requiring the attention to learn it from scratch. (Run 20)

- **Softcap=12 is optimal for batch=8 regime.** The original softcap=15 was too loose, softcap=10 was slightly too tight, softcap=8 was too tight. Softcap=12 provides the right amount of logit regularisation with the noisy batch=8 gradients. (Runs 30-32)

- **Hyperparams are near-optimal at the current configuration.** Systematic tuning across 30+ runs found: batch=8 (DEVICE_BATCH_SIZE=8, TOTAL_BATCH_SIZE=16K), Muon LR 0.02, x0_lambdas 0.3, softcap 12, WD 0.1, warmdown 0.3, warmup 0, embedding LR 0.6, Muon momentum 0.95 (300-step ramp), Adam β1=0.8, scalar LR 0.5, unembedding LR 0.004. The model is at a well-optimised local minimum for this architecture.

## Cross-Domain Transfer Patterns
Which domain pairings produced genuine insight? What made the transfer work?
This section tracks the meta-question: what kinds of analogies are productive for
this class of problem?

## Cross-Domain Transfer Patterns
Which domain pairings produced genuine insight? What made the transfer work?

- **Optimization theory → batch size.** The most productive insight came not from ML papers but from basic optimization theory: the critical batch size for small models is much lower than the default. The NorMuon paper (optimization) led to understanding that the existing Muon code already implements per-neuron normalization, redirecting attention to batch size as the binding constraint. Cross-domain transfer worked here because the "more steps" insight comes from stochastic optimization theory, not from ML architecture papers.

## Open Questions
Specific, testable hypotheses that you have not yet investigated. Each should be
phrased as a question with a proposed experiment. Remove entries as they are resolved
(move the answer to the appropriate section above).

- **Does increasing model width help now that VRAM is 6.5 GB?** With ~20 GB headroom, ASPECT_RATIO could increase from 64 to 80+ (model_dim 256→384). Test: increase ASPECT_RATIO and measure throughput impact vs val_bpb gain.

- **Should Muon LR change with smaller batches?** Noisier gradients may require different LR. Test: try MATRIX_LR=0.06 and 0.03 at the new batch size.

- **Would x0_lambdas=0.3 help more than 0.2?** The initial skip connection improvement was with batch=32. The optimal might be different at batch=8 with 1200 steps.

- ~~**Does MLP expansion ratio matter?**~~ **RESOLVED: 6× MLP is destructive.** The extra params (85.9M→102.2M) couldn't compensate for 17% fewer steps (DBS=64 needed for VRAM). At this time budget, throughput > capacity. (Run 2)

- **Cosine warmdown IS implicit iterate averaging.** EMA(0.995) over the warmdown phase produces worse results than the final snapshot, because the cosine schedule already progressively reduces the step size (achieving the Polyak-Ruppert averaging effect implicitly). (Run 3)

- **Auxiliary prediction objectives are catastrophic at small scale.** 2-token prediction with 0.5× weight caused +0.794 regression (1.748 vs 0.954). At 85.9M params, the model has no spare capacity for a second objective. Multi-token prediction benefits emerge only at 13B+ scale (per Meta paper). (Run 4)

- **Any change requiring DBS reduction from 128→64 faces a ~12% step count penalty** due to grad_accum overhead. This penalty alone can negate moderate improvements.

## H100 7.5-minute Budget Findings

- **7.5-min budget gives massive improvement over 5-min.** Baseline improved from 1.085 to 0.978 at same config — 50% more time gives -0.107 val_bpb. The extra time enables 2.7× more steps.

- **Architecture modifications are uniformly destructive with Muon.** 7 novel modifications tested (per-dim MLP scaling, low-rank VE, cosine logits, parallel blocks, stochastic depth, deeper-thinner 12×384, depth=11) — all regressed. The Muon+RMSNorm+ReluSquared+softcap combination resists structural changes.

- **Muon makes per-dimension scaling mechanisms incompatible.** nGPT-style modifications (+0.103, +0.744 regressions) fundamentally conflict with Newton-Schulz orthogonalization. Any future idea must preserve Muon's scale-invariance.

- **Optimal H100 7.5min config:** 10×512, Muon LR=0.03, WD=0.1, warmdown=0.7, softcap=12, batch=262K, SSSL/256 window, x0_lambdas=0.25, FINAL_LR_FRAC=0.01. val_bpb=0.958888.

- **GQA is destructive at 60M params.** n_kv_head=2 (half KV heads) gave +0.005 regression. KV capacity per head is load-bearing at this model size. (Run 69)

- **Focal loss hurts BPB.** Down-weighting easy tokens reduces the model's ability to predict them, which contributes to BPB. The standard CE loss IS the optimal training objective for BPB. (Run 74)

- **Deep supervision is compute-wasteful.** Auxiliary prediction head at the middle layer adds FLOPs without quality benefit. The gradient signal from the final loss is sufficient for a 10-layer model. (Run 73)

- **Training seed variance is ~0.002.** Across seeds 0 and 42, val_bpb differs by 0.0016. This means improvements <0.001 are in the noise floor. (Run 72)

- **Weight decay IS necessary.** WD=0 gives +0.005 regression vs WD=0.1. Even with cautious masking, the regularization provides meaningful benefit. (Run 75)
