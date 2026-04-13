# Findings

## Confirmed Mechanisms
Things that work, with mechanistic explanations for *why* they work on this architecture.

- **Stronger input skip connections (x0_lambdas=0.2) help shallow models.** With only 4 layers, each layer's contribution is relatively large, so the initial representation needs more weight in the residual stream to prevent early layers from dominating. (Run 3, -0.005 val_bpb)

- **Shorter warmdown ratio (0.3 vs 0.5) is better with few steps.** With only 321 steps, 50% warmdown means the model spends too long at reduced LR. (Run 2, -0.001 val_bpb)

- **Optimal batch size is DEVICE_BATCH_SIZE=8, TOTAL_BATCH_SIZE=16K (~1200 steps).** The batch-size/step-count tradeoff has a U-shaped optimum. Going from 32→16→8 each improved val_bpb (-0.012, -0.008), but batch=4 (8K tokens/step) regressed badly (+0.024) due to excessive gradient noise. The sweet spot gives ~1200 steps with 6.5 GB VRAM. (Runs 8-10)

## Dead Ends
Approaches that failed, with explanations for *why* they failed. The failure mode matters
more than the fact of failure — it reveals what this architecture cannot absorb.

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

- **Does MLP expansion ratio matter?** Current 4× might be suboptimal with ReluSquared's heavy sparsification. Test: 6× or 8× expansion (adds params but uses VRAM headroom).

- **Deeper-thinner wins on H100: 12×384 beats 8×512 by 0.024 val_bpb.** At 500 steps and 264M tokens, depth (compositional power) is more valuable than width (feature diversity). 12×384 has 46.4M params vs 50.3M for 8×512, so it's also slightly smaller. Deeper models at same width (12×512) are worse due to doubled params → halved steps. The sweet spot is depth increase with proportional width decrease to keep params constant. (Runs H4a, H4b)

- **MPS batch-size-dependent findings don't transfer to H100.** WD=0.1 (best on MPS with batch=8) is worse than WD=0.2 on H100 (batch=524K). Large batch cleans gradients, making more regularization (WD) beneficial. (Run H3)

- **DiffAttn loses params in FA3-compatible implementation.** Single-FA3-call DiffAttn requires V at sub_head_dim, halving c_v and c_proj. The 21% param reduction outweighs any attention quality gain. (Run H2)

## H100 NVL — New Design Space

With CUDA, torch.compile, and FA3, the following constraints from MPS are LIFTED:
- Flash Attention 3 supports sliding window, differential attention, and arbitrary head counts efficiently
- torch.compile eliminates Python overhead
- 94GB VRAM provides massive headroom (baseline uses 45GB)
- More heads are viable (no MPS SDPA per-head memory scaling issue)

**Sliding window size is a throughput lever, not just a quality tradeoff.** Reducing window from half-context (1024) down to 1/8-context (256) progressively improved val_bpb despite limiting attention range, because the step count increase (969→1057) more than compensates. Only the last layer needs full context. Window=128 is too small. (Runs H22, H37-H39)

**FINAL_LR_FRAC=0.05 helps.** Keeping 5% of peak LR during warmdown prevents the model from fully freezing at the end of training, allowing continued refinement. 10% is too high. (Runs H32-H33)

**VE param overhead limits depth scaling.** VE embeddings are vocab_size × kv_dim per VE layer. At depth 20 with 10 VE layers, this inflates the model to 73M params, killing throughput. This means optimal depth depends on VE frequency: alternating VE at depth 12 is the current sweet spot. (Runs H5, H24, H28)

- **Optimal depth/width scales jointly.** At 384-dim, 10 layers beat 12 layers (which beat 8). But at 512-dim, 10 layers is still optimal. The sweet spot is 10×512 (60.8M params, 846 steps). Going wider to 640 or deeper to 11+ hurts because of step count reduction. (Runs H47-H48, H41, H55)

- **Warmdown scales with step count.** With ~1000 steps, warmdown=0.6 is optimal. With ~500 steps, 0.4 was best. With ~300 steps (MPS), 0.3 was best. The pattern: warmdown_ratio ≈ 0.0006 × step_count. (Runs H19-H20, H44-H46)

- **Muon LR is remarkably stable at 0.04 across batch sizes 262K-524K on H100.** The LR scales with batch size on MPS (batch=8K→LR=0.02) but at H100 batch sizes, the optimal Muon LR is 0.04 regardless of whether batch is 262K or 524K. (Runs H6-H8, H14-H16, H49-H50)

**Open H100 Questions:**
- Would reducing VE to lower-rank allow deeper models?
- Can Mixture-of-Depths or token routing allocate compute to harder tokens?
- Is there a SwiGLU formulation that's faster than ReluSquared under torch.compile?
