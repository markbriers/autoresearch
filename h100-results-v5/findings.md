# Findings — v5 (Research-Only)

## Confirmed Mechanisms

- **Pre-attention causal depthwise conv helps Q/K**: A short causal depthwise conv (kernel=4) on Q/K inputs improves val_bpb by -0.0017. Locally-smoothed features make better attention queries/keys. BUT: severe throughput penalty (30% vs 41% MFU) because depthwise Conv1d breaks torch.compile fusion.
- **Per-head learnable attention temperature helps (marginally)**: With QK-norm fixing attention sharpness, a learnable per-head temperature (via Q scaling) provides additional -0.0002 improvement. Very cheap (50 params, zero VRAM overhead).
- **Partial RoPE (75%) is the biggest win**: Applying RoPE to only 75% of head dimensions, leaving 25% position-free, improves val_bpb by -0.0035. This is the largest single improvement. Position-free dimensions enable pure content-based attention matching.
- **Conv smoothing should NOT apply to V**: Smoothing values blurs token-level distinctions. V benefits from per-token granularity.
- **Pre-attn conv and partial RoPE are COMPLEMENTARY**: Removing conv from partial RoPE model causes +0.003 regression. They address different bottlenecks: conv provides local n-gram context, partial RoPE provides position-free semantic matching.

## Dead Ends

- **SwiGLU OOM at DBS=128**: Adding a gate projection to the 4x MLP increases params by ~19% and activation memory proportionally. At 67.8GB baseline with DBS=128, any change that adds significant parameters or activation memory will OOM. Future hypotheses must be VRAM-neutral or VRAM-reducing.
- **Z-loss conflicts with softcap**: When logits are already bounded by tanh softcap (±12), z-loss adds redundant regularization that hurts performance (+0.004 val_bpb). Don't add auxiliary losses that duplicate existing constraints.
- **VRAM constraint is binding**: 67.8GB baseline means only ~12GB headroom. With pre-attn conv at 71GB, only ~9GB headroom. Even parameter-free changes (like extra norm) can OOM due to activation memory for backward pass.
- **Sandwich norm OOM**: Post-norm saves intermediate activations for backward that exceed VRAM budget. Even "zero-param" changes have activation memory costs.
- **Learnable RoPE frequencies OOM**: Making cos/sin depend on a parameter requires saving intermediates for backward, OOMing. Any change that adds new operations in the computation graph risks OOM.

## Architecture Inductive Biases

- **K-norm is essential with Muon**: Removing K normalization while keeping Q-norm causes +0.003 regression. Both Q and K must be normalized for stable training with Muon optimizer. K magnitude variation does not usefully encode salience.
- **RoPE base 10000 is optimal for seq_len=2048**: Increasing to 50000 slightly regresses. The default base is well-matched to the context length.

## Cross-Domain Transfer Patterns

## Open Questions
