# Findings

## Confirmed Mechanisms

### F1: xIELU Activation (H1, Cycle 1) -- delta = -0.003633
Replaces ReluSquared with learnable per-layer piecewise activation. Replicates v6 finding. 20 scalar params, zero VRAM overhead. Drop-in replacement.

### F2: SwiGLU Gated MLP (H3, Cycle 1) -- delta = -0.003363
Three-matrix gated MLP at hidden_dim=1664 (parameter-matched). Replicates v4 finding. VRAM slightly below baseline. Note: H1 and H3 are mutually exclusive (both modify MLP path).

### F3: Learnable Per-Head Attention Temperature (H4, Cycle 1) -- delta = -0.004839
Strongest single intervention. 60 scalar params (6 heads x 10 layers). Orthogonal to MLP-path changes. Strong stacking candidate with H1 or H3. Implementation note: cast scalar params to tensor dtype before multiplication.

## Dead Ends

### D1: Peri-LN Post-Sublayer Normalization (H2, Cycle 1) -- delta = -0.001184, VRAM +6.4 GB
Failed both val_bpb and VRAM thresholds. The existing resid_lambdas/x0_lambdas already control residual magnitude, making additional post-sublayer norm redundant. The 6.4 GB VRAM cost (storing norm activations for backward) is disproportionate. Future normalisation hypotheses should avoid the residual path.

## Architecture Inductive Biases

## Cross-Domain Transfer Patterns

## Open Questions

1. H1 vs H3 for stacking with H4: xIELU slightly outperforms SwiGLU standalone, but which stacks better with attention temperature?
2. Could xIELU replace SiLU inside SwiGLU's gate (xIELU-gated GLU)?
3. What did the 60 temperature scalars in H4 converge to? Inspecting learned values would validate the mechanism.
