# Tasks: Phase 2 - Sequential Recurrence & Latent World Model

## 1. Global Configuration & Mamba SSM Module

- [x] 1.1 Update `jed_m2/config.py` with Mamba hyperparameters (`mamba_d_state`, `mamba_d_conv`, `mamba_expand`, `mamba_dt_rank`) and JEPA hyperparameters (`jepa_latent_dim`, `jepa_ema_decay`, `jepa_loss_cos_weight`), and verify configuration instantiation in Python.
- [x] 1.2 Implement `jed_m2/mamba.py` containing `MambaBlock` in pure PyTorch with in-projection, depthwise causal 1D convolution, input-dependent $(\Delta, B, C)$ projection, discretized state update, and gated out-projection; verify forward pass and gradient backpropagation.
- [x] 1.3 Implement autoregressive `step(x_t, ssm_state, conv_state)` method in `MambaBlock` with rolling buffer state tracking; verify mathematical equivalence between sequential `step` rollout and full-sequence `forward` within $10^{-2}$ tolerance in `bfloat16`.

## 2. JEPA Latent World Model

- [x] 2.1 Implement `LatentStateEncoder` and `EMATargetEncoder` in `jed_m2/jepa.py` with stop-gradient target evaluation and Polyak momentum update `update_target_network()`; verify parameter detach and EMA tracking.
- [x] 2.2 Implement `LatentPredictor` in `jed_m2/jepa.py` predicting next latent state from current latent state and action embedding; verify output shape `[B, D_latent]` and gradient flow.
- [x] 2.3 Implement unified `JEPAWorldModel` class in `jed_m2/jepa.py` combining encoders, predictor, and composite Smooth L1 + cosine distance loss; verify end-to-end forward loss and backward pass with gradients isolated from target encoder.

## 3. Module Exports, Verification Suite & Context Update

- [x] 3.1 Update `jed_m2/__init__.py` to export `MambaBlock`, `LatentStateEncoder`, `EMATargetEncoder`, `LatentPredictor`, and `JEPAWorldModel`; verify clean package imports.
- [x] 3.2 Implement and execute `verify_phase2.py` covering forward passes, step equivalence assertions, backward passes (zero NaNs/Infs), parameter counts, and peak CUDA VRAM profiling in `bfloat16`; verify all assertions pass.
- [x] 3.3 Update `PROJECT_CONTEXT.md` roadmap marking Phase 2 as complete with verified parameter and VRAM metrics.
