# Proposal: Phase 2 - Sequential Recurrence & Latent World Model

## Why

JED-M2 requires constant-memory long-horizon sequence modeling and abstract state representation to function as a native Large Action Model (LAM). Transformers with quadratic attention degrade over extended digital interaction trajectories, and raw pixel/token-level prediction introduces unnecessary high-frequency noise. Implementing a pure PyTorch Mamba Selective State Space Model (SSM) provides linear-time recurrence and constant-memory recurrent inference, while a Joint-Embedding Predictive Architecture (JEPA) provides self-supervised abstract world modeling without pixel decoding overhead.

## What Changes

- **Mamba SSM Recurrence (`jed_m2/mamba.py`)**:
  - Implement `MambaBlock` in 100% native PyTorch without external compiled C++/CUDA dependencies (avoiding MSVC/Blackwell `sm_120` compatibility issues).
  - Implement selective scan discretization ($\bar{A} = \exp(\Delta A)$, $\bar{B} = \Delta B$) with input-dependent $\Delta$, $B$, and $C$.
  - Implement 1D causal depthwise convolution in pure PyTorch with state caching for fast autoregressive stepping.
  - Implement `step(...)` method for constant-memory $O(1)$ token-by-token state updates during recurrent rollouts.
  - Provide mathematical equivalence between sequence `forward(...)` and iterative `step(...)`.
- **JEPA Latent World Model (`jed_m2/jepa.py`)**:
  - Implement `LatentStateEncoder` mapping raw observation vectors $o_t$ to latent space $z_t$.
  - Implement `EMATargetEncoder` maintaining an exponential moving average (momentum $\tau$) of encoder parameters with stop-gradient targets.
  - Implement `LatentPredictor` predicting $\hat{z}_{t+1} = \text{Predictor}(z_t, a_t)$ conditioned on discrete/continuous action representations.
  - Implement composite JEPA loss combining Smooth L1 (Huber) loss and cosine similarity distance.
- **Global Configuration Updates (`jed_m2/config.py`)**:
  - Expand `JEDM2Config` with Mamba hyperparameters (`mamba_d_state`, `mamba_d_conv`, `mamba_expand`, `mamba_dt_rank`) and JEPA hyperparameters (`latent_dim`, `jepa_ema_decay`).
- **Module Exports (`jed_m2/__init__.py`)**:
  - Export `MambaBlock`, `LatentStateEncoder`, `EMATargetEncoder`, `LatentPredictor`, and `JEPAWorldModel`.
- **Phase 2 Verification Suite (`verify_phase2.py`)**:
  - Standalone script testing forward pass, backward gradient validity (no NaNs or Infs), exact sequence-step equivalence, parameter sizing (~45M envelope), and peak VRAM profiling on CUDA in `bfloat16`.

## Capabilities

### New Capabilities
- `mamba-ssm`: Pure PyTorch selective state space model block supporting both sequence parallel forward passes and constant-memory recurrent step updates.
- `jepa-world-model`: Joint-Embedding Predictive Architecture providing latent observation encoding, momentum target updates, action-conditioned transition prediction, and representation consistency loss.

### Modified Capabilities
<!-- No existing capability specs exist prior to this change. -->

## Impact

- **Affected Code**: `jed_m2/config.py`, `jed_m2/__init__.py`, `jed_m2/mamba.py` (new), `jed_m2/jepa.py` (new), `verify_phase2.py` (new).
- **APIs**: Exposes `MambaBlock` and `JEPAWorldModel` public classes for downstream integration in Phase 3 (`jed_m2/model.py`).
- **Hardware/Dependencies**: Pure PyTorch 2.x without external C++/CUDA builds (`causal-conv1d` or `mamba-ssm` are prohibited). Strict `bfloat16` support within RTX 5060 Laptop GPU memory envelope (< 500 MB peak VRAM for Phase 2).
