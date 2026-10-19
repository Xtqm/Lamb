# Proposal: Phase 3 - Action Heads & Unified LAM Backbone

## Why

Phases 1 and 2 established the core attention, routing, recurrence, and representation components: Multi-Head Latent Attention (MLA), Sparse Top-2 Mixture of Experts (MoE), pure PyTorch Mamba SSM, and JEPA latent world model. However, these modules currently exist as decoupled primitives without an action decoding interface or unified model backbone. 

To function as a native Large Action Model (LAM), JED-M2 requires:
1. A dual-system action decoding subsystem (System 1 action emission vs System 2 latent reasoning continuation) that predicts discrete interaction primitives (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH), continuous normalized spatial coordinates $(x, y) \in [0, 1]$, and tokenized payload arguments.
2. A unified 3-layer recurrent backbone (`JEDM2Model`) interleaving Mamba recurrence, low-rank latent attention, and sparse expert routing, operating within a ~45M total parameter budget (~14M active parameters) and an 8GB VRAM envelope (< 2.0 GB during active verification in `bfloat16`).
3. Support for both sequence-parallel training passes and token-by-token autoregressive recurrent rollouts (`step(...)`) preserving KV-cache and Mamba rolling state buffers.

## What Changes

- **Dual-System Action Decoding Heads (`jed_m2/heads.py`)**:
  - `ActionGate`: Projection from `hidden_dim -> 1` predicting scalar probability/logits for emitting an executable action versus continuing latent reasoning.
  - `ActionTypeHead`: MLP (`hidden_dim -> hidden_dim // 2 -> GELU -> Dropout -> num_action_types`) predicting discrete UI primitive distributions (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH).
  - `CoordinateHead`: Spatial pointer regressor (`hidden_dim -> hidden_dim // 4 -> GELU -> 2 -> Sigmoid`) predicting bounded screen coordinates $(\hat{x}, \hat{y}) \in [0, 1]$.
  - `ArgumentHead`: Linear projection (`hidden_dim -> vocab_size`) for tokenized payload arguments.
  - `NativeActionHead`: Composite module coordinating all four sub-heads into a unified output mapping.
- **Unified LAM Backbone (`jed_m2/model.py`)**:
  - `JEDM2Block`: Composite recurrent block interleaving RMSNorm -> MambaBlock -> RMSNorm -> MultiHeadLatentAttention -> RMSNorm -> SparseTop2MoE with residual connections.
  - `JEDM2Model`: Full LAM backbone stacking 3 `JEDM2Block` layers over observation embeddings and `JEPAWorldModel`, feeding `NativeActionHead` and autoregressive LM head.
  - Autoregressive rollout `step(...)` method supporting state progression preserving Mamba conv/SSM state buffers and MLA KV cache across all 3 stacked layers.
- **Module Exports (`jed_m2/__init__.py`)**:
  - Expose `ActionGate`, `ActionTypeHead`, `CoordinateHead`, `ArgumentHead`, `NativeActionHead`, `ActionType`, `JEDM2Block`, and `JEDM2Model`.
- **Phase 3 Verification Suite (`verify_phase3.py`)**:
  - Standalone script testing forward pass, dual action output shapes, recurrent rollout `step(...)` execution, backward pass (asserting no NaNs/Infs), parameter scaling (~45M total / ~14M active), and peak VRAM allocation (< 2.0 GB) on CUDA in `bfloat16`.

## Capabilities

### New Capabilities
- `dual-system-action-heads`: Action decoding heads predicting action emission gating, discrete action primitives, continuous spatial pointer coordinates, and tokenized arguments.
- `unified-lam-backbone`: 3-layer LAM backbone stacking Mamba SSM, MLA, and Sparse Top-2 MoE blocks over observation embeddings, JEPA latent world model, and action heads with both sequence forward and recurrent step rollout modes.

### Modified Capabilities
<!-- No requirement changes to existing specs (mamba-ssm, jepa-world-model). -->

## Impact

- **Affected Code**: `jed_m2/heads.py` (new), `jed_m2/model.py` (new), `jed_m2/__init__.py` (updated), `jed_m2/mla.py` (support KV-cache in step mode if needed), `verify_phase3.py` (new).
- **APIs**: Exposes `JEDM2Model`, `JEDM2Block`, and `NativeActionHead` for trajectory training and rollout generation in Phase 4.
- **Hardware/Dependencies**: Pure PyTorch 2.x in `bfloat16`. Parameter envelope strictly ~45M total / ~14M active; peak VRAM < 2.0 GB on RTX 5060 Laptop GPU.
