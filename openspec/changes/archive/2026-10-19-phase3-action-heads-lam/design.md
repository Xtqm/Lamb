# Technical Design: Phase 3 - Action Heads & Unified LAM Backbone

## Context

Phases 1 and 2 implemented and verified the individual building blocks of JED-M2:
- Low-rank Multi-Head Latent Attention (`MLA`, ~0.70M params) with decoupled RoPE.
- Sparse Top-2 Mixture of Experts (`MoE`, ~12.59M params) with SwiGLU routing and auxiliary load-balancing loss.
- Pure PyTorch Mamba Selective State Space Model (`MambaBlock`, ~1.69M params) with continuous-time ZOH discretization and constant-memory recurrent step updates.
- Latent World Model (`JEPAWorldModel`, ~1.61M params) with EMA momentum target updates and representation consistency loss.

See `proposal.md` for motivation. Phase 3 unifies these primitives into an end-to-end Large Action Model (`JEDM2Model`) and constructs the dual-system action decoding heads (`heads.py`).

## Goals / Non-Goals

**Goals:**
- Implement `jed_m2/heads.py`:
  - `ActionGate`: Scalar projection (`hidden_dim -> 1`) predicting action emission vs rationale continuation.
  - `ActionTypeHead`: MLP (`hidden_dim -> hidden_dim // 2 -> GELU -> Dropout -> num_action_types`) classifying UI action primitives (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH).
  - `CoordinateHead`: Spatial pointer regressor (`hidden_dim -> hidden_dim // 4 -> GELU -> 2 -> Sigmoid`) predicting bounded screen coordinates in $[0, 1]^2$.
  - `ArgumentHead`: Linear projection (`hidden_dim -> vocab_size`) for tokenized arguments.
  - `NativeActionHead`: Composite module coordinating all four heads into structured action dictionaries.
- Implement `jed_m2/model.py`:
  - `JEDM2Block`: Composite recurrent layer executing RMSNorm -> MambaBlock -> RMSNorm -> MLA -> RMSNorm -> SparseTop2MoE with residual connections and MoE auxiliary loss collection.
  - `JEDM2Model`: Full 3-block LAM backbone stacking observation embeddings, blocks, autoregressive LM head, `JEPAWorldModel`, and `NativeActionHead`.
  - Recurrent rollout `step(...)` method supporting single-token autoregressive generation while propagating Mamba conv/SSM buffers and MLA KV-caches across all 3 layers.
- Expose all new classes in `jed_m2/__init__.py`.
- Implement `verify_phase3.py` demonstrating:
  - Forward pass output tensor validity.
  - Dual action output shape correctness.
  - Multi-step recurrent rollout `step(...)` execution.
  - Backward pass gradient integrity (asserting zero NaNs and Infinities across all trainable parameters).
  - Parameter scaling (~45M total / ~14M active parameters).
  - Peak VRAM allocation (< 2.0 GB) on CUDA in `bfloat16`.

**Non-Goals:**
- Offline dataset loading or trajectory batching (reserved for Phase 4).
- End-to-end training loop with fused 8-bit AdamW optimizer (reserved for Phase 5).
- Compiled C++/CUDA extensions (`mamba-ssm`, `causal-conv1d`).

## Decisions

### 1. Dual-System Action Head Architecture (`heads.py`)
- **System 1 (Action Heads) vs System 2 (Latent Rationale)**:
  - System 2 generates internal thought tokens autoregressively through the LM head to formulate plans before taking action.
  - System 1 projects the model's current latent state $h_t$ to an action tuple $a_t = (\text{gate}, \text{type}, (x, y), \text{arg})$.
- **Primitives**:
  - `ActionType` enum:
    - 0: `CLICK`
    - 1: `TYPE`
    - 2: `SCROLL`
    - 3: `NAVIGATE`
    - 4: `API_CALL`
    - 5: `FINISH`
- **Output Container**:
  - `NativeActionHead.forward(...)` returns `ActionOutput` (or dictionary) containing `gate_logits`, `action_type_logits`, `coordinates`, and `argument_logits`.
  - `predict_action(...)` provides a clean inference-time decoding utility applying sigmoid to the gate and softmax/argmax to type and argument distributions.

### 2. Composite `JEDM2Block` Interleaving Order
- **Layer Flow**:
  1. $h_1 = x + \text{MambaBlock}(\text{RMSNorm}_1(x))$
  2. $h_2 = h_1 + \text{MultiHeadLatentAttention}(\text{RMSNorm}_2(h_1))$
  3. $h_3 = h_2 + \text{SparseTop2MoE}(\text{RMSNorm}_3(h_2))$
- **Rationale**:
  - Mamba acts first as a linear sequential filter capturing broad temporal dependencies and local 1D convolution context.
  - MLA follows to perform selective content-based associative retrieval over compressed latent queries and keys.
  - MoE provides massive feed-forward parameter capacity with sparse conditional compute (only 2 of 8 experts active per token).

### 3. Recurrent Rollout State Management
- **Mamba State**: Each layer maintains `ssm_state` $[B, D_{inner}, d_{state}]$ and `conv_state` $[B, D_{inner}, d_{conv}]$.
- **MLA KV-Cache**: Each layer caches $K \in [B, \text{num\_heads}, t, d_k]$ and $V \in [B, \text{num\_heads}, t, d_v]$ (or compressed representations), updated dynamically at each rollout step $t$.
- **Unified State Container**: A list of per-layer state tuples `[(ssm_state_l, conv_state_l, kv_cache_l), ...]` passed to and returned from `model.step(...)`.

### 4. Parameter and Budget Calibration
- Stacking 3 blocks:
  - 3x `MambaBlock`: $3 \times 1.69\text{M} = 5.07\text{M}$ params.
  - 3x `MultiHeadLatentAttention`: $3 \times 0.70\text{M} = 2.10\text{M}$ params.
  - 3x `SparseTop2MoE`: $3 \times 12.59\text{M} = 37.77\text{M}$ params.
  - Embeddings & heads: `tok_embeddings` ($8192 \times 512 = 4.19\text{M}$), `ActionGate` + `ActionTypeHead` + `CoordinateHead` (~0.2M), `ArgumentHead` (4.19M or tied), `JEPAWorldModel` (~1.61M).
  - Total parameter count: ~45M - 50M total parameters.
  - Active parameter count per token: ~14M - 17M parameters.
  - Memory consumption in `bfloat16`: Model weights ~95 MB. Peak VRAM during sequence length 1024 forward + backward pass remains well below the 2.0 GB threshold on the RTX 5060 Laptop GPU.

## Risks / Trade-offs

- **[Risk 1] KV-Cache concatenation overhead during multi-step rollout**:
  - *Mitigation*: For verification and short agent rollouts, simple tensor concatenation along the sequence dimension is fast and clean in PyTorch without memory fragmentation.
- **[Risk 2] Auxiliary loss accumulation across 3 MoE layers**:
  - *Mitigation*: `JEDM2Block` returns `(hidden_out, aux_loss)`, and `JEDM2Model` computes the mean or sum across the 3 layers, seamlessly incorporating $\mathcal{L}_{\text{aux}}$ into backward gradients.
- **[Risk 3] Precision or shape mismatch between sequence mode and single-step mode**:
  - *Mitigation*: Ensure `step(...)` accepts unbatched or single-step inputs `[B, D]` consistently and returns explicit shapes matching verification assertions.
