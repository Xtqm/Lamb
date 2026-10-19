# Proposal

## Why

With Phases 1-3 complete and verified (MLA, MoE, Mamba SSM, JEPA World Model, Dual-Process Action Heads, and Unified LAM Backbone), the JED-M2 model requires a specialized data pipeline and an end-to-end composite training loss harness to enable model optimization.

Standard causal LM datasets only contain raw text tokens and lack structured multi-turn action annotations (DOM trees, UI bounding boxes, discrete tool types, spatial screen coordinates, and rationale masks). Phase 4 introduces a synthetic multi-turn web/OS trajectory generator and streaming PyTorch Dataset (`jed_m2/dataset.py`) modeled after WebArena / Mind2Web, paired with a unified composite loss harness (`jed_m2/losses.py`) that calibrates and optimizes reasoning, discrete action classification, continuous coordinate regression, argument token decoding, JEPA latent consistency, and MoE routing uniformity.

## What Changes

- Implement `jed_m2/dataset.py`:
  - Synthetic Web/OS trajectory generator modeling realistic multi-turn agent interaction tuples: $\tau = (s_0, a_0, r_0, s_1, a_1, \dots, s_T)$ with DOM element trees, UI bounding boxes, discrete tools (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH), continuous spatial coordinates $(x, y) \in [0, 1]$, and argument payloads.
  - Streaming PyTorch Dataset (`AgentTrajectoryDataset`) yielding structured multi-turn trajectories with configurable horizon $T$ and reasoning steps.
  - Dynamic trajectory collator (`TrajectoryCollator`) constructing unified padded token sequences, 2D causal attention masks, action token index mapping, and component loss masks (reasoning mask, action type mask, coordinate mask, argument mask, and JEPA transition pairs).
- Implement `jed_m2/losses.py`:
  - `JEDM2LossConfig`: Dataclass specifying loss scaling coefficients ($\lambda_{\text{reason}}, \lambda_{\text{type}}, \lambda_{\text{coord}}, \lambda_{\text{arg}}, \lambda_{\text{jepa}}, \lambda_{\text{aux}}$) and Huber/Smooth L1 thresholds.
  - `JEDM2CompositeLoss`: Unified loss harness computing:
    1. Reasoning Loss ($\mathcal{L}_{\text{reason}}$): Autoregressive cross-entropy on thought/rationale tokens.
    2. Action Primitive Loss ($\mathcal{L}_{\text{type}}$): Multi-class cross-entropy on discrete tool categories.
    3. Coordinate Loss ($\mathcal{L}_{\text{coord}}$): Smooth L1 / Huber loss on predicted $(x, y) \in [0, 1]$ screen coordinates.
    4. Argument Loss ($\mathcal{L}_{\text{arg}}$): Masked cross-entropy on argument token payloads.
    5. JEPA Latent Consistency Loss ($\mathcal{L}_{\text{jepa}}$): Smooth L1 + cosine distance between predicted $\hat{z}_{t+1}$ and detached EMA target $z_{t+1}$.
    6. MoE Load-Balancing Loss ($\mathcal{L}_{\text{aux}}$): Auxiliary routing penalty from SparseTop2MoE.
  - Logging metric dictionary returning detached scalar values for monitoring and optimization tracking.
- Expose dataset, collator, and loss harness in `jed_m2/__init__.py`.
- Implement `verify_phase4.py`:
  - Comprehensive standalone verification script testing trajectory generation, dynamic batch collation, forward loss evaluation, full backward pass with gradient assertions (zero NaNs/Infs), and memory consumption on CUDA in `bfloat16` within the 2.0 GB allocation budget.

## Capabilities

### New Capabilities
- `synthetic-trajectory-dataset`: Synthetic agent trajectory generation, streaming PyTorch Dataset, and dynamic batch collation for multi-turn web/OS interactions.
- `composite-loss-harness`: Unified multi-task loss harness for JED-M2 combining reasoning, action classification, spatial coordinates, argument decoding, JEPA latent consistency, and MoE load balancing.

### Modified Capabilities
<!-- None: Existing Phase 1-3 specifications remain unchanged. -->

## Impact

- New modules added: `jed_m2/dataset.py`, `jed_m2/losses.py`, `verify_phase4.py`.
- Modified modules: `jed_m2/__init__.py` (re-exports new classes).
- Dependencies: Pure PyTorch, zero external C++/CUDA packages or pretrained Hugging Face pipelines.
- Hardware envelope: Verified on NVIDIA GeForce RTX 5060 Laptop GPU in `bfloat16`, peak VRAM well within the 2.0 GB verification envelope.
