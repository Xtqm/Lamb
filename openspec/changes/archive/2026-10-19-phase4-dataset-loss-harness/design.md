# Design: Synthetic Trajectory Dataset & Unified Composite Loss Harness

## Context

Phase 1 through Phase 3 established and verified the core components of JED-M2:
- Multi-Head Latent Attention (MLA) and Sparse Top-2 MoE recurrent layer (`mla.py`, `moe.py`)
- Mamba Selective State Space Model (SSM) and JEPA Latent World Model (`mamba.py`, `jepa.py`)
- Dual-Process Action Decoupling Heads and 3-block unified LAM backbone (`heads.py`, `model.py`)

To optimize this ~45M parameter Large Action Model without external pretrained models (strict hard rule: no Hugging Face pretrained models or pipelines), Phase 4 constructs:
1. `jed_m2/dataset.py`: A native synthetic agent trajectory generator and streaming PyTorch Dataset modeled after WebArena and Mind2Web digital interaction traces, together with dynamic batch collation.
2. `jed_m2/losses.py`: A calibrated composite multi-task loss harness uniting reasoning, discrete tool classification, spatial coordinate regression, argument token prediction, JEPA latent world model prediction, and MoE load balancing.

Hardware constraints: All tensors and operations must run efficiently within the 8GB VRAM envelope (NVIDIA GeForce RTX 5060 Laptop GPU, sm_120 Blackwell architecture) in `bfloat16`, with standalone verification keeping peak memory under 2.0 GB.

## Goals / Non-Goals

**Goals:**
- Provide a synthetic multi-turn web/OS trajectory generator that synthesizes realistic DOM trees, UI bounding boxes, discrete tools (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH), continuous pointer coordinates $(x, y) \in [0, 1]$, and argument payloads.
- Implement `AgentTrajectoryDataset`, a memory-efficient streaming PyTorch Dataset that generates or streams multi-turn trajectories on demand without caching large static files in memory.
- Implement `TrajectoryCollator` providing padded token sequences, 2D causal attention masks, action token index mapping, component loss masks, and aligned JEPA $(s_t, a_t, s_{t+1})$ transition state pairs.
- Implement `JEDM2CompositeLoss` computing $\mathcal{L}_{\text{total}}$ with calibrated scaling weights ($\lambda_{\text{reason}}, \lambda_{\text{type}}, \lambda_{\text{coord}}, \lambda_{\text{arg}}, \lambda_{\text{jepa}}, \lambda_{\text{aux}}$) and comprehensive metric reporting.
- Implement `verify_phase4.py` verifying synthetic batch collation, forward loss evaluation, gradient backpropagation (asserting zero NaNs/Infs), and memory consumption on CUDA in `bfloat16`.

**Non-Goals:**
- Training the full model to convergence or running large-scale pretraining across epochs (deferred to Phase 5).
- Integrating external web browser drivers or headless Chrome environments (WebArena synthetic environment simulation satisfies training generation natively).
- Introducing external compiled packages (`mamba-ssm`, `causal-conv1d`).

## Decisions

### 1. Trajectory Structure and Tokenization Strategy
- **Decision**: Model trajectories as structured Python dataclasses (`InteractionTurn`, `TrajectorySample`) containing DOM elements, bounding boxes, thoughts, discrete actions, continuous $(x, y)$ coordinates, and string arguments.
- **Token Format**: Use a dedicated, deterministic vocabulary tokenizer matching `config.vocab_size = 8192`.
  - Special tokens: `<pad>=0`, `<bos>=1`, `<eos>=2`, `<obs>=3`, `</obs>=4`, `<thought>=5`, `</thought>=6`, `<action>=7`, `</action>=8`, `<arg>=9`, `</arg>=10`.
  - Turn structure in token sequence:
    `[<obs> DOM element summary </obs> <thought> multi-step rationale </thought> <action> discrete tool </action> <arg> argument payload </arg>]`
- **Alternatives Considered**:
  - *Raw character / ASCII encoding*: Lacks structural token boundaries, making action position indexing ambiguous.
  - *External Hugging Face tokenizer*: Violates the strict project hard rule forbidding pretrained Hugging Face pipelines.

### 2. Action Index Mapping & Dynamic Loss Masks
- **Decision**: The collator returns explicit index tensors (`action_indices`, `action_mask`) mapping which sequence positions represent action commitment points.
- **Component Masks**:
  - `reason_loss_mask`: Boolean mask active across `<thought> ... </thought>` rationale tokens for autoregressive LM loss.
  - `action_type_mask`: Binary mask active at the `<action>` token positions for discrete tool classification.
  - `coord_loss_mask`: Binary mask active only at action steps requiring spatial targeting (CLICK, TYPE, SCROLL).
  - `arg_loss_mask`: Binary mask active over `<arg> ... </arg>` tokens for argument payload cross-entropy.
- **Alternatives Considered**:
  - *Single combined loss mask with token type IDs*: Harder to handle heterogeneous loss functions (classification vs regression). Explicit boolean/binary masks ensure clean, vectorizable PyTorch loss calculations.

### 3. JEPA Observation Transition Pairs
- **Decision**: Consecutive observation turns $(s_t, s_{t+1})$ are paired with executed action $a_t$. The collator constructs observation representation tensors (`obs_t`, `obs_next` of shape `[B, hidden_dim]`) and action tensors (`action_t` of shape `[B]`) to directly feed `JEDM2Model.forward(..., obs_t=obs_t, action_t=action_t, obs_next=obs_next)`.
- **Alternatives Considered**:
  - *Separate JEPA dataloader*: Incurs redundant data loading and unaligned state batches. Joint collation ensures end-to-end multi-task synchronization.

### 4. Loss Formulation & Calibration
- **Decision**:
  - Reasoning loss ($\mathcal{L}_{\text{reason}}$): Autoregressive cross-entropy on thought tokens.
  - Action primitive loss ($\mathcal{L}_{\text{type}}$): Multi-class cross-entropy over discrete action types at action token indices.
  - Coordinate regression loss ($\mathcal{L}_{\text{coord}}$): Smooth L1 (Huber) loss between predicted $(x, y) \in [0, 1]$ and ground-truth normalized coordinates.
  - Argument loss ($\mathcal{L}_{\text{arg}}$): Masked cross-entropy over argument token payload.
  - JEPA loss ($\mathcal{L}_{\text{jepa}}$): Smooth L1 + cosine distance on next latent state representation.
  - MoE auxiliary loss ($\mathcal{L}_{\text{aux}}$): Load-balancing routing penalty directly from `JEDM2Output.aux_loss`.
  - Default scaling weights: $\lambda_{\text{reason}}=1.0, \lambda_{\text{type}}=1.0, \lambda_{\text{coord}}=5.0, \lambda_{\text{arg}}=1.0, \lambda_{\text{jepa}}=1.0, \lambda_{\text{aux}}=1.0$. Coordinate loss has higher weight due to small coordinate error magnitude ($[0, 1]^2$).

## Risks / Trade-offs

- **[Risk]** Variable sequence lengths across multi-turn trajectories could cause memory spikes or irregular padding.
  - **Mitigation**: Trajectory collator truncates or bounds total sequence length to $\le 512$ during verification (and $\le 2048$ for production training), padding with `<pad>=0` and setting causal attention masks accordingly.
- **[Risk]** Zero loss / NaN gradients when a batch contains zero spatial actions (e.g., all FINISH or NAVIGATE).
  - **Mitigation**: Coordinate and argument loss calculations check `mask.sum() > 0`. If `mask.sum() == 0`, return zero loss with gradient tracking (`0.0 * coords.sum()`).
- **[Risk]** Memory consumption during joint collation and multi-task loss backward pass.
  - **Mitigation**: Verification uses batch size 2, sequence length $\le 512$, with `bfloat16` precision on CUDA, staying well under the 2.0 GB verification limit.
