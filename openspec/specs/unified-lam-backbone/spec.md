# unified-lam-backbone Specification

## Purpose

Provides a unified Large Action Model backbone that stacks interleaved Mamba SSM recurrence, Multi-Head Latent Attention, and Sparse Top-2 Mixture of Experts over shared observation embeddings, JEPA latent world model, and dual-system action decoding heads.

## Requirements

### Requirement: Interleaved Recurrent JEDM2Block Execution
The system SHALL support modular recurrent block execution, combining optional Mamba state space recurrence (`use_mamba`), configurable attention (`attn_type` in `["mla", "mha"]`), and configurable feed-forward networks (`ffn_type` in `["moe", "dense"]`) with pre-RMSNorm and residual connections.

#### Scenario: Forward transformation through recurrent block
- **WHEN** a tensor of shape `[B, S, D]` is passed through `JEDM2Block` with default configuration (`use_mamba=True`, `attn_type="mla"`, `ffn_type="moe"`)
- **THEN** the block returns an output tensor of shape `[B, S, D]` along with the auxiliary routing loss accumulated from the Sparse Top-2 MoE

#### Scenario: Forward transformation through ablation baseline blocks
- **WHEN** a tensor of shape `[B, S, D]` is passed through `JEDM2Block` configured as a pure Transformer block (`use_mamba=False`, `attn_type="mha"`, `ffn_type="dense"`) or dense hybrid block (`use_mamba=True`, `attn_type="mla"`, `ffn_type="dense"`)
- **THEN** the block applies the selected attention and FFN modules with residual connections and returns an auxiliary loss of zero

### Requirement: Stacked 3-Layer LAM Backbone Assembly
The system SHALL construct a unified architecture stacking observation embeddings, configurable recurrent block layers, an autoregressive language modeling head, an optional contextual JEPA world model, and a composite action decoding head across configurable layer depths.

#### Scenario: Full backbone sequence forward pass
- **WHEN** input token sequences `[B, S]` and optional observation/action tuples are evaluated by `JEDM2Model`
- **THEN** the model outputs LM rationale logits `[B, S, vocab_size]`, composite action predictions across sequence positions, and accumulated MoE auxiliary loss

#### Scenario: End-to-end JEPA gradient propagation from backbone representations
- **WHEN** JEPA state transition targets are evaluated during backbone forward pass with `use_jepa=True`
- **THEN** the model routes contextual backbone hidden states $h_t$ from the recurrent stack into `self.jepa`, allowing JEPA consistency loss gradients to flow directly into the recurrent blocks and token embeddings

#### Scenario: Backbone forward pass without JEPA world model
- **WHEN** `JEDM2Model` is initialized with `use_jepa=False` and evaluated on input tokens
- **THEN** the model performs forward propagation through all recurrent blocks and action heads without evaluating JEPA latent state transitions, returning `jepa_loss=None`

### Requirement: Autoregressive Recurrent Rollout Mode
The system SHALL support incremental token-by-token state updates via a `step` method preserving recurrent states across layers, supporting both MLA compressed latent KV-caches and standard MHA multi-head KV-caches with optional Mamba state space buffers.

#### Scenario: Multi-step recurrent rollout with state persistence
- **WHEN** single tokens `[B, 1]` or `[B]` are passed sequentially through `step` with previous recurrent state containers
- **THEN** the model updates internal states with recurrent step complexity and outputs current step rationale and action predictions

#### Scenario: True MLA latent cache compression and weight absorption
- **WHEN** autoregressive generation steps are executed in Multi-Head Latent Attention
- **THEN** the KV-cache stores only the compressed latent representation $c_{KV} \in \mathbb{R}^{B, 1, d_c}$ and shared decoupled RoPE key $K_{\text{rope}} \in \mathbb{R}^{B, 1, d_R}$, absorbing $W_{UK}$ into $W_Q$ while matching sequence forward attention outputs within floating point tolerance

#### Scenario: Standard MHA recurrent rollout step
- **WHEN** autoregressive generation steps are executed with `attn_type="mha"`
- **THEN** the recurrent step stores standard uncompressed Key-Value tensors $[B, H, S, d_h]$ in the KV-cache, demonstrating token-dependent $O(N)$ cache growth across recurrent steps

### Requirement: Parameter Scaling and Memory Budget Compliance
The system SHALL support parameter-matched configurations calibrated within $50\text{M} \pm 5\%$ total parameters across 3-layer and 4-layer architectures, executing forward and backward passes under the 6.0 GB VRAM limit on CUDA in `bfloat16`.

#### Scenario: Hardware envelope verification
- **WHEN** the full `JEDM2Model` executes sequence forward and backward passes with sequence length 1024 on CUDA in `bfloat16`
- **THEN** total parameters remain within $45\text{M} \pm 5\text{M}$, active parameters remain within $14\text{M} \pm 3\text{M}$, peak VRAM remains below 2.0 GB, and all gradients are free of NaNs or Infinities

#### Scenario: Baseline parameter parity verification
- **WHEN** any of the four baseline models (Baseline A, B, C, D) executes sequence forward and backward passes on CUDA in `bfloat16`
- **THEN** total parameters remain within $50.0\text{M} \pm 5\%$ ($47.5\text{M} \le P \le 52.5\text{M}$ for A/B/C and $\le 53.5\text{M}$ with JEPA), peak allocated VRAM remains below 6.0 GB, and all gradients are free of NaNs or Infinities

### Requirement: Selective Gradient Checkpointing Support
The system SHALL provide selective activation checkpointing across the stacked JEDM2Block layers within JEDM2Model to constrain backpropagation activation memory at long sequence lengths.

#### Scenario: Checkpointed sequence forward and backward pass
- **WHEN** gradient checkpointing is enabled on JEDM2Model and a sequence of length 2048 is evaluated through backward pass on CUDA in bfloat16
- **THEN** intermediate activations are recomputed during backward pass, gradients are computed without NaNs or Infs, and peak activation memory is reduced compared to non-checkpointed training

### Requirement: Dual Multimodal Observation Ingestion
The system SHALL ingest dual multimodal observations combining visual screenshot patch tokens and discrete DOM/text token sequences into a unified sequence representation.

#### Scenario: Sequence forward pass with visual patch tokens
- **WHEN** visual inputs (`images` of shape `[B, 3, H, W]` or `visual_tokens` `[B, N_v, D]`) and text token sequences `[B, S]` are provided to `JEDM2Model`
- **THEN** the model fuses visual tokens with text tokens via early-fusion prefixing and processes the joint sequence through all recurrent blocks

### Requirement: Visually Grounded Action Decoding in LAM Backbone
The system SHALL forward visual patch representations alongside hidden states to the action decoding heads to visually ground spatial coordinate regressions.

#### Scenario: Grounded action prediction in forward pass
- **WHEN** multimodal observations are evaluated by `JEDM2Model`
- **THEN** the model passes visual patch tokens to the action decoding head, yielding visually grounded coordinate outputs

### Requirement: Multimodal Recurrent Step Rollout Mode
The system SHALL support recurrent single-step rollout with visual observations, updating recurrent states and decoding actions with visual grounding.

#### Scenario: Recurrent step with visual observation
- **WHEN** `step(...)` is called with a token input and visual patch context
- **THEN** the model updates internal states and returns current step rationale and visually grounded action predictions
