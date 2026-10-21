# unified-lam-backbone Specification

## Purpose

Provides a unified Large Action Model backbone that stacks interleaved Mamba SSM recurrence, Multi-Head Latent Attention, and Sparse Top-2 Mixture of Experts over shared observation embeddings, JEPA latent world model, and dual-system action decoding heads.

## Requirements

### Requirement: Interleaved Recurrent JEDM2Block Execution
The system SHALL combine Mamba state space modeling, Multi-Head Latent Attention (MLA), and Sparse Top-2 MoE with pre-RMSNorm and residual connections into a unified recurrent block.

#### Scenario: Forward transformation through recurrent block
- **WHEN** a tensor of shape `[B, S, D]` is passed through `JEDM2Block`
- **THEN** the block returns an output tensor of shape `[B, S, D]` along with the auxiliary routing loss accumulated from the Sparse Top-2 MoE

### Requirement: Stacked 3-Layer LAM Backbone Assembly
The system SHALL construct a unified 3-layer architecture stacking observation embeddings, three `JEDM2Block` layers, an autoregressive language modeling head, a JEPA world model receiving contextual backbone representations $h_t$, and a composite action decoding head.

#### Scenario: Full backbone sequence forward pass
- **WHEN** input token sequences `[B, S]` and optional observation/action tuples are evaluated by `JEDM2Model`
- **THEN** the model outputs LM rationale logits `[B, S, vocab_size]`, composite action predictions across sequence positions, and accumulated MoE auxiliary loss

#### Scenario: End-to-end JEPA gradient propagation from backbone representations
- **WHEN** JEPA state transition targets are evaluated during backbone forward pass
- **THEN** the model routes contextual backbone hidden states $h_t$ from the recurrent stack into `self.jepa`, allowing JEPA consistency loss gradients to flow directly into the recurrent blocks and token embeddings

### Requirement: Autoregressive Recurrent Rollout Mode
The system SHALL support incremental token-by-token state updates via a `step` method preserving Mamba convolution and SSM state buffers as well as Multi-Head Latent Attention compressed latent KV caches ($c_{KV}, K_{\text{rope}}$) and absorbing $W_{UK}$ into $W_Q$ across all stacked layers.

#### Scenario: Multi-step recurrent rollout with state persistence
- **WHEN** single tokens `[B, 1]` or `[B]` are passed sequentially through `step` with previous recurrent state containers
- **THEN** the model updates internal states with $O(1)$ recurrent step complexity and outputs current step rationale and action predictions

#### Scenario: True MLA latent cache compression and weight absorption
- **WHEN** autoregressive generation steps are executed in Multi-Head Latent Attention
- **THEN** the KV-cache stores only the compressed latent representation $c_{KV} \in \mathbb{R}^{B, 1, d_c}$ and shared decoupled RoPE key $K_{\text{rope}} \in \mathbb{R}^{B, 1, d_R}$, absorbing $W_{UK}$ into $W_Q$ while matching sequence forward attention outputs within floating point tolerance

### Requirement: Parameter Scaling and Memory Budget Compliance
The system SHALL scale within ~45M total parameters and ~14M active parameters per token, executing verification forward and backward passes under 2.0 GB peak VRAM on CUDA in `bfloat16`.

#### Scenario: Hardware envelope verification
- **WHEN** the full `JEDM2Model` executes sequence forward and backward passes with sequence length 1024 on CUDA in `bfloat16`
- **THEN** total parameters remain within $45\text{M} \pm 5\text{M}$, active parameters remain within $14\text{M} \pm 3\text{M}$, peak VRAM remains below 2.0 GB, and all gradients are free of NaNs or Infinities

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
