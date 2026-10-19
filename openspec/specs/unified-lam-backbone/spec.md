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
The system SHALL construct a unified 3-layer architecture stacking observation embeddings, three `JEDM2Block` layers, an autoregressive language modeling head, a JEPA world model, and a composite action decoding head.

#### Scenario: Full backbone sequence forward pass
- **WHEN** input token sequences `[B, S]` and optional observation/action tuples are evaluated by `JEDM2Model`
- **THEN** the model outputs LM rationale logits `[B, S, vocab_size]`, composite action predictions across sequence positions, and accumulated MoE auxiliary loss

### Requirement: Autoregressive Recurrent Rollout Mode
The system SHALL support incremental token-by-token state updates via a `step` method preserving Mamba convolution and SSM state buffers as well as Multi-Head Latent Attention KV caches across all stacked layers.

#### Scenario: Multi-step recurrent rollout with state persistence
- **WHEN** single tokens `[B, 1]` or `[B]` are passed sequentially through `step` with previous recurrent state containers
- **THEN** the model updates internal states with $O(1)$ recurrent step complexity and outputs current step rationale and action predictions

### Requirement: Parameter Scaling and Memory Budget Compliance
The system SHALL scale within ~45M total parameters and ~14M active parameters per token, executing verification forward and backward passes under 2.0 GB peak VRAM on CUDA in `bfloat16`.

#### Scenario: Hardware envelope verification
- **WHEN** the full `JEDM2Model` executes sequence forward and backward passes with sequence length 1024 on CUDA in `bfloat16`
- **THEN** total parameters remain within $45\text{M} \pm 5\text{M}$, active parameters remain within $14\text{M} \pm 3\text{M}$, peak VRAM remains below 2.0 GB, and all gradients are free of NaNs or Infinities
