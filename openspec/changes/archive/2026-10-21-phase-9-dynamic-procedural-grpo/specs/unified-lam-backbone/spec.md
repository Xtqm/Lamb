# Spec Delta

## MODIFIED Requirements

### Requirement: Autoregressive Recurrent Rollout Mode
The system SHALL support incremental token-by-token state updates via a `step` method preserving Mamba convolution and SSM state buffers as well as Multi-Head Latent Attention compressed latent KV caches ($c_{KV}, K_{\text{rope}}$) and absorbing $W_{UK}$ into $W_Q$ across all stacked layers.

#### Scenario: Multi-step recurrent rollout with state persistence
- **WHEN** single tokens `[B, 1]` or `[B]` are passed sequentially through `step` with previous recurrent state containers
- **THEN** the model updates internal states with $O(1)$ recurrent step complexity and outputs current step rationale and action predictions

#### Scenario: True MLA latent cache compression and weight absorption
- **WHEN** autoregressive generation steps are executed in Multi-Head Latent Attention
- **THEN** the KV-cache stores only the compressed latent representation $c_{KV} \in \mathbb{R}^{B, 1, d_c}$ and shared decoupled RoPE key $K_{\text{rope}} \in \mathbb{R}^{B, 1, d_R}$, absorbing $W_{UK}$ into $W_Q$ while matching sequence forward attention outputs within floating point tolerance

### Requirement: Stacked 3-Layer LAM Backbone Assembly
The system SHALL construct a unified 3-layer architecture stacking observation embeddings, three `JEDM2Block` layers, an autoregressive language modeling head, a JEPA world model receiving contextual backbone representations $h_t$, and a composite action decoding head.

#### Scenario: Full backbone sequence forward pass
- **WHEN** input token sequences `[B, S]` and optional observation/action tuples are evaluated by `JEDM2Model`
- **THEN** the model outputs LM rationale logits `[B, S, vocab_size]`, composite action predictions across sequence positions, and accumulated MoE auxiliary loss

#### Scenario: End-to-end JEPA gradient propagation from backbone representations
- **WHEN** JEPA state transition targets are evaluated during backbone forward pass
- **THEN** the model routes contextual backbone hidden states $h_t$ from the recurrent stack into `self.jepa`, allowing JEPA consistency loss gradients to flow directly into the recurrent blocks and token embeddings
