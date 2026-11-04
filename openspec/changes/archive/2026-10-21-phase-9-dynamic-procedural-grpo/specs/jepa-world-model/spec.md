# Spec Delta

## MODIFIED Requirements

### Requirement: Latent State Encoding
The latent state encoder SHALL map digital observation features or contextual backbone hidden state representations $h_t$ into an abstract latent vector space $z_t \in \mathbb{R}^{B \times D_{latent}}$.

#### Scenario: Observation encoding forward pass
- **WHEN** observation vectors of shape `[B, D_obs]` are passed to the online encoder
- **THEN** the encoder outputs normalized latent representations of shape `[B, D_latent]`

#### Scenario: Contextual backbone representation encoding
- **WHEN** contextual backbone hidden states $h_t$ of shape `[B, hidden_dim]` from the recurrent block stack are passed to the online encoder
- **THEN** the encoder projects the representations into normalized latent vectors $z_t$ of shape `[B, D_{latent}]$ with full gradient backpropagation into the backbone
