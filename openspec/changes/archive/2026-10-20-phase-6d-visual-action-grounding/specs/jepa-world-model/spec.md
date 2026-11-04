# Spec Delta

## ADDED Requirements

### Requirement: Multimodal Observation Encoding in JEPA
The JEPA world model SHALL accept multimodal observations composed of combined visual patch features and text/DOM features, mapping them to abstract latent representations $z_t \in \mathbb{R}^{B \times D_{latent}}$ in both online and target encoders.

#### Scenario: Multimodal observation state encoding
- **WHEN** multimodal observation inputs combining visual patch features and DOM features are provided to `JEPAWorldModel`
- **THEN** the online encoder maps the combined inputs into normalized latent vectors $z_t$ of shape `[B, D_latent]`

#### Scenario: Multimodal target encoding and transition prediction
- **WHEN** multimodal state transitions $(s_t, a_t, s_{t+1})$ with visual representations are evaluated
- **THEN** the predictor predicts next latent state $\hat{z}_{t+1}$, the target encoder computes $z_{t+1}$ with detached gradients, and composite consistency loss is returned
