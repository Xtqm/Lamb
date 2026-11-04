# jepa-world-model Specification

## Purpose
Provides a Joint-Embedding Predictive Architecture (JEPA) world model for encoding observations into abstract latent state representations, tracking momentum target representations, predicting transition states conditioned on actions, and measuring latent predictive consistency.

## Requirements

### Requirement: Latent State Encoding
The latent state encoder SHALL map digital observation features or contextual backbone hidden state representations $h_t$ into an abstract latent vector space $z_t \in \mathbb{R}^{B \times D_{latent}}$.

#### Scenario: Observation encoding forward pass
- **WHEN** observation vectors of shape `[B, D_obs]` are passed to the online encoder
- **THEN** the encoder outputs normalized latent representations of shape `[B, D_latent]`

#### Scenario: Contextual backbone representation encoding
- **WHEN** contextual backbone hidden states $h_t$ of shape `[B, hidden_dim]` from the recurrent block stack are passed to the online encoder
- **THEN** the encoder projects the representations into normalized latent vectors $z_t$ of shape `[B, D_{latent}]$ with full gradient backpropagation into the backbone

### Requirement: EMA Target Network Updates
The target encoder network SHALL maintain an exponential moving average (EMA) of the online state encoder parameters with momentum decay $\tau \in [0, 1)$ and execute forward passes with strictly detached gradients.

#### Scenario: Target encoding without gradient flow
- **WHEN** target observations of shape `[B, D_obs]` are passed to the EMA target encoder
- **THEN** the target encoder computes target representations $z_{t+1}$ with detached gradients (`requires_grad=False`)

#### Scenario: Momentum parameter update
- **WHEN** `update_target_network()` is invoked after an optimization step
- **THEN** target parameters $\theta_{target}$ are updated according to $\theta_{target} \leftarrow \tau \theta_{target} + (1 - \tau) \theta_{online}$

### Requirement: Action-Conditioned Latent Transition Prediction
The latent transition predictor SHALL take a current latent state $z_t$ and an action embedding $a_t$ and predict the expected next latent state $\hat{z}_{t+1}$.

#### Scenario: Next latent state prediction
- **WHEN** current latent state `[B, D_latent]` and action representation `[B, D_action]` are provided to the predictor
- **THEN** the predictor outputs predicted next state `[B, D_latent]` matching the latent space dimension

### Requirement: Composite Latent Consistency Loss
The JEPA world model SHALL compute a composite consistency loss between predicted latent states and stop-gradient target representations combining Smooth L1 (Huber) loss and cosine similarity distance.

#### Scenario: Consistency loss calculation
- **WHEN** predicted states $\hat{z}_{t+1}$ and target states $z_{t+1}$ are provided to the loss calculation function
- **THEN** the module returns a non-negative scalar loss with non-zero gradients flowing exclusively to the online encoder and predictor

### Requirement: Multimodal Observation Encoding in JEPA
The JEPA world model SHALL accept multimodal observations composed of combined visual patch features and text/DOM features, mapping them to abstract latent representations $z_t \in \mathbb{R}^{B \times D_{latent}}$ in both online and target encoders.

#### Scenario: Multimodal observation state encoding
- **WHEN** multimodal observation inputs combining visual patch features and DOM features are provided to `JEPAWorldModel`
- **THEN** the online encoder maps the combined inputs into normalized latent vectors $z_t$ of shape `[B, D_latent]`

#### Scenario: Multimodal target encoding and transition prediction
- **WHEN** multimodal state transitions $(s_t, a_t, s_{t+1})$ with visual representations are evaluated
- **THEN** the predictor predicts next latent state $\hat{z}_{t+1}$, the target encoder computes $z_{t+1}$ with detached gradients, and composite consistency loss is returned
