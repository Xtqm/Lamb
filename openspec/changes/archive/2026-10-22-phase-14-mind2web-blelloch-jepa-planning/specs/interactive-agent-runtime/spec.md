# Spec Delta: interactive-agent-runtime

## ADDED Requirements

### Requirement: Active JEPA Model-Predictive Latent Rollout Planning
The agent runtime SHALL provide an active model-predictive planning mechanism utilizing the JEPA latent world model to simulate and evaluate future state representations before committing to actions.

#### Scenario: Candidate action latent rollout evaluation
- **WHEN** the agent reaches action commitment for an observation with goal embedding $z_{\text{goal}}$ and current latent state $z_t$
- **THEN** the agent predicts next latent states $\hat{z}_{t+1} = \text{Predictor}(z_t, a_i)$ for candidate action primitives $a_i \in \{\text{CLICK}, \text{TYPE}, \text{SCROLL}, \text{NAVIGATE}\}$ and computes the cosine distance $1 - \cos(\hat{z}_{t+1}, z_{\text{goal}})$ for each candidate

#### Scenario: Predictive action selection and re-ranking
- **WHEN** active JEPA planning is enabled during agent execution
- **THEN** candidate actions are scored by combining policy logits with negative predictive goal distance, selecting the action that most directly progresses toward the latent goal state

#### Scenario: Graceful fallback when JEPA model is unavailable
- **WHEN** `model.jepa` is not configured or disabled
- **THEN** the agent falls back to direct policy argmax decoding without error or latency degradation

## MODIFIED Requirements

### Requirement: Decoupled Dual-System Decision Loop
The agent runtime SHALL execute a decoupled dual-system decision loop that generates latent thought tokens (System 2) until the ActionGate logit exceeds the commitment threshold, then emits the discrete action primitive (optionally refined via JEPA latent rollout planning), continuous $(x, y) \in [0, 1]$ screen coordinates, and argument payloads (System 1).

#### Scenario: System 2 reasoning commits to System 1 action
- **WHEN** the agent processes a new environment turn observation
- **THEN** it generates reasoning thought tokens until the action gate fires or maximum thought length is reached, returning both the rationale and structured action

#### Scenario: Thought generation guardrail
- **WHEN** the action gate does not exceed threshold within a configurable maximum reasoning budget
- **THEN** the runtime forcefully triggers action commitment to prevent execution deadlock

#### Scenario: Latent planning integration within decision loop
- **WHEN** active JEPA planning is enabled and thought generation concludes
- **THEN** the decision loop runs latent action rollouts to verify or re-rank candidate action primitives before dispatching the chosen action to the environment
