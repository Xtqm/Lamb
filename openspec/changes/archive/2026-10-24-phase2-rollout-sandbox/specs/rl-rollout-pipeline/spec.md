# Spec Delta: rl-rollout-pipeline

## MODIFIED Requirements

### Requirement: Stochastic Interactive Trajectory Sampling
The system SHALL sample multi-turn interaction trajectories in static, procedural, or live browser environments by prefixing each episode with the task goal, ingesting structured interactive DOM element observations without artificial token truncation up to the sequence limit, and generating temperature-scaled actions alongside continuous Gaussian spatial coordinates.

#### Scenario: Temperature-based stochastic rollout execution
- **WHEN** the rollout sampler executes an episode for an environment scenario with sampling temperature $T > 0$
- **THEN** the agent generates exploration actions and captures token sequences, log-probabilities, step rewards, coordinates, and episode termination status

#### Scenario: Continuous coordinate and token stochastic rollout execution
- **WHEN** the rollout sampler executes an episode in a procedural or benchmark environment with exploration temperature $T > 0$
- **THEN** the agent generates exploration actions, samples continuous coordinates from a Gaussian spatial distribution centered at predicted coordinates, and records token sequences, continuous coordinate log-probabilities, step rewards, and episode termination status

#### Scenario: Goal-conditioned trajectory initialization
- **WHEN** an episode begins in any environment backend
- **THEN** the rollout sampler prepends `Goal: {goal}` to the token sequence before the initial observation, matching the pretraining token format

#### Scenario: High-fidelity DOM observation ingestion
- **WHEN** a web observation containing structured interactive elements is encoded
- **THEN** the rollout sampler ingests up to the full observation sequence budget (at least 512 tokens) without arbitrarily truncating observations to 24 tokens
