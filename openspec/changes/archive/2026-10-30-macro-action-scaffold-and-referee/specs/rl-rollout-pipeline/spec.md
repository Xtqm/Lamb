# Spec Delta: rl-rollout-pipeline

## MODIFIED Requirements

### Requirement: Stochastic Interactive Trajectory Sampling
The system SHALL sample multi-turn interaction trajectories in static, procedural, or live browser environments by prefixing each episode with the task goal, ingesting structured interactive DOM element observations without artificial token truncation up to the sequence limit, executing macro-action input clearing and render settling, checking programmatic completion referees to eliminate infinite truncation, and generating temperature-scaled actions alongside continuous Gaussian spatial coordinates.

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

#### Scenario: Referee-assisted task termination
- **WHEN** the agent reaches the target goal state across any rollout turn
- **THEN** the completion referee detects milestone completion, emits or validates the FINISH transition, terminates the rollout successfully, and awards the terminal completion reward instead of truncating at maximum steps

#### Scenario: Macro-action input clearing and render settling
- **WHEN** the agent dispatches a TYPE action or navigates to a new page
- **THEN** the execution scaffold automatically clears pre-existing input text and polls until DOM mutations settle before capturing the next observation

### Requirement: Dual-Modality GRPO Verification and Optimization
The system SHALL compute joint policy gradient updates across both discrete token policy heads and continuous spatial Gaussian coordinate heads weighted by coordinate factor `lambda_coord`, optimizing with a decoupled hierarchical reward function rewarding candidate element hits, argument matches, and completion bonuses.

#### Scenario: Dual-modality policy gradient update
- **WHEN** a group of browser rollouts is evaluated by GRPOTrainer
- **THEN** the surrogate loss computes token log-probability ratios and continuous Gaussian coordinate log-probability ratios, applying `lambda_coord` to coordinate gradients

#### Scenario: Group-normalized coordinate advantage weighting
- **WHEN** coordinate log-probabilities are updated during GRPO
- **THEN** relative advantages normalized across group $G$ scale the spatial coordinate gradients identically to token rationale gradients

#### Scenario: Decoupled hierarchical reward computation
- **WHEN** rollout rewards are collated for GRPO advantage computation
- **THEN** rewards are computed using decoupled components: action primitive type match, candidate-snapped or continuous point-in-box element hit, textual argument alignment, and referee-verified terminal completion bonus
