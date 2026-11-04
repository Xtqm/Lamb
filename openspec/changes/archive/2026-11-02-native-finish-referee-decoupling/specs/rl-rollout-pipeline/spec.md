# Spec Delta: rl-rollout-pipeline

## MODIFIED Requirements

### Requirement: Stochastic Interactive Trajectory Sampling
The system SHALL sample multi-turn interaction trajectories in simulated or live browser environments by prefixing each episode with the task goal, ingesting structured interactive DOM observations up to the sequence limit, executing macro-action input clearing and render settling, generating temperature-scaled actions with continuous Gaussian coordinates, and terminating episodes autonomously when the agent emits FINISH or exhausts the maximum step budget.

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
- **WHEN** the agent reaches the target goal state or terminates an episode
- **THEN** the completion referee acts as a post-hoc evaluation oracle to validate milestone achievement and award terminal completion rewards without overriding in-the-loop actions

#### Scenario: Autonomous agent task termination
- **WHEN** the agent predicts an `ActionType.FINISH` primitive or reaches the maximum step budget
- **THEN** the rollout episode terminates, and an external oracle scores whether the goal was satisfied to compute post-hoc terminal rewards or premature finish penalties without intervening in the live decision loop

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
- **THEN** rewards are computed using decoupled components: action primitive type match, candidate-snapped or continuous point-in-box element hit, textual argument alignment, a positive terminal bonus for valid autonomous FINISH actions, and a negative penalty for premature FINISH emissions on unsatisfied states
