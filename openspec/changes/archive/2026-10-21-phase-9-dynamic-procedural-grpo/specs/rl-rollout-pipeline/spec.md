# Spec Delta

## MODIFIED Requirements

### Requirement: Stochastic Interactive Trajectory Sampling
The system SHALL sample multi-turn interaction trajectories in static or procedural simulated Web/OS environments using temperature-scaled stochastic action and thought token generation alongside continuous Gaussian spatial coordinate sampling and log-probability tracking.

#### Scenario: Temperature-based stochastic rollout execution
- **WHEN** the rollout sampler executes an episode for an environment scenario with sampling temperature $T > 0$
- **THEN** the agent generates exploration actions and captures token sequences, log-probabilities, step rewards, coordinates, and episode termination status

#### Scenario: Continuous coordinate and token stochastic rollout execution
- **WHEN** the rollout sampler executes an episode in a procedural or benchmark environment with exploration temperature $T > 0$
- **THEN** the agent generates exploration actions, samples continuous coordinates from a Gaussian spatial distribution centered at predicted coordinates, and records token sequences, continuous coordinate log-probabilities, step rewards, and episode termination status

### Requirement: Group Rollout Collection for GRPO
The system SHALL collect groups of $G$ rollouts per task goal with joint token and continuous spatial coordinate log-probabilities for Group Relative Policy Optimization.

#### Scenario: Multi-rollout group generation and advantage normalization
- **WHEN** a group of $G$ rollouts is sampled for a target scenario
- **THEN** exactly $G$ trajectories are recorded with terminal outcome rewards, and group-normalized relative advantages are computed across the group

#### Scenario: Joint token and coordinate rollout batch collation
- **WHEN** a group of $G$ rollouts is sampled and collated for GRPO
- **THEN** exactly $G$ trajectories are batched into padded token IDs, loss masks, old token log-probabilities, old coordinate log-probabilities, target coordinate vectors, and group-normalized relative advantages

## ADDED Requirements

### Requirement: Curriculum Learning and Exploration Scheduling
The system SHALL schedule training curriculum tiers and adaptively regulate policy entropy and sampling temperature during reinforcement learning.

#### Scenario: Multi-tier curriculum progression
- **WHEN** an online RL training session runs with curriculum enabled and the moving window success rate exceeds 80% on the active tier
- **THEN** the environment sampler automatically transitions progression from Level 1 (single-step atomic actions) to Level 2 (2-3 step workflows) to Level 3 (5-8 step long-horizon workflows)

#### Scenario: Adaptive entropy and temperature scheduling
- **WHEN** policy entropy drops below a minimum threshold or training steps progress
- **THEN** the exploration scheduler adjusts temperature and applies an entropy regularization bonus to prevent policy mode collapse
