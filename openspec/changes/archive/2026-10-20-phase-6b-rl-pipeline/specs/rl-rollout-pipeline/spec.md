# Spec Delta

## Purpose

Enables closed-loop reinforcement learning and preference optimization for JED-M2 by coupling live interactive rollouts in simulated environments to GRPO and DPO algorithms within a strict 6.0 GB VRAM ceiling.

## ADDED Requirements

### Requirement: Stochastic Interactive Trajectory Sampling
The system SHALL sample multi-turn interaction trajectories in simulated Web/OS environments using temperature-scaled stochastic action and thought token generation.

#### Scenario: Temperature-based stochastic rollout execution
- **WHEN** the rollout sampler executes an episode for an environment scenario with sampling temperature $T > 0$
- **THEN** the agent generates exploration actions and captures token sequences, log-probabilities, step rewards, coordinates, and episode termination status

### Requirement: Group Rollout Collection for GRPO
The system SHALL collect groups of $G$ rollouts per task goal with outcome reward assignment for Group Relative Policy Optimization.

#### Scenario: Multi-rollout group generation and advantage normalization
- **WHEN** a group of $G$ rollouts is sampled for a target scenario
- **THEN** exactly $G$ trajectories are recorded with terminal outcome rewards, and group-normalized relative advantages are computed across the group

### Requirement: Preference Trajectory Pairing for DPO
The system SHALL construct paired preference datasets by matching successful (winning) and failed (losing) trajectories for the same task goal.

#### Scenario: Winning and losing trajectory pairing
- **WHEN** a collection of completed rollouts contains both successful and unsuccessful trajectories
- **THEN** the system generates paired chosen and rejected trajectory tuples formatted for Direct Policy Optimization

### Requirement: Online Policy Optimization with Reference KL Constraint
The system SHALL execute online GRPO policy updates against rollout trajectories using clipped surrogate objectives and reference model KL penalties within a 6.0 GB VRAM ceiling.

#### Scenario: Online policy gradient update and telemetry logging
- **WHEN** an online training iteration evaluates grouped rollouts against active and frozen reference models
- **THEN** active policy weights are updated with bounded KL divergence, reporting average reward, advantage variance, KL divergence, and token throughput while remaining below 6.0 GB VRAM
