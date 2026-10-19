# trajectory-policy-optimization Specification

## Purpose
Provides policy optimization scaffolding for the JED-M2 Large Action Model supporting Direct Policy Optimization (DPO) and Group Relative Policy Optimization (GRPO) on multi-turn trajectory outcome rewards.

## Requirements

### Requirement: Direct Policy Optimization on Trajectory Pairs
The system SHALL compute Direct Policy Optimization (DPO) loss comparing chosen vs rejected multi-turn agent interaction trajectories using implicit reward estimation against a reference policy.

#### Scenario: DPO loss computation on trajectory pairs
- **WHEN** paired winning and losing trajectories are evaluated against active and reference model log-probabilities
- **THEN** the system computes bounded DPO loss with scaling temperature beta and reports margin metrics

### Requirement: Group Relative Policy Optimization on Trajectory Outcomes
The system SHALL compute Group Relative Policy Optimization (GRPO) loss evaluating groups of sampled rollout trajectories for a task with normalized group-relative advantages and token-level KL penalties against a reference policy.

#### Scenario: GRPO advantage and policy loss evaluation
- **WHEN** a group of trajectory rollouts with outcome reward signals is evaluated
- **THEN** the system computes normalized relative advantages within the group and calculates clipped surrogate policy gradient loss with reference policy KL divergence penalty
