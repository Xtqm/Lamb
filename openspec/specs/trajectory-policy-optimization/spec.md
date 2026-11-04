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
The system SHALL compute Group Relative Policy Optimization (GRPO) loss evaluating groups of sampled rollout trajectories with joint token and continuous spatial coordinate policy gradient updates, group-normalized advantages, and reference policy KL divergence penalties.

#### Scenario: GRPO advantage and policy loss evaluation
- **WHEN** a group of trajectory rollouts with outcome reward signals is evaluated
- **THEN** the system computes normalized relative advantages within the group and calculates clipped surrogate policy gradient loss with reference policy KL divergence penalty

#### Scenario: Joint token and coordinate GRPO loss evaluation
- **WHEN** grouped trajectory rollouts with token log-probabilities, continuous spatial coordinate log-probabilities, and shaped outcome rewards are evaluated
- **THEN** the system computes group-normalized advantages, evaluates the joint policy importance ratio $r_{i,t} = \exp((\log \pi_{\text{token}} - \log \pi_{\text{old\_token}}) + \lambda_{\text{coord}}(\log \pi_{\text{coord}} - \log \pi_{\text{old\_coord}}))$, clips the surrogate objective with threshold $\epsilon$, and penalizes reference policy divergence with bounded KL penalties

### Requirement: Dense Multi-Factor Reward Shaping
The system SHALL compute dense multi-factor trajectory rewards combining task completion bonuses, milestone achievement, coordinate overlap accuracy, step efficiency, and repetition penalties.

#### Scenario: Multi-factor reward calculation on agent rollout
- **WHEN** an interaction trajectory completes an episode
- **THEN** the system awards +2.0 for final task completion, +0.5 for each sub-goal milestone achieved, $+0.3 \times \text{IoU}$ for bounding box coordinate overlaps, penalizes execution by -0.05 per interaction turn, and applies -0.5 for duplicate consecutive actions

### Requirement: Modular Curriculum and Exploration Schedulers
The policy optimization runtime SHALL provide modular curriculum and exploration schedulers (`CurriculumScheduler` and `ExplorationScheduler`) exported from `jed_m2.rl` and `jed_m2` to regulate task complexity tiers and action entropy throughout reinforcement learning.

#### Scenario: Curriculum tier progression
- **WHEN** multi-turn RL rollout episodes are recorded with success or failure outcomes
- **THEN** `CurriculumScheduler` tracks moving success rates over a sliding window and advances from Level 1 (atomic) to Level 2 (compound) and Level 3 (long-horizon) upon exceeding the configured success threshold

#### Scenario: Exploration temperature and entropy decay
- **WHEN** exploration step intervals elapse during RL policy optimization
- **THEN** `ExplorationScheduler` anneals sampling temperature and action entropy regularization coefficients toward configured minimum bounds

#### Scenario: Library export and script interoperability
- **WHEN** `CurriculumScheduler` and `ExplorationScheduler` are imported from `jed_m2` or `jed_m2.rl`
- **THEN** they provide standalone, framework-agnostic step and outcome recording interfaces compatible with `train_rl.py` and evaluation suites
