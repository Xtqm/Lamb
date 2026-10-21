# Spec Delta

## MODIFIED Requirements

### Requirement: Group Relative Policy Optimization on Trajectory Outcomes
The system SHALL compute Group Relative Policy Optimization (GRPO) loss evaluating groups of sampled rollout trajectories with joint token and continuous spatial coordinate policy gradient updates, group-normalized advantages, and reference policy KL divergence penalties.

#### Scenario: GRPO advantage and policy loss evaluation
- **WHEN** a group of trajectory rollouts with outcome reward signals is evaluated
- **THEN** the system computes normalized relative advantages within the group and calculates clipped surrogate policy gradient loss with reference policy KL divergence penalty

#### Scenario: Joint token and coordinate GRPO loss evaluation
- **WHEN** grouped trajectory rollouts with token log-probabilities, continuous spatial coordinate log-probabilities, and shaped outcome rewards are evaluated
- **THEN** the system computes group-normalized advantages, evaluates the joint policy importance ratio $r_{i,t} = \exp((\log \pi_{\text{token}} - \log \pi_{\text{old\_token}}) + \lambda_{\text{coord}}(\log \pi_{\text{coord}} - \log \pi_{\text{old\_coord}}))$, clips the surrogate objective with threshold $\epsilon$, and penalizes reference policy divergence with bounded KL penalties

## ADDED Requirements

### Requirement: Dense Multi-Factor Reward Shaping
The system SHALL compute dense multi-factor trajectory rewards combining task completion bonuses, milestone achievement, coordinate overlap accuracy, step efficiency, and repetition penalties.

#### Scenario: Multi-factor reward calculation on agent rollout
- **WHEN** an interaction trajectory completes an episode
- **THEN** the system awards +2.0 for final task completion, +0.5 for each sub-goal milestone achieved, $+0.3 \times \text{IoU}$ for bounding box coordinate overlaps, penalizes execution by -0.05 per interaction turn, and applies -0.5 for duplicate consecutive actions
