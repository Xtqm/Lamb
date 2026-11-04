# Spec Delta: rl-rollout-pipeline

## MODIFIED Requirements

### Requirement: Online Policy Optimization with Reference KL Constraint
The system SHALL execute online GRPO policy updates against rollout trajectories using clipped surrogate objectives, continuous spatial coordinate scaling via `lambda_coord`, and reference model KL penalties within a 6.0 GB VRAM ceiling.

#### Scenario: Online policy gradient update and telemetry logging
- **WHEN** an online training iteration evaluates grouped rollouts against active and frozen reference models
- **THEN** active policy weights are updated with bounded KL divergence, reporting average reward, advantage variance, KL divergence, and token throughput while remaining below 6.0 GB VRAM

#### Scenario: Online browser GRPO rollout execution
- **WHEN** online GRPO rollouts run with `--env_type browser`
- **THEN** groups of $G=4$ rollouts execute in headless Chromium across benchmark scenarios, computing shaped rewards, group advantages, and policy gradient updates
