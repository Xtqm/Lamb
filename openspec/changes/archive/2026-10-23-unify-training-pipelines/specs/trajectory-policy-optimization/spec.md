# Spec Delta: trajectory-policy-optimization

## ADDED Requirements

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
