# Spec Delta: rl-rollout-pipeline

## ADDED Requirements

### Requirement: Headless Interactive Browser Sandbox Rollout Execution
The rollout pipeline SHALL execute interactive multi-turn exploration episodes within headless browser environments using normalized spatial coordinates and native primitive dispatch.

#### Scenario: Browser rollout episode execution
- **WHEN** the rollout sampler executes an episode configured with a browser environment
- **THEN** the agent generates exploration actions, executes native browser primitives, and records observation tokens, coordinate log-probabilities, and oracle rewards

#### Scenario: Oracle reward integration in browser rollouts
- **WHEN** an episode progresses in the browser environment
- **THEN** the sampler records milestone and terminal rewards evaluated by programmatic oracles and computes cumulative trajectory returns

### Requirement: Dual-Modality GRPO Verification and Optimization
The system SHALL compute joint policy gradient updates across both discrete token policy heads and continuous spatial Gaussian coordinate heads weighted by coordinate factor `lambda_coord`.

#### Scenario: Dual-modality policy gradient update
- **WHEN** a group of browser rollouts is evaluated by GRPOTrainer
- **THEN** the surrogate loss computes token log-probability ratios and continuous Gaussian coordinate log-probability ratios, applying `lambda_coord` to coordinate gradients

#### Scenario: Group-normalized coordinate advantage weighting
- **WHEN** coordinate log-probabilities are updated during GRPO
- **THEN** relative advantages normalized across group $G$ scale the spatial coordinate gradients identically to token rationale gradients

### Requirement: Offline Real-World DPO Pipeline Integration
The system SHALL support Direct Policy Optimization directly from offline real-world web preference datasets without requiring synthetic procedural mock rollouts.

#### Scenario: Offline real web preference training step
- **WHEN** `train_rl.py` is executed with `--mode dpo --dataset real_mind2web`
- **THEN** the trainer loads paired Mind2Web trajectories $(\tau_w, \tau_l)$ and executes policy updates using `calculate_dpo_loss()`

#### Scenario: VRAM ceiling enforcement during offline DPO
- **WHEN** offline DPO batches are processed with bfloat16 precision and gradient checkpointing
- **THEN** peak memory consumption remains strictly under 6.0 GB VRAM on NVIDIA RTX 5060 Laptop GPU
