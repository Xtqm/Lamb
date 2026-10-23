# rl-rollout-pipeline Specification

## Purpose

Enables closed-loop reinforcement learning and preference optimization for JED-M2 by coupling live interactive rollouts in simulated environments to GRPO and DPO algorithms within a strict 6.0 GB VRAM ceiling.

## Requirements

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

### Requirement: Curriculum Learning and Exploration Scheduling
The system SHALL schedule training curriculum tiers and adaptively regulate policy entropy and sampling temperature during reinforcement learning.

#### Scenario: Multi-tier curriculum progression
- **WHEN** an online RL training session runs with curriculum enabled and the moving window success rate exceeds 80% on the active tier
- **THEN** the environment sampler automatically transitions progression from Level 1 (single-step atomic actions) to Level 2 (2-3 step workflows) to Level 3 (5-8 step long-horizon workflows)

#### Scenario: Adaptive entropy and temperature scheduling
- **WHEN** policy entropy drops below a minimum threshold or training steps progress
- **THEN** the exploration scheduler adjusts temperature and applies an entropy regularization bonus to prevent policy mode collapse

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

