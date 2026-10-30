# rl-rollout-pipeline Specification

## Purpose

Enables closed-loop reinforcement learning and preference optimization for JED-M2 by coupling live interactive rollouts in simulated environments to GRPO and DPO algorithms within a strict 6.0 GB VRAM ceiling.

## Requirements

### Requirement: Stochastic Interactive Trajectory Sampling
The system SHALL sample multi-turn interaction trajectories in static, procedural, or live browser environments by prefixing each episode with the task goal, ingesting structured interactive DOM element observations without artificial token truncation up to the sequence limit, executing macro-action input clearing and render settling, checking programmatic completion referees to eliminate infinite truncation, and generating temperature-scaled actions alongside continuous Gaussian spatial coordinates.

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
- **WHEN** the agent reaches the target goal state across any rollout turn
- **THEN** the completion referee detects milestone completion, emits or validates the FINISH transition, terminates the rollout successfully, and awards the terminal completion reward instead of truncating at maximum steps

#### Scenario: Macro-action input clearing and render settling
- **WHEN** the agent dispatches a TYPE action or navigates to a new page
- **THEN** the execution scaffold automatically clears pre-existing input text and polls until DOM mutations settle before capturing the next observation

### Requirement: Group Rollout Collection for GRPO
The system SHALL collect groups of $G$ rollouts per task goal with joint token and continuous spatial coordinate log-probabilities for Group Relative Policy Optimization across diverse real-world web tasks.

#### Scenario: Multi-rollout group generation and advantage normalization
- **WHEN** a group of $G$ rollouts is sampled for a target scenario or real-world web task
- **THEN** exactly $G$ trajectories are recorded with terminal outcome rewards, and group-normalized relative advantages are computed across the group

#### Scenario: Joint token and coordinate rollout batch collation
- **WHEN** a group of $G$ rollouts is sampled and collated for GRPO
- **THEN** exactly $G$ trajectories are batched into padded token IDs, loss masks, old token log-probabilities, old coordinate log-probabilities, target coordinate vectors, and group-normalized relative advantages

#### Scenario: Real-task goal sampling during group rollouts
- **WHEN** group rollouts are sampled during online GRPO
- **THEN** each group of $G$ rollouts is initialized with an identical goal and initial DOM state drawn from the real-world web task dataset (Mind2Web and WebArena)

### Requirement: Preference Trajectory Pairing for DPO
The system SHALL construct paired preference datasets by matching successful (winning) and failed (losing) trajectories for the same task goal.

#### Scenario: Winning and losing trajectory pairing
- **WHEN** a collection of completed rollouts contains both successful and unsuccessful trajectories
- **THEN** the system generates paired chosen and rejected trajectory tuples formatted for Direct Policy Optimization

### Requirement: Online Policy Optimization with Reference KL Constraint
The system SHALL execute online GRPO policy updates against rollout trajectories using clipped surrogate objectives, continuous spatial coordinate scaling via `lambda_coord`, and reference model KL penalties within a 6.0 GB VRAM ceiling.

#### Scenario: Online policy gradient update and telemetry logging
- **WHEN** an online training iteration evaluates grouped rollouts against active and frozen reference models
- **THEN** active policy weights are updated with bounded KL divergence, reporting average reward, advantage variance, KL divergence, and token throughput while remaining below 6.0 GB VRAM

#### Scenario: Online browser GRPO rollout execution
- **WHEN** online GRPO rollouts run with `--env_type browser`
- **THEN** groups of $G=4$ rollouts execute in headless Chromium across benchmark scenarios, computing shaped rewards, group advantages, and policy gradient updates

### Requirement: Curriculum Learning and Exploration Scheduling
The system SHALL schedule training curriculum tiers and adaptively regulate policy entropy and sampling temperature during reinforcement learning.

#### Scenario: Multi-tier curriculum progression
- **WHEN** an online RL training session runs with curriculum enabled and the moving window success rate exceeds 80% on the active tier
- **THEN** the environment sampler automatically transitions progression from Level 1 (single-step atomic actions) to Level 2 (2-3 step workflows) to Level 3 (5-8 step long-horizon workflows)

#### Scenario: Adaptive entropy and temperature scheduling
- **WHEN** policy entropy drops below a minimum threshold or training steps progress
- **THEN** the exploration scheduler adjusts temperature and applies an entropy regularization bonus to prevent policy mode collapse

### Requirement: Headless Interactive Browser Sandbox Rollout Execution
The rollout pipeline SHALL execute interactive multi-turn exploration episodes within headless browser environments using normalized spatial coordinates, candidate DOM element parsing, and native primitive dispatch across synthetic fixtures and real-world DOM snapshots.

#### Scenario: Browser rollout episode execution
- **WHEN** the rollout sampler executes an episode configured with a browser environment
- **THEN** the agent generates exploration actions, executes native browser primitives, and records observation tokens, coordinate log-probabilities, and oracle rewards

#### Scenario: Oracle reward integration in browser rollouts
- **WHEN** an episode progresses in the browser environment
- **THEN** the sampler records milestone, spatial IoU, and terminal rewards evaluated by programmatic oracles and computes cumulative trajectory returns

#### Scenario: Live DOM candidate element tokenization
- **WHEN** an interactive webpage is loaded in the browser environment
- **THEN** candidate interactive elements extracted from the live DOM are formatted with bounding boxes and ingested into the agent observation

#### Scenario: Real-world DOM replay rollout execution
- **WHEN** the rollout sampler runs against offline real-world task traces from Mind2Web or WebArena
- **THEN** the environment injects the recorded HTML DOM snapshot, parses interactive element bounding boxes, and executes agent actions against matching DOM nodes

### Requirement: Real-Task Programmatic Element-Matching Oracles
The system SHALL evaluate agent exploration actions during real-world web rollouts using programmatic multi-attribute matching against ground-truth targets, awarding dense spatial IoU rewards and textual typing rewards.

#### Scenario: Multi-attribute target element verification
- **WHEN** an agent emits a CLICK, TYPE, or SELECT action on an element during a real-task rollout
- **THEN** the oracle matches the target node by comparing element tag, ARIA role, textual label, and bounding box spatial overlap (IoU >= 0.5) against the ground-truth target annotation

#### Scenario: Continuous coordinate spatial proximity reward
- **WHEN** an agent predicts continuous spatial coordinates $(x, y) targeting a real DOM element
- **THEN** the reward function computes the 2D Gaussian proximity score relative to the ground-truth target bounding box center, penalizing off-target clicks

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
- **THEN** rewards are computed using decoupled components: action primitive type match, candidate-snapped or continuous point-in-box element hit, textual argument alignment, and referee-verified terminal completion bonus

### Requirement: Offline Real-World DPO Pipeline Integration
The system SHALL support Direct Policy Optimization directly from offline real-world web preference datasets without requiring synthetic procedural mock rollouts, serving as the required calibration phase prior to online browser GRPO.

#### Scenario: Offline real web preference training step
- **WHEN** `train_rl.py` is executed with `--mode dpo --dataset real_mind2web`
- **THEN** the trainer loads paired Mind2Web trajectories $(\tau_w, \tau_l)$ and executes policy updates using `calculate_dpo_loss()`

#### Scenario: VRAM ceiling enforcement during offline DPO
- **WHEN** offline DPO batches are processed with bfloat16 precision and gradient checkpointing
- **THEN** peak memory consumption remains strictly under 6.0 GB VRAM on NVIDIA RTX 5060 Laptop GPU

#### Scenario: Full-corpus preference pair generation
- **WHEN** preference pairs are constructed from the expanded Mind2Web dataset
- **THEN** winning expert demonstrations are paired with negative spatial coordinate perturbations ($|\Delta x| > 0.15, |\Delta y| > 0.15$) and sub-optimal action primitive substitutions across all 1,009 pretraining tasks
