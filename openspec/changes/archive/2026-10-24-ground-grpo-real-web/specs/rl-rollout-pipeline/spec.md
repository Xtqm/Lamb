# Spec Delta: rl-rollout-pipeline

## MODIFIED Requirements

### Requirement: Stochastic Interactive Trajectory Sampling
The system SHALL sample multi-turn interaction trajectories in static, procedural, or live browser environments by prefixing each episode with the task goal, ingesting structured interactive DOM element observations without artificial token truncation up to the sequence limit, and generating temperature-scaled actions alongside continuous Gaussian spatial coordinates.

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

### Requirement: Headless Interactive Browser Sandbox Rollout Execution
The rollout pipeline SHALL execute interactive multi-turn exploration episodes within headless browser environments using normalized spatial coordinates, candidate DOM element parsing, and native primitive dispatch.

#### Scenario: Browser rollout episode execution
- **WHEN** the rollout sampler executes an episode configured with a browser environment
- **THEN** the agent generates exploration actions, executes native browser primitives, and records observation tokens, coordinate log-probabilities, and oracle rewards

#### Scenario: Oracle reward integration in browser rollouts
- **WHEN** an episode progresses in the browser environment
- **THEN** the sampler records milestone, spatial IoU, and terminal rewards evaluated by programmatic oracles and computes cumulative trajectory returns

#### Scenario: Live DOM candidate element tokenization
- **WHEN** an interactive webpage is loaded in the browser environment
- **THEN** candidate interactive elements extracted from the live DOM are formatted with bounding boxes and ingested into the agent observation

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
