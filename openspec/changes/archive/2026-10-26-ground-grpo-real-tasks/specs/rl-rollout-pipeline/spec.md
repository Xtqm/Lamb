# Spec Delta

## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: Real-Task Programmatic Element-Matching Oracles
The system SHALL evaluate agent exploration actions during real-world web rollouts using programmatic multi-attribute matching against ground-truth targets, awarding dense spatial IoU rewards and textual typing rewards.

#### Scenario: Multi-attribute target element verification
- **WHEN** an agent emits a CLICK, TYPE, or SELECT action on an element during a real-task rollout
- **THEN** the oracle matches the target node by comparing element tag, ARIA role, textual label, and bounding box spatial overlap (IoU >= 0.5) against the ground-truth target annotation

#### Scenario: Continuous coordinate spatial proximity reward
- **WHEN** an agent predicts continuous spatial coordinates $(x, y)$ targeting a real DOM element
- **THEN** the reward function computes the 2D Gaussian proximity score relative to the ground-truth target bounding box center, penalizing off-target clicks
