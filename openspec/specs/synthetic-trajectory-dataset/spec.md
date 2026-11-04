# synthetic-trajectory-dataset Specification

## Purpose
Provides a synthetic multi-turn agent trajectory generator and dynamic PyTorch batch collator modeled after WebArena and Mind2Web interaction traces for training the JED-M2 Large Action Model.

## Requirements

### Requirement: Synthetic Multi-Turn Trajectory Generation
The trajectory generator SHALL synthesize realistic multi-turn web and OS interaction trajectories represented as tuples tau = (s_0, a_0, r_0, s_1, a_1, ..., s_T). Each turn SHALL produce structured observation features using canonical `UIElement` instances with normalized bounding boxes, reasoning rationale tokens, discrete tool primitives (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH), normalized spatial coordinates (x, y) in [0, 1], and argument payloads. Dense observation feature vectors SHALL be computed using the centralized `compute_obs_features` encoder.

#### Scenario: Generate valid multi-turn interaction sequence
- **WHEN** a synthetic trajectory of length T is generated
- **THEN** it contains T observation states, T action steps with valid tool primitives, normalized pointer coordinates in [0, 1], and string argument payloads

#### Scenario: DOM and UI bounding box consistency
- **WHEN** UI elements are generated for an interaction turn
- **THEN** each interactable element is a canonical `UIElement` with bounding box coordinates (x_min, y_min, x_max, y_max) strictly bounded within [0, 1]

#### Scenario: Canonical element serialization
- **WHEN** `to_dom_string()` is called on a generated `UIElement`
- **THEN** it returns a serialized representation including tag name, element id, role, normalized center coordinates, and text payload

### Requirement: Streaming Trajectory Dataset
The system SHALL provide a PyTorch Dataset yielding multi-turn agent trajectory samples dynamically on-the-fly, supporting configurable horizon length, random seed determinism, and streaming generation without unbounded memory growth.

#### Scenario: Infinite or streaming dataset sampling
- **WHEN** worker processes or dataloaders iterate over the trajectory dataset
- **THEN** new trajectory tuples are yielded without loading monolithic datasets into memory

### Requirement: Dynamic Trajectory Collation
The trajectory collator SHALL assemble variable-length multi-turn trajectories into batched tensors. The collated batch MUST provide padded token sequences, 2D causal attention masks, action token position indices, component loss masks (reasoning tokens, action type tokens, coordinate tokens, argument tokens), and JEPA observation-action-next transition pairs.

#### Scenario: Dynamic batch collation with causal masks
- **WHEN** a list of trajectory samples is passed to the dynamic collator
- **THEN** it outputs padded input_ids, causal attention_mask, loss masks with accurate binary selectors, and aligned JEPA (obs_t, action_t, obs_next) tensors

### Requirement: Synthetic Screenshot Rendering and Visual Feature Synthesis
The trajectory generator SHALL generate synthetic screenshot image representations `[3, H, W]` or visual patch features for interaction turns, corresponding to rendered UI element bounding boxes and screen canvas.

#### Scenario: Synthetic screenshot generation per turn
- **WHEN** interaction turns are generated in `SyntheticTrajectoryGenerator`
- **THEN** each turn provides a synthetic screenshot tensor of shape `[3, H, W]` with pixel features corresponding to the UI elements

### Requirement: Multimodal Batch Collation with Screenshot Tensors
The trajectory collator SHALL assemble screenshot images alongside text tokens, producing batched image tensors `[B, 3, H, W]` in `TrajectoryBatch`.

#### Scenario: Collate batch with visual screenshot images
- **WHEN** a list of multimodal trajectory samples is collated by `TrajectoryCollator`
- **THEN** the resulting `TrajectoryBatch` includes an `images` tensor of shape `[B, 3, H, W]` aligned with the trajectory batch

