# Spec Delta

## ADDED Requirements

### Requirement: Candidate Bounding Box Coordinate Snapping
The agent runtime SHALL support candidate bounding box snapping for continuous coordinate predictions, identifying the nearest candidate interactable element in the current observation and snapping emitted action coordinates to the candidate element center.

#### Scenario: Coordinate snapping to nearest candidate
- **WHEN** the agent emits a spatial action and candidate interactive elements with bounding boxes are present in the observation
- **THEN** the emitted action coordinates snap to the center of the candidate element closest in Euclidean distance to the raw predicted coordinates

#### Scenario: Snapping fallback when no candidates exist
- **WHEN** the agent emits a spatial action and no candidate elements with bounding boxes are provided
- **THEN** the emitted action coordinates retain the raw predicted continuous coordinates without error

#### Scenario: Gated distance snapping threshold
- **WHEN** a maximum snapping distance threshold is configured and the nearest candidate element exceeds that distance
- **THEN** the emitted coordinates remain the raw predicted coordinates to prevent erratic snapping across the viewport

### Requirement: Autoregressive Multi-Token Argument Generation
The agent runtime SHALL autoregressively generate multi-token string arguments for text entry actions until reaching an end-of-argument delimiter or maximum argument token length, emitting the complete decoded text payload to the environment.

#### Scenario: Multi-token search query generation
- **WHEN** the agent selects a TYPE action primitive targeting a text input element
- **THEN** the runtime autoregressively decodes argument tokens until an end delimiter or maximum length is reached, decoding the full multi-token string for environment dispatch

#### Scenario: Single-turn non-typing action argument
- **WHEN** the agent selects an action primitive that does not require textual payloads (such as CLICK or SCROLL)
- **THEN** argument decoding is bypassed or returns an empty payload
