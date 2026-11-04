# Spec Delta: synthetic-trajectory-dataset

## MODIFIED Requirements

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
