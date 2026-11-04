# Spec Delta: real-web-trajectory-dataset

## MODIFIED Requirements

### Requirement: Normalized Spatial Bounding Box and Coordinate Extraction
The system SHALL extract element bounding box coordinates $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ normalized in $[0, 1]^2$ and compute normalized center pointer coordinates $(x, y)$ for each interactive element, structured as canonical `UIElement` instances with `InteractiveElement` maintained as a backward-compatible alias. Dense observation features for JEPA and model conditioning SHALL be produced using the centralized `compute_obs_features` utility.

#### Scenario: Bounding box normalization
- **WHEN** raw pixel or viewport bounding boxes are provided
- **THEN** the system normalizes coordinates to $[0, 1]^2$ and computes center points matching target click locations

#### Scenario: InteractiveElement type aliasing and interoperability
- **WHEN** downstream code accesses elements via `InteractiveElement` or `UIElement`
- **THEN** both types resolve to the same underlying canonical class and provide equivalent attributes, centers, and `to_dom_string()` representations
