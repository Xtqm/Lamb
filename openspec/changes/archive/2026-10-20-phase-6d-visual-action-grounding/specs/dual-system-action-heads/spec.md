# Spec Delta

## ADDED Requirements

### Requirement: Visually Grounded Spatial Coordinate Regression
The spatial coordinate regression head SHALL support cross-attention visual grounding, querying visual patch tokens using the action token hidden representation to predict bounded continuous coordinates $(x, y) \in [0, 1] \times [0, 1]$.

#### Scenario: Grounded coordinate regression with visual tokens
- **WHEN** action hidden state tensor `[B, D]` and visual patch tokens `[B, N_v, D]` are passed to the coordinate head
- **THEN** the module attends to the visual patch tokens and outputs normalized $(x, y)$ coordinates bounded in $[0, 1]$

#### Scenario: Graceful fallback without visual tokens
- **WHEN** visual patch tokens are absent (None)
- **THEN** the coordinate head regresses coordinates directly from the hidden state without error, matching text-only behavior
