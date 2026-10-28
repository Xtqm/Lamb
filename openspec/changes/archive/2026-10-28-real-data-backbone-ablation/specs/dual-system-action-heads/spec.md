# Spec Delta

## MODIFIED Requirements

### Requirement: Discrete Primitive Action Classification
The system SHALL classify discrete action primitives across standard interaction types including CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH, and SELECT.

#### Scenario: Classify primitive action type
- **WHEN** a hidden representation `[B, D]` or `[B, S, D]` is evaluated by the action type head
- **THEN** the module outputs unnormalized class logits `[B, num_action_types]` or `[B, S, num_action_types]` for the action types declared by the model configuration

#### Scenario: Train a real-data ablation model
- **WHEN** an ablation model is configured to represent Mind2Web operation labels
- **THEN** its seven output classes distinguish CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH, and SELECT, with SELECT assigned a new class index after the existing six

#### Scenario: Load a legacy six-action checkpoint
- **WHEN** a checkpoint created with the six-action configuration is loaded for reference evaluation
- **THEN** its original six class indices and output shape remain intact, and it is labeled as a legacy reference without a SELECT prediction class
