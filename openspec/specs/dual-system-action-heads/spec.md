# dual-system-action-heads Specification

## Purpose

Provides a dual-system action decoding subsystem for Large Action Models, predicting when to emit executable actions versus continue latent reasoning, along with discrete action primitives, continuous screen pointer coordinates, and tokenized arguments.

## Requirements

### Requirement: Action Emission Gating
The system SHALL provide an action gate that maps latent representations to a scalar projection predicting whether to commit to an executable environment action or continue latent rationale generation.

#### Scenario: Predict action commitment
- **WHEN** a latent hidden state tensor of shape `[B, D]` or sequence `[B, S, D]` is passed to the action gate
- **THEN** the module outputs a tensor of shape `[B, 1]` or `[B, S, 1]` representing action emission gating logits or probabilities

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

### Requirement: Bounded Continuous Coordinate Regression
The system SHALL regress continuous screen pointer coordinates normalized within the unit interval $[0, 1] \times [0, 1]$ using a sigmoid-bounded projection.

#### Scenario: Spatial coordinate prediction
- **WHEN** a hidden representation is processed by the coordinate regressor
- **THEN** the predicted coordinates $(x, y)$ have shape `[B, 2]` or `[B, S, 2]` with all values strictly bounded in the range $[0, 1]$

### Requirement: Tokenized Action Argument Projection
The system SHALL project hidden representations to discrete vocabulary logits for generating tokenized action arguments such as keystroke text and query inputs.

#### Scenario: Argument token distribution projection
- **WHEN** a hidden representation is passed to the argument head
- **THEN** the module outputs logits of shape `[B, vocab_size]` or `[B, S, vocab_size]` matching the model vocabulary dimension

### Requirement: Composite Native Action Coordination
The system SHALL coordinate the action gate, primitive classifier, coordinate regressor, and argument head in a unified composite module capable of predicting complete structured action outputs in a single forward execution.

#### Scenario: Composite action forward evaluation
- **WHEN** a sequence or single-step hidden state tensor is passed to the composite native action head
- **THEN** the module returns a structured container containing gate predictions, action type logits, normalized coordinates, and argument logits with consistent batch and sequence dimensions

### Requirement: Visually Grounded Spatial Coordinate Regression
The spatial coordinate regression head SHALL support cross-attention visual grounding, querying visual patch tokens using the action token hidden representation to predict bounded continuous coordinates $(x, y) \in [0, 1] \times [0, 1]$.

#### Scenario: Grounded coordinate regression with visual tokens
- **WHEN** action hidden state tensor `[B, D]` and visual patch tokens `[B, N_v, D]` are passed to the coordinate head
- **THEN** the module attends to the visual patch tokens and outputs normalized $(x, y)$ coordinates bounded in $[0, 1]$

#### Scenario: Graceful fallback without visual tokens
- **WHEN** visual patch tokens are absent (None)
- **THEN** the coordinate head regresses coordinates directly from the hidden state without error, matching text-only behavior
