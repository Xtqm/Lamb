# composite-loss-harness Specification

## Purpose
Provides a unified composite loss harness calibrating reasoning, action classification, spatial coordinate regression, argument token decoding, JEPA latent consistency, and MoE routing uniformity.

## Requirements

### Requirement: Calibrated Composite Loss Evaluation
The composite loss harness SHALL evaluate a calibrated weighted sum of multi-task objectives: autoregressive reasoning cross-entropy, discrete action type cross-entropy, spatial coordinate Smooth L1 / Huber loss, argument payload masked cross-entropy, JEPA latent consistency loss, and MoE auxiliary load-balancing loss, properly normalizing token cross-entropy against coordinate regression to ensure balanced gradient magnitudes across heads.

#### Scenario: Forward loss computation
- **WHEN** model outputs and target batch labels are passed to the composite loss module
- **THEN** it returns a single scalar total loss tensor that is differentiable and ready for backpropagation

#### Scenario: No NaN or Inf loss evaluation
- **WHEN** computing loss on valid collated batches in bfloat16 precision
- **THEN** the returned loss is finite and gradients backpropagate without producing NaNs or Infinities

#### Scenario: Balanced gradient contribution across loss components
- **WHEN** backward pass is executed on a batch containing active thought tokens, action types, and continuous coordinates
- **THEN** parameter gradient norms originating from reasoning cross-entropy and spatial coordinate Huber loss remain within calibrated proportional bounds without reasoning dominating coordinate gradients

### Requirement: Component Loss Masking and Isolation
The composite loss module SHALL strictly isolate loss computations using binary masks: spatial coordinate loss MUST only be accumulated for actions requiring coordinates (CLICK, TYPE, SCROLL), argument loss MUST only be accumulated on designated argument tokens, and reasoning loss MUST only be computed over rationale tokens.

#### Scenario: Coordinate loss masking on non-spatial actions
- **WHEN** an action step corresponds to a non-spatial primitive (such as FINISH)
- **THEN** spatial coordinate loss for that step is zeroed out by the loss mask and does not contribute to gradient updates

### Requirement: Detailed Training Metric Logging
The loss harness SHALL return a comprehensive dictionary containing detached scalar metric values for every individual sub-loss, including total loss, reasoning loss, action type loss, coordinate loss, argument loss, JEPA consistency loss, and MoE aux loss.

#### Scenario: Logging dictionary generation
- **WHEN** the composite loss forward pass completes
- **THEN** the returned metrics dictionary contains key-value pairs of detached float scalars for all component losses
