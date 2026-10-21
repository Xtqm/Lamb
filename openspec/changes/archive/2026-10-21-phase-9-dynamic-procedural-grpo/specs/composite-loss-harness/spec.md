# Spec Delta

## MODIFIED Requirements

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
