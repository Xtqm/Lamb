# Spec Delta

## ADDED Requirements

### Requirement: Calibrated Text Argument Loss Weighting
The composite loss harness SHALL calibrate argument cross-entropy loss weighting relative to coordinate Huber loss and reasoning cross-entropy, preventing vocabulary-size scaling from attenuating argument prediction gradients below action classification and spatial regression magnitudes.

#### Scenario: Balanced argument gradient scale
- **WHEN** backward pass is executed on a batch containing text argument tokens alongside continuous coordinates and action types
- **THEN** argument cross-entropy contributes proportional gradient norms to the model backbone, preventing coordinate and action type losses from overwhelming text query optimization

#### Scenario: Argument loss calculation across multi-token payloads
- **WHEN** a batch contains multi-token argument sequences marked by the argument loss mask
- **THEN** the loss module computes masked cross-entropy over all valid argument token positions and records the detached scalar metric
