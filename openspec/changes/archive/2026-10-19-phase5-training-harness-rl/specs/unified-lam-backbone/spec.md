# Spec Delta

## ADDED Requirements

### Requirement: Selective Gradient Checkpointing Support
The system SHALL provide selective activation checkpointing across the stacked JEDM2Block layers within JEDM2Model to constrain backpropagation activation memory at long sequence lengths.

#### Scenario: Checkpointed sequence forward and backward pass
- **WHEN** gradient checkpointing is enabled on JEDM2Model and a sequence of length 2048 is evaluated through backward pass on CUDA in bfloat16
- **THEN** intermediate activations are recomputed during backward pass, gradients are computed without NaNs or Infs, and peak activation memory is reduced compared to non-checkpointed training
