# Spec Delta: Mamba SSM

## Purpose

Provides a pure PyTorch selective state space model (Mamba SSM) that supports both sequence-level parallel processing and step-level recurrent constant-memory inference without external compiled C++/CUDA dependencies.

## ADDED Requirements

### Requirement: Pure PyTorch Selective State Space Model
The Mamba SSM module SHALL execute forward sequence transformations natively in PyTorch without loading or invoking external compiled C++ or CUDA extensions.

#### Scenario: Sequence forward pass on CUDA in bfloat16
- **WHEN** a 3D tensor of shape `[B, L, D]` is passed to the module in `bfloat16` on CUDA
- **THEN** the module produces an output tensor of shape `[B, L, D]` with identical spatial dimensions without raising C++/CUDA runtime errors

### Requirement: Step-Level Recurrent Inference
The Mamba SSM module SHALL provide a recurrent `step` method that updates internal state incrementally for single-token autoregressive generation.

#### Scenario: Single-step state progression
- **WHEN** an input token vector `[B, D]` is passed to `step` along with previous SSM and convolution states
- **THEN** the module outputs the predicted representation `[B, D]` and updated state tensors with constant memory footprint

### Requirement: Mathematical Equivalence of Sequence and Step Modes
The sequence forward pass and consecutive single-step recurrent evaluations SHALL yield mathematically equivalent representations within numerical tolerance.

#### Scenario: Autoregressive rollout matches sequence output
- **WHEN** a sequence of length $L$ is evaluated via `forward(x)` and independently evaluated via sequential calls to `step` starting from empty states
- **THEN** the maximum absolute error across all sequence positions SHALL remain within `bfloat16` numerical precision bounds (tolerance $\le 10^{-2}$)

### Requirement: Gradient Backpropagation Integrity
The Mamba SSM module SHALL support backward differentiation through all learnable parameters without producing NaNs or Infinities.

#### Scenario: Backward pass gradient computation
- **WHEN** a scalar loss is backpropagated through the Mamba SSM output
- **THEN** all parameter gradients are non-None, finite, and free of NaN or Inf values
