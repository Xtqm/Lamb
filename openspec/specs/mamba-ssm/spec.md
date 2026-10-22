# mamba-ssm Specification

## Purpose
Provides a pure PyTorch selective state space model (Mamba SSM) that supports both sequence-level parallel processing and step-level recurrent constant-memory inference without external compiled C++/CUDA dependencies.

## Requirements

### Requirement: Pure PyTorch Selective State Space Model
The Mamba SSM module SHALL execute forward sequence transformations natively in PyTorch using a parallel associative prefix scan (Blelloch scan) without loading or invoking external compiled C++ or CUDA extensions.

#### Scenario: Sequence forward pass on CUDA in bfloat16
- **WHEN** a 3D tensor of shape `[B, L, D]` is passed to the module in `bfloat16` on CUDA
- **THEN** the module produces an output tensor of shape `[B, L, D]` with identical spatial dimensions without raising C++/CUDA runtime errors

#### Scenario: Parallel Blelloch associative scan execution
- **WHEN** evaluating a sequence forward pass of length $L \ge 1024$
- **THEN** the SSM recurrence is computed via tree-based parallel associative prefix scan operations in native PyTorch tensors, eliminating sequential Python for-loops over time steps and achieving $>1,200$ tokens/second training throughput

### Requirement: Step-Level Recurrent Inference
The Mamba SSM module SHALL provide a recurrent `step` method that updates internal state incrementally for single-token autoregressive generation.

#### Scenario: Single-step state progression
- **WHEN** an input token vector `[B, D]` is passed to `step` along with previous SSM and convolution states
- **THEN** the module outputs the predicted representation `[B, D]` and updated state tensors with constant memory footprint

### Requirement: Mathematical Equivalence of Sequence and Step Modes
The parallel prefix scan sequence forward pass and consecutive single-step recurrent evaluations SHALL yield mathematically equivalent representations within strict numerical tolerance.

#### Scenario: Autoregressive rollout matches sequence output
- **WHEN** a sequence of length $L$ is evaluated via `forward(x)` using parallel associative scan and independently evaluated via sequential calls to `step` starting from empty states
- **THEN** the maximum absolute error across all sequence positions SHALL remain within numerical precision bounds (tolerance $\le 10^{-5}$ in float32 and $\le 10^{-2}$ in bfloat16)

#### Scenario: Equivalence with legacy sequential recurrence
- **WHEN** comparing the parallel associative scan output against the sequential recurrence loop output on identical inputs
- **THEN** the maximum element-wise difference between both computation paths across all hidden states SHALL be strictly less than $1\text{e-}5$ in float32 precision

### Requirement: Gradient Backpropagation Integrity
The Mamba SSM module SHALL support backward differentiation through all learnable parameters without producing NaNs or Infinities.

#### Scenario: Backward pass gradient computation
- **WHEN** a scalar loss is backpropagated through the Mamba SSM output
- **THEN** all parameter gradients are non-None, finite, and free of NaN or Inf values
