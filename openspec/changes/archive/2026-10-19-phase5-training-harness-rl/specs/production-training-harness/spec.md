# Spec Delta

## Purpose

Provides an end-to-end memory-calibrated training harness for the JED-M2 Large Action Model on 8GB VRAM consumer GPUs with fused 8-bit AdamW, activation checkpointing, BF16 mixed-precision, gradient accumulation, and checkpoint state recovery.

## ADDED Requirements

### Requirement: Hardware-Calibrated Optimizer and Memory Optimization
The system SHALL provide an optimizer setup supporting fused 8-bit AdamW when bitsandbytes is available and falling back to native PyTorch CUDA-fused AdamW, slashing optimizer memory states while maintaining parameter convergence.

#### Scenario: Optimizer instantiation on CUDA
- **WHEN** the training harness initializes on a CUDA-enabled GPU
- **THEN** it configures 8-bit or fused AdamW optimizer state representations and asserts valid weight updates during training steps

### Requirement: Mixed-Precision Training with Gradient Accumulation
The system SHALL execute training steps in native `torch.bfloat16` precision, accumulating gradients over multiple micro-batches before clipping gradient norms and applying optimizer steps.

#### Scenario: Multi-step gradient accumulation
- **WHEN** training with micro-batch size 2-4 and gradient accumulation steps $K$
- **THEN** gradients accumulate across $K$ backward passes without intermediate parameter updates, gradients are clipped to the target max norm, and the optimizer step executes every $K$ steps with zero NaNs or Infs

### Requirement: Checkpoint Persistence and Resumption
The system SHALL atomically persist and restore model weights, optimizer states, learning rate scheduler state, and training metrics to and from disk.

#### Scenario: Checkpoint save and restore
- **WHEN** a training checkpoint is saved and subsequently restored
- **THEN** the model parameters, optimizer states, and current step match exactly, and training continues without loss divergence

### Requirement: CLI Training Orchestration
The system SHALL provide a command-line interface executable script `train.py` with configurable training epochs, warmup, cosine decay schedule, logging intervals, and real-time metric reporting.

#### Scenario: CLI training invocation
- **WHEN** `train.py` is executed with CLI arguments for steps, learning rate, and batch size
- **THEN** the script initializes dataset streaming, runs training iterations, logs loss and throughput metrics to the console, and produces valid checkpoints
