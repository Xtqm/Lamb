# sft-convergence-benchmarking Specification

## Purpose
Orchestrates comprehensive 1,000-step supervised fine-tuning (SFT) pretraining runs on synthetic WebArena/Mind2Web trajectories, tracking convergence across all six loss heads, coordinate MAE, and primitive accuracy while curating production baseline checkpoints.

## Requirements

### Requirement: SFT Pretraining Benchmark Orchestration
The system SHALL provide an automated benchmarking script `benchmark_sft.py` that orchestrates a full 1,000-step supervised fine-tuning run on synthetic WebArena and Mind2Web trajectories using `train.py` with effective batch size 16 (micro-batch size 2, gradient accumulation 8) and sequence length 2048.

#### Scenario: 1,000-step pretraining run execution
- **WHEN** `benchmark_sft.py` is executed with default parameters or target step budget
- **THEN** it configures and runs `train.py` with sequence length 2048, micro-batch size 2, and gradient accumulation 8, executing 1,000 training steps within the 6.0 GB VRAM envelope

### Requirement: Multi-Head Loss and Accuracy Convergence Logging
The system SHALL track and log loss curves across all six loss heads (reasoning, action gate, action primitive, coordinate regression, JEPA latent consistency, and MoE routing) along with coordinate Mean Absolute Error (MAE) and action primitive classification accuracy over training steps.

#### Scenario: Metric curve logging over training steps
- **WHEN** training steps are executed during the SFT benchmark run
- **THEN** the system logs individual loss metrics for reasoning, gate, primitive, coordinates, JEPA, MoE load balancing, coordinate MAE, and primitive classification accuracy at each logging interval

### Requirement: Held-Out Trajectory Validation Tracking
The system SHALL evaluate model checkpoints and validation loss on held-out synthetic trajectories at periodic intervals during the pretraining run.

#### Scenario: Validation loss evaluation on held-out trajectories
- **WHEN** the benchmark reaches a configured evaluation checkpoint interval
- **THEN** the system evaluates held-out synthetic trajectories without updating model weights, logging evaluation loss and spatial accuracy metrics

### Requirement: Hardware Telemetry and Throughput Monitoring
The system SHALL monitor and record token throughput (tokens per second), parameter gradient norm stability, and peak GPU VRAM allocation during the SFT run.

#### Scenario: Hardware efficiency and stability profiling
- **WHEN** training steps execute on the target GPU
- **THEN** the system records token throughput, gradient clipping norms, and peak allocated VRAM, asserting peak memory remains strictly below 6.0 GB and gradient norms remain non-divergent without NaNs or Infs

### Requirement: Production Baseline Checkpoint Curation
The system SHALL save the final converged production model weights, optimizer states, scheduler state, and training configuration as the baseline checkpoint `checkpoints/jed_m2_sft_base.pt`.

#### Scenario: Baseline checkpoint verification
- **WHEN** the 1,000-step SFT benchmark run completes successfully
- **THEN** the baseline checkpoint `checkpoints/jed_m2_sft_base.pt` is atomically saved to disk and can be reloaded with full model and optimizer state parity
