# Spec Delta

## MODIFIED Requirements

### Requirement: CLI Training Orchestration
The system SHALL provide a command-line interface executable script `train.py` with configurable training epochs, warmup, cosine decay schedule, logging intervals, evaluation intervals, held-out validation passes, structured history export, and real-time metric reporting.

#### Scenario: CLI training invocation
- **WHEN** `train.py` is executed with CLI arguments for steps, learning rate, and batch size
- **THEN** the script initializes dataset streaming, runs training iterations, logs loss and throughput metrics to the console, and produces valid checkpoints

#### Scenario: CLI training with periodic validation
- **WHEN** `train.py` is executed with `--eval_interval` and `--eval_steps` specified
- **THEN** the script evaluates held-out trajectories periodically and exports structured convergence history upon completion

## ADDED Requirements

### Requirement: Held-Out Trajectory Validation and Evaluation Passes
The system SHALL provide evaluation passes that compute validation loss across all 6 loss heads on held-out trajectories without backpropagation or optimizer updates.

#### Scenario: Periodic validation evaluation
- **WHEN** the training harness reaches a configured evaluation interval or evaluation pass
- **THEN** it executes inference passes over a held-out evaluation dataset, records validation losses across all heads without gradient updates, and logs evaluation metrics
