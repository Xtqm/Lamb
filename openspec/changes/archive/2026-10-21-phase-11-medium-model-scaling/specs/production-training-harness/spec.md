# Spec Delta

## MODIFIED Requirements

### Requirement: CLI Training Orchestration
The system SHALL provide command-line interface executable scripts (`train.py` and `train_rl.py`) with configurable training epochs, model scale tiers (`--model_size {nano,medium}`), warmup, cosine decay schedule, logging intervals, evaluation intervals, held-out validation passes, structured history export, and real-time metric reporting.

#### Scenario: CLI training invocation
- **WHEN** `train.py` is executed with CLI arguments for steps, learning rate, and batch size
- **THEN** the script initializes dataset streaming, runs training iterations, logs loss and throughput metrics to the console, and produces valid checkpoints

#### Scenario: CLI training with periodic validation
- **WHEN** `train.py` is executed with `--eval_interval` and `--eval_steps` specified
- **THEN** the script evaluates held-out trajectories periodically and exports structured convergence history upon completion

#### Scenario: CLI training with configurable model scale
- **WHEN** `train.py` or `train_rl.py` is invoked with `--model_size medium`
- **THEN** the harness instantiates the medium architecture (~273M parameters), enables gradient activation checkpointing, and completes training steps without exceeding the GPU memory limit
