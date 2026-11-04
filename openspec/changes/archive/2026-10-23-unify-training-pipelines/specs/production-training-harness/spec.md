# Spec Delta: production-training-harness

## MODIFIED Requirements

### Requirement: CLI Training Orchestration
The system SHALL provide command-line interface executable scripts (`train.py`, `train_rl.py`, and `train_pretrain.py`) powered by the unified `JEDM2Trainer` harness with configurable training epochs, model scale tiers (`--model_size {nano,medium}`), warmup, cosine decay schedule, logging intervals, evaluation intervals, held-out validation passes, structured history export, and real-time metric reporting.

#### Scenario: CLI training invocation
- **WHEN** `train.py` is executed with CLI arguments for steps, learning rate, and batch size
- **THEN** the script initializes dataset streaming, runs training iterations via `JEDM2Trainer`, logs loss and throughput metrics to the console, and produces valid checkpoints

#### Scenario: CLI training with periodic validation
- **WHEN** `train.py` is executed with `--eval_interval` and `--eval_steps` specified
- **THEN** the script evaluates held-out trajectories periodically via `JEDM2Trainer.evaluate()` and exports structured convergence history upon completion

#### Scenario: CLI training with configurable model scale
- **WHEN** `train.py` or `train_rl.py` is invoked with `--model_size medium`
- **THEN** the harness instantiates the medium architecture (~273M parameters), enables gradient activation checkpointing, and completes training steps without exceeding the GPU memory limit

#### Scenario: Scaled real-world trajectory pretraining invocation
- **WHEN** `train_pretrain.py` is invoked with `--model_size {nano,medium}` and step targets up to 10000 steps
- **THEN** the harness streams batches from the real web trajectory dataset, executes gradient accumulation with fused AdamW using `JEDM2Trainer`, and saves pretraining checkpoints

#### Scenario: Real-time telemetry and resource monitoring
- **WHEN** pretraining executes across training iterations
- **THEN** the engine logs decomposed loss values (NLL, CE, Huber, JEPA, Aux), token processing throughput in tokens/second, gradient norms, and live GPU VRAM usage

#### Scenario: Unified trainer delegation and checkpoint parity
- **WHEN** `train_pretrain.py` executes training steps or serializes checkpoints
- **THEN** steps execute through `JEDM2Trainer.training_step()` with automatic Polyak momentum updates to the JEPA target encoder, and checkpoints match `JEDM2Trainer.save_checkpoint()` state structure
