# Design: Unify Training Pipelines & Schedulers

## Context
See `proposal.md` for motivation. Currently, `JEDM2Trainer` (`jed_m2/trainer.py`) provides a production-grade training harness calibrated for 8GB VRAM consumer GPUs with fused 8-bit / CUDA AdamW, activation checkpointing, Polyak momentum EMA updates to the JEPA target network, atomic checkpoint persistence, and multi-head held-out evaluation.

However:
1. `train_pretrain.py` duplicates the entire training and evaluation loop, gradient accumulation counters, Polyak EMA logic, and checkpoint serialization manually instead of delegating to `JEDM2Trainer`.
2. `train_rl.py` implements `CurriculumScheduler` and `ExplorationScheduler` as private top-level classes in the script rather than modular classes in `jed_m2/rl.py`.
3. Hardware device and precision detection (`probe_device_and_precision`) and history logging dictionaries are duplicated across `train.py`, `train_pretrain.py`, and verification scripts.

## Goals / Non-Goals

**Goals:**
- Unify the pretraining loop in `train_pretrain.py` by delegating training steps, gradient accumulation, optimizer updates, evaluation, and checkpoint persistence to `JEDM2Trainer`.
- Modularize `CurriculumScheduler` and `ExplorationScheduler` into `jed_m2/rl.py` and re-export them from `jed_m2/__init__.py`.
- Provide canonical training device probing (`probe_training_device`) and history export utilities in `jed_m2/trainer.py`.
- Maintain 100% backward compatibility for all CLI flags, arguments, and public import symbols across `train.py`, `train_pretrain.py`, and `train_rl.py`.

**Non-Goals:**
- Modifying core loss mathematical formulations (e.g. composite loss weights, DPO loss, GRPO advantage normalization).
- Altering model architecture definitions or configuration dataclasses (`JEDM2Config`, `JEDM2MediumConfig`).
- Changing directory structures or output formats of existing benchmarks.

## Decisions

### 1. Delegate `train_pretrain.py` Step Execution to `JEDM2Trainer`
- **Decision**: In `train_pretrain.py`, instantiate `JEDM2Trainer` with `JEDM2TrainerConfig` derived from CLI arguments. Replace the manual backward, gradient accumulation, clipping, Polyak EMA update, and checkpoint save code with calls to `trainer.training_step(batch)`, `trainer.evaluate(...)`, and `trainer.save_checkpoint(...)`.
- **Rationale**: `RealTrajectoryCollator` outputs standard `TrajectoryBatch` objects identical to `TrajectoryCollator`. `JEDM2Trainer` already implements exact gradient accumulation, clipping, scheduler stepping, Polyak EMA target network updates, and atomic checkpoint persistence. Delegating eliminates ~250 lines of duplicate code and guarantees behavior consistency across pretraining and fine-tuning.
- **Alternatives Considered**: Keep a distinct `PretrainTrainer` subclass. Rejected as unnecessary overhead; `JEDM2Trainer` is already designed to train any `TrajectoryBatch` over `JEDM2Model`.

### 2. Move RL Schedulers to `jed_m2/rl.py`
- **Decision**: Move `CurriculumScheduler` and `ExplorationScheduler` into `jed_m2/rl.py`. Re-export both classes in `jed_m2/__init__.py` and import them into `train_rl.py`.
- **Rationale**: Schedulers control RL-specific training mechanics (advancing difficulty tiers across multi-turn procedural tasks, annealing exploration temperature, and decaying entropy bonus coefficients). Placing them in `jed_m2/rl.py` makes them reusable across RL scripts, pipeline unit tests, and offline evaluations.
- **Alternatives Considered**: Keep them in `train_rl.py`. Rejected because it prevents automated unit testing and third-party orchestration without importing a top-level CLI script.

### 3. Canonical Device & Precision Probing in `jed_m2/trainer.py`
- **Decision**: Add `probe_training_device()` in `jed_m2/trainer.py` returning `(device, dtype, is_cuda, gpu_name)`.
- **Rationale**: Currently, `train.py`, `train_pretrain.py`, and verification scripts duplicate CUDA checking, peak memory resetting, and bfloat16 detection. A centralized helper ensures uniform device configuration and fallback.
- **Alternatives Considered**: Put it in `jed_m2/config.py`. Rejected because hardware probing involves runtime torch device allocation and benchmark memory resetting, which belongs in runtime execution (`trainer.py`).

## Risks / Trade-offs

- **[Risk]**: Pretraining checkpoint format divergence.
  - **Mitigation**: `JEDM2Trainer.save_checkpoint()` stores model state, optimizer state, scheduler state, model config, trainer config, loss config, global step, and allows passing custom metadata via `additional_info`. `train_pretrain.py` can pass custom tags (`dataset_size`, `model_size`) in `additional_info`, ensuring 100% checkpoint compatibility.
- **[Risk]**: Data collation type mismatch between synthetic and real trajectories.
  - **Mitigation**: `RealTrajectoryCollator` outputs `TrajectoryBatch` with tensors on CPU. The training loop calls `batch.to(device=trainer.device, dtype=trainer.dtype)` before `trainer.training_step(batch)`, exactly as in `train.py`.
