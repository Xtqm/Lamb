# Proposal: Unify Training Pipelines & Schedulers

## Why
Currently, training execution logic is duplicated across multiple scripts in the repository:
1. `train_pretrain.py` reimplements its own manual forward/backward passes, gradient accumulation loop, Polyak momentum EMA updates to the JEPA target network, and checkpoint serialization rather than reusing `JEDM2Trainer` from `jed_m2/trainer.py`.
2. `train_rl.py` defines `CurriculumScheduler` and `ExplorationScheduler` as private, script-level classes rather than modular, exportable components in `jed_m2/rl.py`.
3. Device and precision probing, ASCII progress formatting, and structured metrics telemetry are repeatedly duplicated across CLI scripts.

Consolidating these pipelines into `jed_m2/trainer.py` and `jed_m2/rl.py` eliminates redundant code, prevents divergent training behaviors, improves testability, and ensures all training modes share identical memory-efficient optimization pathways on 8GB consumer GPUs.

## What Changes
- **Refactor `train_pretrain.py`**: Migrate the pretraining loop to delegate training step execution, gradient accumulation, optimizer scheduling, Polyak EMA target network updates, validation passes, and checkpoint persistence to `JEDM2Trainer`.
- **Modularize RL Schedulers**: Move `CurriculumScheduler` and `ExplorationScheduler` from `train_rl.py` into `jed_m2/rl.py`, re-exporting them in `jed_m2/__init__.py`. Update `train_rl.py` to import them cleanly.
- **Standardize Training Utilities**: Provide shared device/precision detection and training history logging utilities in `jed_m2/trainer.py` to eliminate copy-pasted setup code.
- **Full Backward Compatibility**: Maintain identical CLI arguments and flags for `python train.py`, `python train_pretrain.py`, and `python train_rl.py`.

## Capabilities

### Modified Capabilities
- `production-training-harness`: Update `CLI Training Orchestration` to require that pretraining and fine-tuning harness pipelines execute via the canonical `JEDM2Trainer`, guaranteeing unified gradient accumulation, Polyak momentum EMA updates, and checkpoint format parity.
- `trajectory-policy-optimization`: Update to require that multi-turn workflow curriculum scheduling and exploration temperature annealing are exposed as modular library classes in `jed_m2.rl`.

## Impact
- **Modified Files**:
  - `jed_m2/trainer.py`: Add shared device probe and history formatting utilities; support dataset/dataloader ingestion.
  - `jed_m2/rl.py`: Add `CurriculumScheduler` and `ExplorationScheduler`.
  - `jed_m2/__init__.py`: Re-export `CurriculumScheduler` and `ExplorationScheduler`.
  - `train_pretrain.py`: Refactor to use `JEDM2Trainer`.
  - `train_rl.py`: Import schedulers from `jed_m2.rl`.
- **Breaking Changes**: None. All CLI flags, function signatures, and exported symbols remain backward-compatible.
