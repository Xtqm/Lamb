# Tasks

## 1. Core Schedulers & Shared Training Utilities

- [x] 1.1 Migrate `CurriculumScheduler` and `ExplorationScheduler` into `jed_m2/rl.py`, re-export them from `jed_m2/__init__.py`, and verify standalone instantiation, curriculum tier promotion, and temperature annealing with unit assertions.
- [x] 1.2 Implement canonical `probe_training_device()` in `jed_m2/trainer.py` returning `(device, dtype, is_cuda, gpu_name)` with automatic CUDA bfloat16 probing and peak memory reset; verify return values on active hardware.

## 2. Refactor Training CLI Scripts

- [x] 2.1 Refactor `train_rl.py` to import `CurriculumScheduler` and `ExplorationScheduler` from `jed_m2.rl`, removing duplicated definitions; verify CLI argument parsing via `python train_rl.py --help`.
- [x] 2.2 Refactor `train_pretrain.py` to delegate batch optimization, gradient accumulation, Polyak momentum target network updates, evaluation passes, and checkpoint persistence to `JEDM2Trainer`; verify CLI argument parsing via `python train_pretrain.py --help`.
- [x] 2.3 Validate `train_pretrain.py` pretraining step execution with a short multi-step dry run on sample batches, verifying loss computation, gradient updates, and telemetry output.

## 3. Integration Verification & Hardware Budget

- [x] 3.1 Run `pytest tests/test_real_rl.py` and verify all reinforcement learning unit tests pass cleanly with zero regressions.
- [x] 3.2 Run `python verify_phase6c.py` to verify SFT training harness, multi-head loss convergence, and baseline checkpoint parity under `JEDM2Trainer`.
- [x] 3.3 Verify peak GPU memory allocation remains strictly below the 6.0 GB hardware budget during training execution.
