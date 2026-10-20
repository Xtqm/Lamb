# Tasks

## 1. Interactive RL Pipeline (`jed_m2/rl_pipeline.py`)

- [x] 1.1 Implement `RolloutTrajectory` and trajectory turn records capturing token IDs, log-probabilities, actions, coordinates, arguments, step rewards, and terminal success rewards. Verify data structures and dictionary serialization.
- [x] 1.2 Implement temperature-scaled stochastic exploration sampling in `TrajectoryRolloutSampler` wrapping `JEDM2Agent` and `WebOSEnvironment` for System 2 thoughts and System 1 action primitives. Verify stochastic diversity across repeated rollouts.
- [x] 1.3 Implement group rollout collection (`sample_group_rollouts`) computing group-normalized relative advantages for GRPO, and preference pairing (`create_preference_pairs`) matching winning ($R=1.0$) and losing ($R=0.0$) trajectories for DPO. Verify group sizing, advantage normalization, and pair matching.
- [x] 1.4 Implement batch collation utilities converting groups of multi-turn rollouts into padded PyTorch tensors (`input_ids`, `targets`, `mask`, `old_token_logps`, `rewards`) compatible with `GRPOTrainer` and `DPOTrainer`. Verify padding, masks, and tensor shapes.

## 2. Production RL Training CLI Script (`train_rl.py`)

- [x] 2.1 Implement CLI interface in `train_rl.py` supporting `--mode grpo` and `--mode dpo` with configurable hyperparameters (learning rate, temperature, group size, beta KL, iterations, scenarios, checkpointing). Verify CLI parser with `python train_rl.py --help`.
- [x] 2.2 Implement online GRPO training loop: environment group rollouts -> advantage normalization -> active and reference forward passes -> clipped surrogate loss with KL divergence constraint -> optimizer step. Verify policy gradient backpropagation without NaNs/Infs.
- [x] 2.3 Implement offline DPO training loop optimizing implicit reward margins on collected preference trajectory pairs. Verify DPO objective evaluation and accuracy reporting.
- [x] 2.4 Implement live ASCII telemetry reporting average episode reward, advantage variance, reference KL divergence, clip fraction, parameter gradient norms, and token throughput.

## 3. Verification Suite & Architecture Context

- [x] 3.1 Export `TrajectoryRolloutSampler`, `RolloutTrajectory`, and pipeline helpers in `jed_m2/__init__.py`. Verify clean imports in Python.
- [x] 3.2 Implement `verify_phase6b.py` covering multi-group rollout sampling, preference pairing, GRPO policy gradient updates, 25-iteration reward improvement on simulated tasks, and VRAM memory ceiling compliance (< 6.0 GB). Verify by executing `python verify_phase6b.py`.
- [x] 3.3 Update `PROJECT_CONTEXT.md` roadmap marking Phase 6B as completed with benchmark results and hardware memory profiles.
