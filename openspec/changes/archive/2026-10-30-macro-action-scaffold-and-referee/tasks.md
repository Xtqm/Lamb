# Tasks

## 1. Macro-Action Scaffold & Completion Referee Implementation

- [x] 1.1 Implement `ExecutionScaffold` and `CompletionReferee` in `jed_m2/scaffold.py` supporting input clearing before typing, DOM render stability polling, and goal milestone detection; verify with new unit tests in `tests/test_scaffold.py`.
- [x] 1.2 Update `WebOSEnvironment` and `RealTaskDOMEnvironment` to expose referee-queryable goal criteria and handle macro-action input clearing; verify with scenario step tests in `tests/test_environment.py`.

## 2. Rollout Pipeline & Decoupled Reward Integration

- [x] 2.1 Integrate `ExecutionScaffold` and `CompletionReferee` into `TrajectoryRolloutSampler.sample_trajectory` in `jed_m2/rl_pipeline.py` to detect goal completion and terminate rollouts cleanly; verify via trajectory rollout tests in `tests/test_rl_pipeline.py`.
- [x] 2.2 Calibrate `compute_shaped_reward` in `jed_m2/rl.py` to implement the decoupled multi-factor formulation ($R = w_{\text{type}} R_{\text{type}} + w_{\text{coord}} R_{\text{coord}} + w_{\text{arg}} R_{\text{arg}} + w_{\text{complete}} R_{\text{complete}}$); verify via unit tests in `tests/test_rl.py`.

## 3. End-to-End Verification & RL Sanity Probe

- [x] 3.1 Run interactive episode rollouts across WebOS scenarios (`ecommerce`, `travel`, `os_admin`) using `checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt` and verify that episodes achieve non-zero task completion and avoid 8-step infinite truncation.
- [x] 3.2 Execute a 1-iteration online GRPO probe run via `train_rl.py` with the active scaffold and referee, verifying stable gradient norms ($\le 5.0$), valid advantages, and uncorrupted memory telemetry.

## Workflow follow-up

- Review and archive the change with `/openspec-archive-change` upon successful verification.
- Proceed to full online GRPO reinforcement learning training run.
