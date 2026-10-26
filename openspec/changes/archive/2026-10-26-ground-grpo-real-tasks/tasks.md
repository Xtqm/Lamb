# Tasks: Ground GRPO Rollouts on Diverse Real Web Tasks

## 1. Real-Task DOM Replay Environment

- [x] 1.1 Implement `RealTaskDOMEnvironment` in `jed_m2/real_environment.py` to ingest offline task traces from Mind2Web and WebArena, parse interactive element candidate trees using `FastDOMParser`, and verify task initialization with `pytest tests/test_real_environment.py -k test_task_init`.
- [x] 1.2 Implement the multi-attribute programmatic oracle in `jed_m2/real_environment.py` computing composite step rewards ($R_{\text{action\_type}}$, $R_{\text{element\_match}}$, $R_{\text{spatial\_iou}}$, $R_{\text{coord\_dist}}$, $R_{\text{argument}}$), and verify scoring across positive and negative actions with `pytest tests/test_real_environment.py -k test_oracle_rewards`.

## 2. Rollout Sampler and GRPO Pipeline Integration

- [x] 2.1 Refactor `TrajectoryRolloutSampler` in `jed_m2/rl.py` to support `--env_type real_web`, generating groups of $G=4$ rollouts per sampled real web task with group advantage normalization, and verify group collation with `pytest tests/test_rl_rollout.py -k test_real_task_rollout_sampling`.
- [x] 2.2 Wire `--env_type real_web` into `train_rl.py` with arguments for Mind2Web and WebArena trace paths, sequence length 1024 enforcement, and verify dry-run startup with `python train_rl.py --mode grpo --env_type real_web --dry_run`.

## 3. End-to-End Calibration and Verification

- [x] 3.1 Execute end-to-end online GRPO training across real web tasks for 10 iterations ($G=4$) and verify joint discrete and continuous coordinate policy updates, advantage variance, and VRAM stability (< 6.0 GB) with `python train_rl.py --mode grpo --env_type real_web --max_steps 10 --group_size 4`.
- [x] 3.2 Evaluate the GRPO checkpoint on the held-out real web evaluation suite with `python eval_benchmarks.py --benchmark real_web --split val --num_episodes 15` and verify that Action Macro F1, TYPE F1, and Coordinate MAE meet or exceed SFT baselines.

## Workflow follow-up

- Archive the change after the project's review requirements are satisfied.
- Verify the archived result.
