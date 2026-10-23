# Tasks

## 1. Track A — Offline Mind2Web Preference Dataset & DPO Integration

- [x] 1.1 Implement `RealWebPreferenceDataset` in `jed_m2/real_preference_dataset.py` with expert demonstration extraction $(\tau_w)$ and spatial coordinate perturbation $(|\Delta x| > 0.15, |\Delta y| > 0.15)$ / action substitution $(\tau_l)$; verify extraction and perturbation generation with unit test assertions.
- [x] 1.2 Implement `RealPreferenceCollator` in `jed_m2/real_preference_dataset.py` to collate paired chosen and rejected trajectories into padded token tensors, attention masks, and action loss masks; verify batched tensor shapes in `tests/test_real_rl.py`.
- [x] 1.3 Wire `--dataset real_mind2web` into `train_rl.py` for `--mode dpo`, executing `calculate_dpo_loss()` and logging reward margins; verify offline DPO loop execution via `python train_rl.py --mode dpo --dataset real_mind2web --iterations 2`.

## 2. Track B — Headless Interactive Browser Sandbox & Oracle Verifiers

- [x] 2.1 Implement `PlaywrightWebEnvironment` in `jed_m2/playwright_env.py` conforming to the `WebOSEnvironment` interface (`reset()`, `step(action_type, coordinates, argument)`), normalized coordinate scaling to 1280x800 viewport, and native Chromium event dispatch; verify headless browser launch and stepping.
- [x] 2.2 Implement declarative `WebTaskOracle` engine in `jed_m2/playwright_env.py` evaluating URL patterns, DOM selector presence/absence, and milestone/terminal reward shaping; verify oracle evaluation with unit tests.
- [x] 2.3 Create reproducible local HTML fixtures (e-commerce catalog, form wizard) and wire them as default local benchmark tasks in `PlaywrightWebEnvironment`; verify navigation and observation generation on local fixtures.

## 3. Track C — Dual-Modality GRPO Verification on Interactive Web

- [x] 3.1 Extend `TrajectoryRolloutSampler` in `jed_m2/rl_pipeline.py` to support `PlaywrightWebEnvironment` instances and collect groups of $G$ rollouts with continuous spatial Gaussian coordinate sampling; verify group rollout generation.
- [x] 3.2 Update `train_rl.py --mode grpo` to support browser environment rollouts, computing joint policy updates across discrete token heads and continuous spatial coordinate heads (`lambda_coord`); verify dual-modality surrogate loss computation.
- [x] 3.3 Validate backward pass gradient flow and peak VRAM budget (< 6.0 GB) on NVIDIA RTX 5060 Laptop GPU with bfloat16 and gradient checkpointing; verify execution via `python train_rl.py --mode grpo --env_type browser --iterations 2`.

## 4. End-to-End Verification & Test Suite

- [x] 4.1 Implement comprehensive unit and integration test suite in `tests/test_real_rl.py` covering dataset collation, browser interaction, oracle reward shaping, and training step backward passes; verify all tests pass with `pytest tests/test_real_rl.py`.
- [x] 4.2 Verify pure 7-bit ASCII console output and git hygiene compliance ensuring `.pt` weights, raw dataset JSONs, and cache files are ignored; verify using `git status` and test log inspection.

## Workflow follow-up

- Review proposal, specs, design, and tasks with the team before implementation.
- Execute `/openspec-apply-change` to implement the tasks.
