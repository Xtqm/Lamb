# Proposal: Phase 3 - Online GRPO Training and Real-Web Benchmark Verification

## Why

With Phase 1 (offline Mind2Web DPO preference calibration) and Phase 2 (grounded observation parsing, goal conditioning, and Playwright spatial reward oracles) complete, the model and rollout environment are fully aligned. Phase 3 executes online Group Relative Policy Optimization (GRPO) on verified web tasks, evaluating grouped rollouts ($G=4$) with reference policy KL divergence regularization ($\beta = 0.04$) and continuous spatial coordinate optimization ($\lambda_{\text{coord}} = 0.5$) under 6.0 GB VRAM, followed by benchmarking on `real_web_val`.

## What Changes

- **Online Browser GRPO Loop Execution**: Execute online GRPO with `train_rl.py --mode grpo --env_type browser` seeding from `checkpoints_rl/jed_m2_nano_dpo_calibrated.pt`.
- **Dynamic Scenario & Local Fixture Routing**: Automatically route each iteration's scenario (`ecommerce`, `form_portal`, `travel`) to its verified local HTML fixture with dense spatial IoU rewards.
- **Checkpoint Persistence**: Save the final calibrated policy as `checkpoints_rl/jed_m2_nano_grpo_final.pt`.
- **Benchmark Evaluation**: Run comprehensive evaluation on `real_web_val` (15 episodes, 109 turns) via `eval_benchmarks.py` and document metrics in `BENCHMARK_REPORT.md`.

## Capabilities

### Modified Capabilities

- `rl-rollout-pipeline`: Verify online policy gradient optimization with reference model KL constraints, group advantage normalization, and continuous coordinate gradient scaling during live browser rollouts.
- `evaluation-benchmark-harness`: Benchmark the final online GRPO-trained policy against pre-RL and pretrain baselines on real web validation tasks.

## Impact

- **Affected Code**: `train_rl.py`, `jed_m2/rl_pipeline.py`, `jed_m2/browser.py`, `BENCHMARK_REPORT.md`.
- **Checkpoints**: Produces `checkpoints_rl/jed_m2_nano_grpo_final.pt`.
