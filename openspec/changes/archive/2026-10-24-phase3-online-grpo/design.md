# Design: Phase 3 - Online GRPO Training and Real-Web Benchmark Verification

## Context

Following Phase 1 (offline DPO calibration) and Phase 2 (grounded observation tokenization, goal conditioning, and Playwright spatial reward oracles), the model and browser environment are fully aligned. Phase 3 executes online Group Relative Policy Optimization (GRPO) on verified web tasks, evaluating grouped rollouts ($G=4$) with reference policy KL divergence regularization ($\beta = 0.04$) and continuous spatial coordinate optimization ($\lambda_{\text{coord}} = 0.5$) under 6.0 GB VRAM, followed by benchmarking on `real_web_val`.

## Goals / Non-Goals

**Goals:**
- Update `train_rl.py` and `jed_m2/browser.py` to route multi-scenario browser rollouts dynamically to local HTML fixtures (`ecommerce`, `form_portal`, `travel`).
- Execute online GRPO training run seeded from `checkpoints_rl/jed_m2_nano_dpo_calibrated.pt` with $G=4$ group rollouts and spatial coordinate policy gradients under 6.0 GB VRAM.
- Save final model checkpoint `checkpoints_rl/jed_m2_nano_grpo_final.pt`.
- Run comparative evaluation on `real_web_val` (15 episodes, 109 turns) with `eval_benchmarks.py` and document metrics in `BENCHMARK_REPORT.md`.

**Non-Goals:**
- Running GRPO on unverified external websites without ground-truth oracles.
- Altering model architecture hyperparameters.

## Decisions

### Decision 1: Seed Online GRPO from DPO-Calibrated Checkpoint
- **Rationale**: The DPO-calibrated checkpoint (`jed_m2_nano_dpo_calibrated.pt`) has already resolved misclicks and action primitive errors on Mind2Web traces, providing a warm policy initialization that prevents early exploration collapse.

### Decision 2: Multi-Domain Local Fixture Routing in `PlaywrightEnvironment.reset`
- **Rationale**: When sampling rollouts across scenarios (`ecommerce`, `form_portal`, `travel`), `reset(scenario_id=...)` dynamically reloads the corresponding local HTML fixture, ensuring 100% reproducible DOM interactions without network variance.

## Risks / Trade-offs

- **[Risk] Peak VRAM exceeding 6.0 GB during batched GRPO backward pass.**
  - *Mitigation*: Sequential rollout sampling with `G=4`, activation checkpointing enabled, and bfloat16 mixed precision.
