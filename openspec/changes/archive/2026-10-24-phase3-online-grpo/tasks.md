# Tasks: Phase 3 - Online GRPO Training and Real-Web Benchmark Verification

## 1. Browser Fixture Routing & Online GRPO Execution

- [x] 1.1 Update `PlaywrightEnvironment.reset` in `jed_m2/browser.py` and `train_rl.py` to dynamically route multi-scenario browser rollouts to local HTML fixtures (`ecommerce`, `form_portal`, `travel`).
- [x] 1.2 Execute online GRPO training via `train_rl.py --mode grpo --env_type browser` seeded from `checkpoints_rl/jed_m2_nano_dpo_calibrated.pt` with group size $G=4$, reference KL penalty $\beta = 0.04$, and spatial coordinate weight $\lambda_{\text{coord}} = 0.5$, verifying peak VRAM stays below 6.0 GB.
- [x] 1.3 Save final calibrated policy checkpoint `checkpoints_rl/jed_m2_nano_grpo_final.pt`.
- [x] 1.4 Benchmark the final RL policy on `real_web_val` (15 episodes, 109 turns) via `eval_benchmarks.py`, comparing spatial MAE, action F1, and step success rate against Step 15,800 baseline, and update `BENCHMARK_REPORT.md`.
