# Tasks: Phase 8 Standardized Evaluation Benchmark Harness

## 1. Standardized Evaluation Metrics & Benchmark Harness (`jed_m2/benchmark.py`)

- [x] 1.1 Implement metric dataclasses (`StepMetric`, `EpisodeMetric`, `BenchmarkSummary`) and `TaskEvaluator` computing Task Success Rate (SR %), Step Success Rate, Action Primitive F1 (micro/macro precision, recall, F1 across 6 discrete categories), Spatial Coordinate Accuracy (MAE and $\epsilon \le 0.05$ tolerance radius / bounding box containment), Trajectory Efficiency Ratio ($L_{\text{optimal}} / L_{\text{agent}}$), and Decision Latency (ms/step, token throughput), verifying metric computation logic and zero-division guards.
- [x] 1.2 Implement offline evaluation routine `evaluate_offline_dataset(...)` replaying trajectory datasets (Mind2Web / WebArena test and validation splits) against ground-truth element bounding boxes, continuous coordinates, and discrete action primitives, verifying step-by-step matching without environment simulation.
- [x] 1.3 Implement online closed-loop evaluation routine `evaluate_online_environment(...)` executing multi-turn agent rollouts across interactive environments (`WebOSEnvironment` sandboxes and live Playwright browser sessions) until goal completion or maximum step ceiling, verifying closed-loop interaction and episode termination.
- [x] 1.4 Export `TaskEvaluator`, `BenchmarkSummary`, `evaluate_offline_dataset`, and `evaluate_online_environment` in `jed_m2/__init__.py`, verifying clean public imports.

## 2. Command-Line Benchmark Runner & Comparative Reporting (`eval_benchmarks.py`)

- [x] 2.1 Implement CLI interface in `eval_benchmarks.py` supporting flags `--benchmark {mind2web,webarena,simulated}`, `--model_path <path>`, `--checkpoints <paths...>`, `--split {val,test}`, `--num_episodes <n>`, `--max_steps <n>`, and `--device {cuda,cpu}`, verifying argument parsing and execution paths.
- [x] 2.2 Implement side-by-side comparative evaluation logic in `eval_benchmarks.py` evaluating multiple checkpoints sequentially on identical seeded task splits with explicit cache clearing (`torch.cuda.empty_cache()`), verifying comparative metric collation.
- [x] 2.3 Implement ASCII terminal summary table formatting and automated Markdown report generation (`BENCHMARK_REPORT.md`), verifying clean 7-bit ASCII table output and complete markdown exports.

## 3. End-to-End Verification Suite (`verify_phase8.py`)

- [x] 3.1 Implement and run `verify_phase8.py` verifying offline dataset replay, online closed-loop evaluation, metric numerical sanity (SR in $[0, 1]$, Step SR in $[0, 1]$, F1 in $[0, 1]$, MAE $\ge 0$, no NaNs/Infs), side-by-side checkpoint comparison logic, and strict compliance with the 6.0 GB VRAM limit (< 6144.0 MB) on CUDA in `bfloat16`.
