# Proposal: Phase 8 Standardized Evaluation Benchmark Harness

## Why

With JED-M2's core architecture, action heads, SFT convergence baseline, online GRPO reinforcement learning pipeline, multimodal visual grounding, and live Playwright browser integration complete (Phases 1 through 7), the system requires a rigorous, standardized evaluation benchmark harness. 

Without standardized offline dataset replay (e.g., Mind2Web/WebArena ground-truth matching) and online closed-loop environment execution (`WebOSEnvironment` and live Playwright sessions), there is no objective, automated mechanism to compute formal action metrics (Task Success Rate, Step Success Rate, Action Primitive F1, Spatial Coordinate Accuracy, Trajectory Efficiency, and Decision Latency) or perform side-by-side comparative evaluations across model checkpoints (such as Base SFT vs. GRPO Step 40 vs. Converged RL policies). Phase 8 establishes this standardized benchmarking infrastructure.

## What Changes

- **Benchmark Evaluation Module (`jed_m2/benchmark.py`)**:
  - Implement `TaskEvaluator` for computing formal action and trajectory metrics:
    - **Task Success Rate (SR %)**: Exact task goal completion.
    - **Step Success Rate**: Fraction of valid intermediate sub-goals achieved across episode steps.
    - **Action Primitive F1**: Multi-class precision, recall, and micro/macro F1 scores across discrete action categories (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH).
    - **Spatial Coordinate Accuracy**: Continuous pointer Mean Absolute Error (MAE) and Intersection-over-Union (IoU) / tolerance radius accuracy ($\epsilon \le 0.05$ normalized distance).
    - **Trajectory Efficiency Ratio**: Normalized trajectory length compared to optimal demonstration baseline ($L_{\text{optimal}} / L_{\text{agent}}$).
    - **Decision Latency & Throughput**: Milliseconds per decision step and token throughput during inference.
  - Support **Offline Dataset Evaluation**: Replaying Mind2Web and WebArena test/validation trajectory splits with element bounding box and continuous coordinate matching against ground truth.
  - Support **Online Closed-Loop Evaluation**: Multi-turn agent rollout against `WebOSEnvironment` sandboxes and live Playwright web sessions.
- **Evaluation CLI Runner (`eval_benchmarks.py`)**:
  - Implement command-line interface with flags `--benchmark {mind2web,webarena,simulated}`, `--model_path <path>`, `--split {val,test}`, `--num_episodes <n>`, `--max_steps <n>`, and `--device {cuda,cpu}`.
  - Implement comparative evaluation engine executing multiple checkpoints side-by-side (e.g. Base SFT vs. RL checkpoints).
  - Generate clean ASCII terminal comparison tables and export comprehensive benchmark reports to Markdown (`BENCHMARK_REPORT.md`).
- **Module Exports (`jed_m2/__init__.py`)**:
  - Export `TaskEvaluator`, `BenchmarkSuite`, `evaluate_offline_dataset`, `evaluate_online_environment`, and comparative runner utilities.
- **Verification Suite (`verify_phase8.py`)**:
  - Verify offline dataset replay and online closed-loop benchmark execution.
  - Validate numerical sanity of all metric calculations (SR in $[0, 1]$, Step SR in $[0, 1]$, F1 in $[0, 1]$, MAE $\ge 0$, no NaNs/Infs).
  - Validate side-by-side multi-checkpoint comparative evaluation and report generation.
  - Profile peak GPU VRAM allocation during evaluation to guarantee strict compliance with the 6.0 GB ceiling (< 6144.0 MB).

## Capabilities

### New Capabilities
- `evaluation-benchmark-harness`: Standardized evaluation benchmark harness for Web and OS agent tasks, providing formal action metrics (SR, Step SR, Action F1, Spatial MAE/tolerance, Efficiency Ratio, Latency), offline dataset trajectory replay, online closed-loop multi-turn agent evaluation, multi-checkpoint comparative benchmarking, and automated Markdown report generation.

### Modified Capabilities
<!-- None -->

## Impact

- **Affected Files**:
  - `jed_m2/benchmark.py` (New): Standardized evaluation harness and `TaskEvaluator`.
  - `eval_benchmarks.py` (New): CLI evaluation and comparative reporting script.
  - `verify_phase8.py` (New): Verification script for Phase 8.
  - `jed_m2/__init__.py` (Modified): Exports new evaluation classes and functions.
- **Hardware & Memory Impact**:
  - Evaluation operates in `torch.bfloat16` and evaluation mode (`model.eval()`, `torch.no_grad()`).
  - Inference memory footprint will remain well under the 6.0 GB VRAM limit (expected < 1.0 GB).
- **Breaking Changes**: None. Fully backward-compatible with all existing models, environments, and datasets.
