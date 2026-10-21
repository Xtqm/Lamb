# evaluation-benchmark-harness Specification

## Purpose
Provides a standardized evaluation benchmark harness for Web and OS agent tasks, computing formal task completion, step accuracy, coordinate precision, trajectory efficiency, latency metrics, and enabling comparative multi-checkpoint evaluation.

## Requirements

### Requirement: Standardized Action and Trajectory Metrics
The system SHALL compute formal action and trajectory evaluation metrics across agent execution runs, including Task Success Rate (SR %), Step Success Rate, Action Primitive F1, Spatial Coordinate Mean Absolute Error (MAE), Spatial Coordinate Tolerance Accuracy ($\epsilon \le 0.05$), Trajectory Efficiency Ratio, and Decision Latency.

#### Scenario: Metric computation on completed episode
- **WHEN** an evaluation episode finishes with predicted actions and ground truth steps
- **THEN** the system returns a metric record containing SR in $[0, 1]$, Step SR in $[0, 1]$, F1 in $[0, 1]$, MAE $\ge 0$, tolerance accuracy in $[0, 1]$, trajectory efficiency in $[0, 1]$, and decision latency in milliseconds per step.

#### Scenario: Metric numerical sanity guarantee
- **WHEN** metrics are computed across any number of evaluated episodes
- **THEN** all computed metric values are valid floating-point numbers containing zero NaNs and zero Infs.

### Requirement: Offline Dataset Replay Evaluation
The system SHALL support offline evaluation by replaying trajectory datasets (such as Mind2Web and WebArena splits) against ground-truth element bounding boxes, continuous coordinates, and discrete action primitives.

#### Scenario: Offline evaluation of Mind2Web trajectory split
- **WHEN** an offline trajectory dataset is provided to the evaluator
- **THEN** the model steps through observations and predictions are evaluated against ground-truth primitives and normalized screen coordinates without launching a live browser.

#### Scenario: Spatial coordinate tolerance matching
- **WHEN** evaluating continuous pointer click coordinates against ground truth
- **THEN** predictions within Euclidean distance $\epsilon \le 0.05$ or within the target element bounding box are scored as correct spatial matches.

### Requirement: Online Closed-Loop Environment Evaluation
The system SHALL support online closed-loop multi-turn agent evaluation across interactive environments, executing live actions in `WebOSEnvironment` sandboxes and live Playwright browser sessions until task termination or maximum step limit.

#### Scenario: Closed-loop multi-turn simulated evaluation
- **WHEN** evaluating a model checkpoint on simulated Web/OS sandbox tasks
- **THEN** the agent interacts multi-turn with `WebOSEnvironment`, tracking step success and overall task success dynamically.

#### Scenario: Closed-loop live Playwright evaluation
- **WHEN** evaluating a model checkpoint on live Playwright browser tasks
- **THEN** the agent navigates real web pages, captures screenshots and DOM snapshots, dispatches browser primitives, and records task completion metrics.

### Requirement: Comparative Checkpoint Benchmarking and Reporting
The system SHALL provide a comparative evaluation runner and CLI script (`eval_benchmarks.py`) supporting configurable model scaling tiers (`--model_size {nano,medium}`), evaluating multiple model checkpoints side-by-side, rendering clean ASCII terminal comparison tables, and exporting formal benchmark reports to Markdown (`BENCHMARK_REPORT.md`).

#### Scenario: Side-by-side checkpoint comparison
- **WHEN** multiple checkpoint paths (e.g. SFT baseline vs. GRPO RL checkpoints) are passed to the comparative evaluation runner
- **THEN** the system evaluates all checkpoints on identical benchmark task distributions and outputs a side-by-side comparative metric table.

#### Scenario: Formal Markdown report export
- **WHEN** a benchmark evaluation run completes
- **THEN** the system exports `BENCHMARK_REPORT.md` documenting task configurations, per-checkpoint metrics, latency, and hardware telemetry.

#### Scenario: Benchmark evaluation with configurable model size
- **WHEN** `eval_benchmarks.py` is executed with `--model_size medium`
- **THEN** the runner loads or instantiates the medium architecture configuration, executes inference evaluation passes with $O(1)$ latent state caching, and computes evaluation metrics without memory errors

### Requirement: Hardware and VRAM Constraint Adherence
The evaluation harness SHALL execute with gradient calculation disabled (`torch.no_grad()`), supporting `torch.bfloat16` precision, and strictly adhere to a peak GPU VRAM allocation ceiling of 6.0 GB (6144.0 MB).

#### Scenario: VRAM allocation during evaluation
- **WHEN** the benchmark evaluation harness runs on CUDA
- **THEN** peak allocated GPU memory does not exceed 6144.0 MB.
