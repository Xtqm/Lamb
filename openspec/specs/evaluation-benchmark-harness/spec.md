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
The system SHALL support offline evaluation by replaying trajectory datasets, specifically including official held-out Mind2Web test splits across real-world domains and synthetic traces, against ground-truth element bounding boxes, continuous coordinates, and discrete action primitives.

#### Scenario: Offline evaluation of Mind2Web trajectory split
- **WHEN** an offline trajectory dataset is provided to the evaluator
- **THEN** the model steps through observations and predictions are evaluated against ground-truth primitives and normalized screen coordinates without launching a live browser.

#### Scenario: Spatial coordinate tolerance matching
- **WHEN** evaluating continuous pointer click coordinates against ground truth
- **THEN** predictions within Euclidean distance $\epsilon \le 0.05$ or within the target element bounding box are scored as correct spatial matches.

#### Scenario: Held-out real-world Mind2Web benchmark evaluation
- **WHEN** evaluating on the held-out Mind2Web test split containing real-world web domains
- **THEN** the benchmark harness computes element ranking accuracy, action type accuracy, coordinate MAE, and overall step success rate under strict deterministic evaluation conditions

### Requirement: Online Closed-Loop Environment Evaluation
The system SHALL support online closed-loop multi-turn agent evaluation across interactive environments, executing live actions in `WebOSEnvironment` sandboxes and live Playwright browser sessions until task termination or maximum step limit.

#### Scenario: Closed-loop multi-turn simulated evaluation
- **WHEN** evaluating a model checkpoint on simulated Web/OS sandbox tasks
- **THEN** the agent interacts multi-turn with `WebOSEnvironment`, tracking step success and overall task success dynamically.

#### Scenario: Closed-loop live Playwright evaluation
- **WHEN** evaluating a model checkpoint on live Playwright browser tasks
- **THEN** the agent navigates real web pages, captures screenshots and DOM snapshots, dispatches browser primitives, and records task completion metrics.

### Requirement: Comparative Checkpoint Benchmarking and Reporting
The system SHALL provide a comparative evaluation runner and CLI script (`eval_benchmarks.py`) supporting configurable model scaling tiers (`--model_size {nano,medium}`), evaluating multiple model checkpoints side-by-side (including Step 15,800 SFT and final online GRPO-trained checkpoints), rendering clean ASCII terminal comparison tables, and exporting formal benchmark reports to Markdown (`BENCHMARK_REPORT.md`).

#### Scenario: Side-by-side checkpoint comparison
- **WHEN** multiple checkpoint paths (e.g. SFT baseline vs. GRPO RL checkpoints) are passed to the comparative evaluation runner
- **THEN** the system evaluates all checkpoints on identical benchmark task distributions and outputs a side-by-side comparative metric table.

#### Scenario: Formal Markdown report export
- **WHEN** a benchmark evaluation run completes
- **THEN** the system exports `BENCHMARK_REPORT.md` documenting task configurations, per-checkpoint metrics, latency, and hardware telemetry.

#### Scenario: Benchmark evaluation with configurable model size
- **WHEN** `eval_benchmarks.py` is executed with `--model_size medium`
- **THEN** the runner loads or instantiates the medium architecture configuration, executes inference evaluation passes with $O(1)$ latent state caching, and computes evaluation metrics without memory errors

#### Scenario: Phase 14 comparative evaluation report
- **WHEN** `eval_benchmarks.py` is run comparing Step 2,500 checkpoint against Phase 14 with active JEPA planning
- **THEN** the report reflects throughput gains from parallel Blelloch scan, action accuracy on Mind2Web test splits, and planning improvements from JEPA rollouts

#### Scenario: Real-web validation benchmark execution
- **WHEN** evaluating on the `real_web_val` split
- **THEN** the harness evaluates action accuracy, F1, coordinate MAE, and tolerance accuracy at $\epsilon \le 0.05$ across all validation episodes

### Requirement: Hardware and VRAM Constraint Adherence
The evaluation harness SHALL execute with gradient calculation disabled (`torch.no_grad()`), supporting `torch.bfloat16` precision, and strictly adhere to a peak GPU VRAM allocation ceiling of 6.0 GB (6144.0 MB).

#### Scenario: VRAM allocation during evaluation
- **WHEN** the benchmark evaluation harness runs on CUDA
- **THEN** peak allocated GPU memory does not exceed 6144.0 MB.

### Requirement: Site-Held-Out Real Trajectory Evaluation
The system SHALL evaluate ablation checkpoints on real trajectory test manifests whose websites are disjoint from the corresponding training and validation manifests.

#### Scenario: Evaluate a held-out real-site manifest
- **WHEN** a checkpoint is evaluated on a real-data test manifest
- **THEN** the harness verifies site disjointness and reports action type accuracy, target-element hit rate, coordinate MAE and tolerance accuracy, step accuracy, and exact full-trajectory match rate for the complete test set and by website, with macro-average-by-website and micro-average results

#### Scenario: Invalid or leaking test manifest
- **WHEN** a test manifest is missing, contains synthetic examples in a real-only evaluation, or shares a website or snapshot with training or validation
- **THEN** the harness stops the real-data evaluation and identifies the manifest validation failure

### Requirement: Evaluation Provenance and Protocol Separation
The system SHALL identify the data source, split, website set, checkpoint, run identifier, and evaluation protocol for every reported result.

#### Scenario: Report offline trace replay
- **WHEN** predictions are scored by replaying recorded real trajectories
- **THEN** the result is labeled offline trace replay and is not presented as closed-loop task completion

#### Scenario: Compare different environment types
- **WHEN** results from real trace replay, synthetic tasks, or closed-loop browser environments appear in one report
- **THEN** each protocol is reported in a separate section with its own metrics and provenance, without combining the results into a single score

### Requirement: Dual-Mode Coordinate Benchmark Reporting
The offline evaluation harness SHALL support dual-mode coordinate evaluation, reporting both unconstrained continuous coordinate metrics and DOM candidate-snapped metrics across target element hit rate, coordinate tolerance accuracy, coordinate MAE, step accuracy, and exact trajectory match rate.

#### Scenario: Reporting raw and candidate-snapped metrics in parallel
- **WHEN** the offline evaluation harness replays context windows from real-world trajectory datasets
- **THEN** it computes and logs both unconstrained continuous coordinate metrics and DOM candidate-snapped metrics side-by-side in evaluation summaries

#### Scenario: Preserving backwards-compatible offline scoring
- **WHEN** offline evaluation is executed with snapping disabled or candidate elements omitted
- **THEN** scoring defaults to standard continuous coordinate tolerance hit-testing without regression
