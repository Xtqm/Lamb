# Spec Delta: evaluation-benchmark-harness

## MODIFIED Requirements

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

### Requirement: Comparative Checkpoint Benchmarking and Reporting
The system SHALL provide a comparative evaluation runner and CLI script (`eval_benchmarks.py`) supporting configurable model scaling tiers (`--model_size {nano,medium}`), evaluating multiple model checkpoints side-by-side (including Step 2,500 baseline and Phase 14 models), rendering clean ASCII terminal comparison tables, and exporting formal benchmark reports to Markdown (`BENCHMARK_REPORT.md`).

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
