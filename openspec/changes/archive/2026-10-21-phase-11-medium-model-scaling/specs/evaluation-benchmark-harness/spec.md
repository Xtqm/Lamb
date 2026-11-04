# Spec Delta

## MODIFIED Requirements

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
