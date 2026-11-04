# Spec Delta: evaluation-benchmark-harness

## MODIFIED Requirements

### Requirement: Comparative Checkpoint Benchmarking and Reporting
The system SHALL provide a comparative evaluation runner and CLI script (`eval_benchmarks.py`) supporting configurable model scaling tiers (`--model_size {nano,medium}`), evaluating multiple model checkpoints (including baseline SFT, unified pretrain with Byte-Level BPE, and online GRPO post-trained policies) and external Small-LM baseline models (T5-Small, DistilGPT-2) side-by-side, automatically selecting the appropriate tokenizer based on checkpoint metadata, rendering clean ASCII terminal comparison tables, and exporting formal benchmark reports to Markdown (`BENCHMARK_REPORT.md`).

#### Scenario: Side-by-side checkpoint comparison
- **WHEN** multiple checkpoint paths (e.g. SFT baseline vs. GRPO RL checkpoints vs. Small-LM baselines) are passed to the comparative evaluation runner
- **THEN** the system evaluates all checkpoints on identical benchmark task distributions and outputs a side-by-side comparative metric table.

#### Scenario: Formal Markdown report export
- **WHEN** a benchmark evaluation run completes
- **THEN** the system exports `BENCHMARK_REPORT.md` documenting task configurations, per-checkpoint metrics, latency, and hardware telemetry.

#### Scenario: Benchmark evaluation with configurable model size
- **WHEN** `eval_benchmarks.py` is executed with `--model_size medium`
- **THEN** the runner loads or instantiates the medium architecture configuration, executes inference evaluation passes with $O(1)$ latent state caching, and computes evaluation metrics without memory errors

#### Scenario: Small language model baseline comparison
- **WHEN** comparative evaluation includes T5-Small or DistilGPT-2 baseline models
- **THEN** the benchmark harness evaluates each baseline on identical prompt sequences and reports parameter counts, latency, action accuracy, coordinate precision, and step accuracy in the comparative report

#### Scenario: Phase 14 comparative evaluation report
- **WHEN** `eval_benchmarks.py` is run comparing Step 2,500 checkpoint against Phase 14 with active JEPA planning
- **THEN** the report reflects throughput gains from parallel Blelloch scan, action accuracy on Mind2Web test splits, and planning improvements from JEPA rollouts

#### Scenario: Real-web validation benchmark execution
- **WHEN** evaluating on the `real_web_val` split
- **THEN** the harness evaluates action accuracy, F1, coordinate MAE, and tolerance accuracy at $\epsilon \le 0.05$ across all validation episodes

#### Scenario: Tokenizer auto-detection for BPE and legacy checkpoints
- **WHEN** loading checkpoints with distinct vocabulary configurations (e.g. BPE-based unified pretrain/RL checkpoints vs. legacy SFT checkpoints)
- **THEN** the runner automatically detects vocabulary size or checkpoint metadata and instantiates the matching tokenizer (`ActionBPETokenizer` or `SimpleLAMTokenizer`)
