# Spec Delta

## Purpose

Provides a scientific 4-way ablation execution framework that profiles memory scaling curves across recurrent horizons, benchmarks inference step latency, measures multi-task convergence stability, and automatically generates publication-ready artifacts.

## ADDED Requirements

### Requirement: Memory Scaling Profiling Across Rollout Horizons
The system SHALL profile active allocated VRAM, reserved VRAM, and per-token KV-cache memory footprints across recurrent rollout horizons of 16, 32, 64, and 128 turns for all 4 baseline architectures in `bfloat16`.

#### Scenario: Multi-turn rollout memory profiling
- **WHEN** recurrent rollout inference is executed for $T \in \{16, 32, 64, 128\}$ steps across Baseline A (Transformer), Baseline B (Hybrid SSM-Attn), Baseline C (JED-M2 w/o JEPA), and Baseline D (Full JED-M2)
- **THEN** the profiler measures peak allocated memory and cache tensor bytes at each horizon, verifying $O(1)$ constant memory slope ($< 5\%$ variance) for MLA/SSM baselines and $O(N)$ linear cache growth for standard MHA

### Requirement: Inference Step Latency Benchmarking
The system SHALL benchmark per-step rollout latency across all 4 baseline architectures on the host GPU in native `bfloat16`, reporting mean, median, and 95th-percentile (p95) latency in milliseconds per step after CUDA warmup.

#### Scenario: Per-step latency measurement
- **WHEN** 100 consecutive recurrent rollout steps are timed on CUDA with synchronization following 10 warmup steps
- **THEN** the benchmark records per-step latency metrics and asserts that all models achieve mean latency below 50 ms/step without CUDA runtime errors

### Requirement: Controlled Multi-Task Trajectory Convergence Evaluation
The system SHALL train all 4 parameter-matched baseline models on identical trajectory batches under identical seeds for 300 steps, logging step-by-step composite losses, individual loss heads, action primitive accuracy, and coordinate MAE.

#### Scenario: Seeded comparative training run
- **WHEN** the 4 baseline models are trained for 300 steps with identical batch seeds, learning rates, and gradient accumulation schedules
- **THEN** the system logs loss curves and evaluation metrics for each baseline without divergence (loss < 100.0, zero NaNs/Infs)

### Requirement: Automated Publication Artifact Generation
The system SHALL automatically compile empirical ablation metrics into `ABLATION_REPORT.md`, producing formatted markdown summary tables, 7-bit ASCII scaling curves, and publication-ready LaTeX table snippets.

#### Scenario: Automated report compilation
- **WHEN** ablation experiments complete successfully
- **THEN** `ABLATION_REPORT.md` is generated containing the 4-way parameter breakdown, memory scaling comparisons across horizons, latency percentiles, 300-step convergence metrics, and valid LaTeX `tabular` snippets
