# backbone-ablation-suite Specification

## Purpose

Provides a scientific 4-way ablation execution framework that profiles memory scaling curves across recurrent horizons, benchmarks inference step latency, measures multi-task convergence stability, and automatically generates publication-ready artifacts.

## Requirements

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
The system SHALL train all 4 parameter-matched baseline models on identical real-data manifests and token budgets under at least 3 configurable matched seeds, logging step-by-step composite losses, individual loss heads, action primitive accuracy, and coordinate MAE.

#### Scenario: Seeded comparative training run
- **WHEN** the 4 baseline architectures are trained using the same real-data manifests, example order, per-run token budget, optimizer schedule, and at least 3 matched seeds
- **THEN** each seed produces comparable learning curves and evaluation metrics, trainable parameter counts are matched within 1% for the primary comparison, and all loss values remain finite without divergence

#### Scenario: Parameter or compute mismatch
- **WHEN** an architecture cannot meet the primary parameter-count tolerance
- **THEN** the run is not labeled a parameter-matched comparison and reports parameter count, training compute, and the mismatch as a separate exploratory result

#### Scenario: Existing Full JED-M2 checkpoint reference
- **WHEN** the step-19,500 Full JED-M2 checkpoint is included in an ablation report
- **THEN** it is labeled as a separately trained reference and is not compared as a matched baseline against variants initialized with fresh weights

### Requirement: Automated Publication Artifact Generation
The system SHALL compile empirical ablation metrics into a report in a unique experiment output directory, producing formatted markdown summary tables, scaling curves, and publication-ready LaTeX table snippets without overwriting prior experiment reports.

#### Scenario: Automated report compilation
- **WHEN** an ablation experiment completes successfully with an explicit run identifier and output directory
- **THEN** a report is generated there containing the four-way parameter breakdown, memory scaling comparisons across horizons, latency percentiles, multi-seed convergence metrics, data provenance, and valid LaTeX `tabular` snippets

#### Scenario: Existing report path collision
- **WHEN** the requested report path already exists and belongs to another run
- **THEN** the system refuses to overwrite it unless an explicit overwrite option is supplied

### Requirement: Reproducible Ablation Run Records
The system SHALL preserve enough configuration and provenance to reproduce and audit each architecture comparison.

#### Scenario: Completed seed run
- **WHEN** a seed run completes or stops with an error
- **THEN** its run record includes model configuration, seed, data and split manifest checksums, source revision, training token budget, optimizer settings, parameter count, resource measurements, metrics, and terminal status

#### Scenario: Aggregate multi-seed results
- **WHEN** all configured seeds finish for all four architectures
- **THEN** the report includes each per-seed result and aggregate mean and dispersion for the declared metrics without hiding failed or incomplete runs
