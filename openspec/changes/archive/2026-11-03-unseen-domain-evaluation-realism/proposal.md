# Proposal: Official Split Ingestion & Real Web Evaluation Realism

## Why

As established in the empirical and publication readiness audit, JED-M2's current evaluation harness reports metrics that partly rely on synthetic fixtures or mock HTML templates (e.g. `mock_retail_001`, `mock_flight_002` in `RealTaskDOMEnvironment`), and evaluation benchmarks are not cleanly aligned with the three official Mind2Web benchmark evaluation protocols:
1. **Cross-Task**: Unseen tasks executed on seen websites.
2. **Cross-Website**: Unseen websites belonging to seen domains.
3. **Cross-Domain**: Entirely unseen domains with zero overlap.

While `data/manifests/real_mind2web_v1/manifest.json` established 51 train, 11 validation, and 11 test disjoint sites, `RealTaskDOMEnvironment` and the online benchmark harness did not consume this manifest, creating a divergence between offline replay and interactive simulation. This change unifies real dataset loading across all evaluation engines, eliminates mock template fallbacks in evaluation modes, and introduces split-aware evaluation reporting.

## What Changes

- **Manifest-Driven Real Environment**: Update `RealTaskDOMEnvironment` in `jed_m2/real_environment.py` to ingest real tasks directly from `data/manifests/real_mind2web_v1/manifest.json` filtered by split (`train`, `val`, `test`), completely removing reliance on synthetic mock fallback tasks during evaluation.
- **Official Generalization Splits**: Classify manifest tasks into the three official Mind2Web generalization splits (`cross_task`, `cross_website`, `cross_domain`) and index tasks by domain and website hierarchy.
- **Strict Disjointness Verification**: Enforce programmatic validation in the evaluation harness ensuring zero site or snapshot leakage between train, validation, and test splits.
- **Generalization Benchmark Reporting**: Update `TaskEvaluator`, `BenchmarkSummary`, `format_ascii_table()`, and `export_markdown_report()` in `jed_m2/benchmark.py` to evaluate and report per-split performance (In-Domain, Cross-Website, Cross-Domain) side-by-side with Native `FINISH` F1 and Autonomous Success Rate (ASR).
- **Benchmark CLI Integration**: Update `eval_benchmarks.py` to support `--split {train,val,test,cross_task,cross_website,cross_domain}` and `--real_only` flags.

## Capabilities

### Modified Capabilities

- `real-web-trajectory-dataset`: Update `RealTaskDOMEnvironment` to load authentic tasks from verified manifest splits (`train`, `val`, `test`) without synthetic fallbacks, and expose domain and website partitioning metadata.
- `evaluation-benchmark-harness`: Add multi-split generalization benchmarking across `cross_task`, `cross_website`, and `cross_domain` splits, enforce site disjointness verification, and document generalization gaps in benchmark summary tables.

## Impact

- Affected files:
  - `jed_m2/real_environment.py`: `RealTaskDOMEnvironment` initialization and task loading.
  - `jed_m2/real_dataset.py`: Manifest indexing and split metadata helper methods.
  - `jed_m2/benchmark.py`: Benchmark evaluation routines, split metadata logging, table formatting.
  - `eval_benchmarks.py`: CLI arguments and evaluation pipeline.
- Dependencies: Pure PyTorch and Python standard library; zero external binary or cloud dependencies. Peak VRAM remains strictly $< 6.0$ GB on RTX 5060 Laptop GPU.
