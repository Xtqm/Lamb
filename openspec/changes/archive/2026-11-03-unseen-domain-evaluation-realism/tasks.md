# Tasks: Official Split Ingestion & Real Web Evaluation Realism

## 1. Real Environment Manifest Integration & Mock Elimination

- [x] 1.1 Update `RealTaskDOMEnvironment` in `jed_m2/real_environment.py` to accept `manifest_path` and `split: str = "test"` and load tasks directly from verified manifest task entries and verify with unit tests
- [x] 1.2 Remove mock template fallbacks from `RealTaskDOMEnvironment` when operating in real mode, strictly raising descriptive errors if raw data or snapshots are missing
- [x] 1.3 Add unit tests in `tests/test_real_environment.py` verifying manifest-driven task loading and zero mock fallback behavior

## 2. Generalization Split Partitioning & Provenance Indexing

- [x] 2.1 Implement `get_generalization_splits()` in `jed_m2/real_dataset.py` to partition test manifest tasks into `cross_task`, `cross_website`, and `cross_domain` subsets
- [x] 2.2 Add unit tests in `tests/test_real_dataset.py` verifying site-disjoint partitioning and accurate task categorization
- [x] 2.3 Add site disjointness assertion utility in `jed_m2/real_dataset.py` to programmatically verify zero train/val/test site overlap

## 3. Generalization Benchmark Runner & CLI Integration

- [x] 3.1 Update `evaluate_offline_dataset` and `evaluate_online_environment` in `jed_m2/benchmark.py` to record split metadata and compute multi-split metrics
- [x] 3.2 Update `format_ascii_table()` and `export_markdown_report()` in `jed_m2/benchmark.py` to output split breakdowns (In-Domain vs Cross-Website vs Cross-Domain)
- [x] 3.3 Update `eval_benchmarks.py` CLI to support `--split {test,cross_task,cross_website,cross_domain}` and `--manifest` flags

## 4. End-to-End Verification & Generalization Audit

- [x] 4.1 Run the complete test suite (`pytest tests/`) and verify all tests pass with zero regressions
- [x] 4.2 Run real web benchmark evaluation across held-out splits to produce and verify official generalization metrics table
