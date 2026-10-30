# Tasks: Realistic Distractor Grounding and Benchmark Alignment

## 1. Candidate Pool and Distractor Ingestion

- [x] 1.1 Ingest both `pos_candidates` and `neg_candidates` into `bbox_map` in `RealWebTrajectoryDataset._generate_from_real_manifest` (`jed_m2/real_dataset.py`), verifying turns now contain candidate pools with 10–64 bounding-box-annotated elements.
- [x] 1.2 Update real-only dataset cache version tag to `v5` in `_get_cache_path` (`jed_m2/real_dataset.py`) to trigger automatic cache refresh of real test split with distractor geometries.
- [x] 1.3 Add unit test in `tests/test_candidate_snapping.py` verifying that parsed turns expose multi-element candidate pools where distractor elements have valid bounding boxes and centers.

## 2. Benchmark Re-Evaluation and Reporting Alignment

- [x] 2.1 Re-evaluate `jed_m2_nano_pretrain_final.pt` on the 147 site-held-out Mind2Web test tasks with multi-element distractor snapping enabled, saving raw and snapped metrics to `runs/ablation/post_pretrain_evaluation.json`.
- [x] 2.2 Update `POST_PRETRAIN_BENCHMARK_REPORT.md` to reflect honest distractor-snapped metrics alongside the unassisted raw continuous coordinate baseline (74.58% Target Hit, 0.0151 MAE) and transparent argument generation disclosure (2.17%).
- [x] 2.3 Run full test suite (`pytest tests/`) to ensure 100% pass rate with zero regressions.

## Workflow follow-up

- Review verified metrics with user before initiating online GRPO fine-tuning (`train_rl.py`).
- Archive the change once implementation and evaluation benchmarks are complete.
