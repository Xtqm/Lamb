# Tasks: Phase 14 - Real-World Mind2Web Corpus Ingestion, Pure-PyTorch Blelloch Associative Scan Acceleration, and Latent JEPA Rollout Planning

## 1. Git Hygiene & Mind2Web Dataset Downloader

- [x] 1.1 Update `.gitignore` to strictly exclude `data/raw_mind2web/`, `data/*.parquet`, `data/*_cache/`, and raw HTML snapshots, and verify `git status --ignored` correctly detects data ignore rules.
- [x] 1.2 Implement `data/download_mind2web.py` to stream official Mind2Web (`osunlp/Mind2Web`) task splits (train, val, test) and unpack raw DOM snapshots into `data/raw_mind2web/`, verifying with a test streaming download of sample tasks.

## 2. Real-World Mind2Web Dataset Refactoring

- [x] 2.1 Refactor `RealWebTrajectoryDataset` in `jed_m2/real_dataset.py` to parse genuine Mind2Web task annotations, extract interactive candidates via `FastDOMParser`, normalize bounding box coordinates $[0, 1]^2$, and map actions to `ActionType`, verifying element parsing on sample Mind2Web records.
- [x] 2.2 Implement procedural HTML template fallback in `RealWebTrajectoryDataset` when `data/raw_mind2web/` is not present, and verify initialization and collation succeed with both real and fallback modes.

## 3. Pure-PyTorch Parallel Prefix Scan (Blelloch Scan) in Mamba SSM

- [x] 3.1 Implement native PyTorch parallel associative scan `parallel_associative_scan` in `jed_m2/mamba.py` using logarithmic prefix scan operations over the sequence dimension without external compiled packages.
- [x] 3.2 Integrate `parallel_associative_scan` into `MambaBlock.forward()`, verifying mathematical parity ($< 1\text{e-}5$ difference against sequential loop in float32) and finite gradient backpropagation.
- [x] 3.3 Benchmark forward and backward throughput at $T=1024$ and $T=2048$, verifying throughput exceeds $1,200$ tokens/second on CUDA in bfloat16.

## 4. Active Latent JEPA Rollout Planning in JEDM2Agent

- [x] 4.1 Implement `predict_next_latent_state` and candidate action rollout evaluation in `JEDM2Agent` (`jed_m2/agent.py`), computing cosine distance to goal embedding $z_{\text{goal}}$.
- [x] 4.2 Integrate predictive re-ranking into `JEDM2Agent.act()`, scoring candidate actions by combining policy logits with negative predictive goal distance, and verify action selection on test observations.

## 5. Benchmarking & Verification Suite

- [x] 5.1 Update `eval_benchmarks.py` to support real Mind2Web offline dataset evaluation and comparative reporting against Step 2,500 checkpoint baseline.
- [x] 5.2 Implement `verify_phase14.py` verifying Blelloch scan parity, real Mind2Web streaming and collation under 6.0 GB VRAM, JEPA rollout action selection, and export of comparative evaluation reports.
- [x] 5.3 Execute `verify_phase14.py` and confirm all 4 pillar checks pass with zero errors under the 6.0 GB VRAM ceiling.

## Workflow follow-up

- Review artifacts with the user before applying changes.
- Execute `/openspec-apply-change` to implement Phase 14 tasks upon user request.
