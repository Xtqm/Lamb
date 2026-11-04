# Proposal: Realistic Distractor Grounding and Benchmark Alignment

## Why

An adversarial scientific audit of commit `91491de` revealed that our candidate bounding box snapping evaluation suffered from a critical data leakage artifact: `RealWebTrajectoryDataset` populated `bbox_map` strictly from `action["pos_candidates"]` (the human-annotated positive target bounding boxes), completely omitting `action["neg_candidates"]` (the 50–200 distractor elements on the page). As a result, in 100% of the turns where bounding boxes existed, the candidate pool had a size of exactly $N=1$, artificially inflating the reported "Snapped Target Element Hit Rate" to 100.00% because snapping acted as an element-presence gate rather than testing true element discrimination among competing on-screen elements.

Furthermore, multi-token argument generation exact match remains at 2.17% (3/138) and closed-loop interactive task completion sits at 0.0% due to missing `FINISH` token emission. Before advancing to online reinforcement learning (GRPO) and manuscript drafting, we must rectify the candidate pool to include genuine distractor elements (`neg_candidates`), re-evaluate true element discrimination across the 147 site-held-out test tasks, and align our benchmark reporting with empirical publication standards.

## What Changes

- **Full-Page Distractor Candidate Ingestion**: Modify `RealWebTrajectoryDataset` in `jed_m2/real_dataset.py` to ingest and bind bounding boxes from both `pos_candidates` and `neg_candidates`, creating realistic candidate pools containing all 20–64 visible interactive DOM elements per turn.
- **True Multi-Element Candidate Snapping Evaluation**: Update `jed_m2/real_eval.py` and evaluation suites to measure candidate snapping accuracy and target element hit rate against the full set of distractor elements, preventing $N=1$ trivial snapping.
- **Dual-Reporting Benchmark Transparency**: Update `POST_PRETRAIN_BENCHMARK_REPORT.md` and benchmark result schemas to feature unassisted raw continuous coordinates (74.58% Target Hit Rate, 0.0151 MAE) as the primary baseline, alongside true distractor-snapped metrics and explicit disclosure of argument token matching (2.17%).
- **Interactive Termination Alignment**: Update policy evaluation and rollout preparation to monitor and incentivize `ActionType.FINISH` emission for closed-loop task completion.

## Capabilities

### New Capabilities
*(None; this change aligns existing dataset and evaluation capabilities).*

### Modified Capabilities
- `real-web-trajectory-dataset`: Extract and register bounding boxes for both positive target candidates and negative distractor candidates (`neg_candidates`) in parsed HTML DOM representations.
- `evaluation-benchmark-harness`: Evaluate candidate coordinate snapping against multi-element candidate pools (including real page distractors) and report raw vs distractor-snapped metrics side-by-side.

## Impact

- **Code Affected**:
  - `jed_m2/real_dataset.py`: Candidate bounding box collection in `_generate_from_real_manifest` and chunk loaders.
  - `jed_m2/real_eval.py`: Candidate element lookup and distractor snapping metrics.
  - `runs/ablation/post_pretrain_evaluation.json`: Regenerated with realistic candidate pools.
  - `POST_PRETRAIN_BENCHMARK_REPORT.md`: Updated with rigorous, defensible empirical findings.
- **Dependencies**: No external dependencies added; native PyTorch on RTX 5060 Laptop GPU.
- **Breaking Changes**: None. Cache keys for real-only datasets will reflect the augmented bounding boxes, triggering a clean cache rebuild.
