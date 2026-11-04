# Design: Official Split Ingestion & Real Web Evaluation Realism

## Context

In JED-M2, `data/manifests/real_mind2web_v1/manifest.json` defines a pairwise-disjoint website partitioning across 1,009 authentic Mind2Web tasks (718 train, 144 val, 147 test across 73 unique websites). However, `RealTaskDOMEnvironment` in `jed_m2/real_environment.py` historically operated independently of this manifest, falling back to two hardcoded mock tasks (`mock_retail_001` and `mock_flight_002`) when unconfigured.

Furthermore, academic web agent evaluation standards (Mind2Web, WebArena) require evaluating generalization across three distinct distributional boundaries:
1. `Cross-Task`: Same website, unseen instruction goal.
2. `Cross-Website`: Same high-level domain, unseen website DOM structure.
3. `Cross-Domain`: Completely unseen website and domain hierarchy.

See `proposal.md` for background and motivation.

## Goals / Non-Goals

**Goals:**
- Connect `RealTaskDOMEnvironment` directly to `data/manifests/real_mind2web_v1/manifest.json`, accepting `split: str = "test"` and `manifest_path: Optional[str] = None`.
- Completely eliminate mock fallback fixtures when evaluating in real-only mode (`real_only=True` or `split in ("train", "val", "test")`).
- Implement `partition_generalization_splits()` to categorize manifest test tasks into official Mind2Web generalization splits: `cross_task`, `cross_website`, and `cross_domain`.
- Enhance `BenchmarkSummary` and table formatters in `jed_m2/benchmark.py` to report generalization performance breakdowns side-by-side with Native `FINISH` F1 and ASR.
- Add split selection flags (`--split {test,cross_task,cross_website,cross_domain}`) and `--manifest` support to `eval_benchmarks.py`.

**Non-Goals:**
- Scraping new live websites or re-downloading Mind2Web: the local verified corpus of 1,009 tasks and raw snapshots is already present on disk.
- Changing model weights or recurrent layer shapes.

## Decisions

### Decision 1: Manifest-Driven Task Loading in `RealTaskDOMEnvironment`
- **Choice:** In `jed_m2/real_environment.py`, update `RealTaskDOMEnvironment.__init__` to accept `manifest_path: Optional[str] = "data/manifests/real_mind2web_v1/manifest.json"` and `split: str = "test"`. When a manifest is provided, tasks are loaded exclusively from the manifest index.
- **Rationale:** Ensures that interactive rollouts, RL exploration, and closed-loop evaluation all operate on identical verified data distributions without divergence.
- **Alternatives Considered:**
  - *Hardcode paths inside the class*: Rejected because it prevents testing custom manifests or ablation splits.

### Decision 2: Generalization Split Categorization
- **Choice:** Provide helper `get_generalization_splits(manifest)` in `jed_m2/real_dataset.py`:
  - `cross_website`: Tasks in test split whose website is absent from train split, but domain is present in train split.
  - `cross_domain`: Tasks in test split whose domain is completely absent from train split.
  - `cross_task`: Tasks in test split whose website is present in train split.
- **Rationale:** Aligns JED-M2 directly with official Mind2Web published benchmark splits.

### Decision 3: Zero-Fallback Enforcement in Real Mode
- **Choice:** If `RealTaskDOMEnvironment` cannot locate raw Mind2Web JSONs or HTML snapshots for a manifest task, it raises a strict `FileNotFoundError` or validation error rather than silently substituting mock tasks.
- **Rationale:** Eliminates the risk of reporting synthetic performance as real-world agent capability.

### Decision 4: Generalization Table Reporting
- **Choice:** Update `TaskEvaluator.format_ascii_table()` and `export_markdown_report()` in `jed_m2/benchmark.py` to display split tags (e.g. `[Cross-Website]`, `[Cross-Domain]`) alongside macro metrics.

## Risks / Trade-offs

- **[Risk] Generalization Degradation on Unseen Domains:**
  - *Mitigation:* It is scientifically normal for small language models to experience performance drop on out-of-distribution websites. Transparently reporting this drop is essential for peer-reviewed academic rigor.
- **[Risk] Slower Evaluation Over Hundreds of Authentic DOM Snapshots:**
  - *Mitigation:* Maintain `--max_episodes` flag to allow rapid diagnostic sweeps while preserving the option for full test set evaluation.

## Migration Plan

1. Update `RealTaskDOMEnvironment` in `jed_m2/real_environment.py` with manifest loading and split filtering.
2. Update `jed_m2/real_dataset.py` with split categorization utilities.
3. Update `jed_m2/benchmark.py` to support generalization split evaluation.
4. Update `eval_benchmarks.py` CLI.
5. Add unit tests in `tests/test_real_environment.py` and run full test suite.
