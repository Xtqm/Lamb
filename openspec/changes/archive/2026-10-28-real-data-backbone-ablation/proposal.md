# Proposal: Real-Data, Site-Held-Out Backbone Ablation

## Why

The current ablation harness trains for 300 steps on synthetic/repeated trajectories and reports mostly step-level metrics. That can smoke-test training code, but it cannot establish that JED-M2 learns from real web interactions or improves real-site task completion. The SFT checkpoint at step 19,500 also cannot be compared fairly against newly initialized ablation baselines. A reproducible, diverse real-data comparison is needed before using the results to support architecture claims.

## What Changes

- Audit and ingest authentic interaction traces with source, license, site/domain, split, and content-hash provenance. The primary cohort must contain captured observations and actions; task definitions without interaction traces, procedural templates, and reconstructed synthetic DOMs remain explicitly labeled and excluded from the real-data cohort. The run must fail clearly when real-only mode has no usable data rather than silently falling back to synthetic data.
- Build deterministic train/validation/test manifests grouped by website, with duplicate-task and snapshot leakage checks. Use the full locally available licensed corpus and report task counts, site counts, task categories, and source coverage. Add additional real trace sources only when their raw observations and actions are available and their use is permitted.
- Preserve Mind2Web's recorded CLICK, TYPE, and SELECT operations as distinct labels by adding SELECT support to the ablation action head. Keep the step-19,500 checkpoint on its existing six-action head as a separately labeled legacy reference.
- Run all four architectures from a comparable initialization regime on identical real-data manifests, token/update budgets, optimizer settings, and evaluation conditions. Match trainable parameter counts where practical and report parameter and compute differences. Give each of at least 3 configurable seeds three full passes over the unique real training set. Do not treat the step-19,500 Full JED-M2 checkpoint as a trained baseline against newly initialized models.
- Make the primary comparison multi-seed (at least three configurable seeds), retain learning curves and per-seed results, and evaluate on site-held-out real trajectories. Report trajectory/step match, action type and target-element hit rate, coordinate error/tolerance, and uncertainty across seeds. Keep offline trajectory replay distinct from closed-loop task completion; label the environment and data provenance for each result.
- Save each run under a unique experiment directory with its config, seed, data/split manifests, source revision, hardware/resource measurements, metrics, and report. Do not overwrite the existing `ABLATION_REPORT.md`; make aggregation an explicit output path.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `real-web-trajectory-dataset`: add provenance-aware real-only ingestion, website-grouped leakage-resistant manifests, corpus diversity reporting, and strict handling when authentic traces are unavailable.
- `backbone-ablation-suite`: replace the synthetic 300-step-only comparison with fair, multi-seed training on shared real-data manifests and isolated reproducible outputs; report parameter/compute budgets and site-held-out results.
- `dual-system-action-heads`: add a distinct SELECT primitive to new ablation model configurations while retaining legacy six-action checkpoint loading.
- `evaluation-benchmark-harness`: add site-held-out real-data evaluation and clearly distinguish offline trajectory metrics from closed-loop task completion and synthetic-environment results.

## Impact

The change will affect the real-data loaders and downloader, action type configuration and decoding, `run_ablation.py`, the comparative evaluation runner, and the four capability specs above. Full corpus coverage depends on obtaining the raw datasets with permitted use; the proposal does not assume that task-definition files or synthetic HTML generation constitute real interaction data. The result will be evidence about these four implementations under the documented data and compute budget, not a claim that any model is generally ready for unrestricted live-web use.
