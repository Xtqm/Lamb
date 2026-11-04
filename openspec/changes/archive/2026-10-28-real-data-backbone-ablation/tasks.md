# Tasks

## 1. Real-Data Intake and Manifest

- [x] 1.1 Make Mind2Web acquisition and fixture generation explicit modes; real mode records source/version/license and file checksums, never silently emits fixtures, and reports a clear failure when authentic source data are unavailable.
- [x] 1.2 Add corpus inventory and deterministic website-grouped manifest generation; verify the report counts unique tasks/actions/sites, detects duplicate IDs and snapshot hashes, and enforces the 50/10/10 website floor.
- [x] 1.3 Update `RealWebTrajectoryDataset` to consume the manifest in strict real-only mode, emit each unique task once per epoch, and cache by source and manifest checksum; verify no fixture-folder records or repeated synthetic padding appear in the prepared cohort.
- [x] 1.4 Preserve recorded rationales, target elements, bounding boxes, and operations without fabricated fallbacks; mask missing/ambiguous labels and verify the audit reports their coverage and exclusions.

## 2. Action Taxonomy and Fair Baselines

- [x] 2.1 Add SELECT as action index 6 for ablation model configurations while retaining six-action legacy configurations; verify the four ablation heads emit seven classes and the step-19,500 checkpoint still loads with six.
- [x] 2.2 Calibrate the four baseline widths to the same instantiated trainable parameter budget within 1%; record per-model trainable and active parameter counts and verify the tolerance before training starts.
- [x] 2.3 Update the ablation trainer to use the real manifest, fresh initialization, matched sample order and optimizer settings, three complete epochs, and at least three configurable matched seeds; verify run records show identical sample/token budgets for every baseline at each seed.

## 3. Evaluation and Reproducible Outputs

- [x] 3.1 Update real-data evaluation to require the selected manifest and reject synthetic fallback or site leakage; verify it reports action accuracy, target-element hit rate, coordinate metrics, step accuracy, and exact offline trajectory match rate by website and in aggregate.
- [x] 3.2 Save per-seed/model checkpoints, training metrics, source/split/config checksums, hardware measurements, and status under a unique ignored run directory; verify collisions do not overwrite earlier runs.
- [x] 3.3 Generate Markdown and JSON reports with per-seed values, mean and dispersion, website-macro and overall-micro metrics, parameter/resource comparisons, memory and latency profiles, provenance, and a separate legacy-checkpoint section; verify the existing `ABLATION_REPORT.md` is not replaced.

## 4. Complete Ablation Run

- [x] 4.1 Prepare and audit the real-only corpus; verify the site split meets the configured diversity floor and is pairwise disjoint before allocating GPU training.
- [x] 4.2 Run memory and latency profiles plus all four matched baselines for every configured seed and three full training passes, then evaluate on the fixed site-held-out test manifest; verify all 12 model-seed runs have terminal records and comparable budgets.
- [x] 4.3 Review the final experiment report for finite metrics, per-site coverage, missing-label denominators, and explicit offline/custom-split limitations; verify outputs are isolated from tracked project data.

## Workflow follow-up

- Review the completed report and artifacts before archiving this change.
