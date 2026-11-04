# Proposal: Fine-Tuning Small-LM Baselines (T5-Small & DistilGPT-2)

## Why

In our comparative evaluation on 11 held-out Mind2Web websites ([`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md)), JED-M2 Nano achieved an 84.55% Step SR and 0.8468 Action F1, outperforming off-the-shelf T5-Small (75.45% Step SR) and DistilGPT-2 (82.73% Step SR). However, comparing a domain-pretrained Large Action Model against raw, un-fine-tuned baseline models leaves a scientific confound: T5-Small and DistilGPT-2 have not yet been fine-tuned on the supervised WebArena / Mind2Web interaction traces. To establish a rigorous, publication-grade benchmark, both Small-LM baselines must be fine-tuned on the identical training split under matched optimization budgets and re-evaluated side-by-side with JED-M2.

## What Changes

- Complete the training pipeline in [`train_baseline_lm.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/train_baseline_lm.py) with validation evaluation, checkpoint saving, learning rate schedule, and reproducible seed controls.
- Execute fine-tuning runs on NVIDIA GeForce RTX 5060 Laptop GPU for both `T5-Small (~60.5M)` and `DistilGPT-2 (~81.9M)` on `data/manifests/real_mind2web_v1/manifest.json` (train split), saving verified weights to `checkpoints/baseline_t5_small` and `checkpoints/baseline_distilgpt2`.
- Update [`eval_benchmarks.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/eval_benchmarks.py) to load fine-tuned baseline weights from directory paths and evaluate them on identical 11 site-held-out test websites.
- Export an updated [`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md) with side-by-side comparisons of fine-tuned T5-Small, fine-tuned DistilGPT-2, `jed_m2_nano_pretrain_final`, and `jed_m2_sft_base`.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `small-lm-baselines`: Add fine-tuning execution, convergence verification, and checkpoint serialization requirements for T5-Small and DistilGPT-2.
- `evaluation-benchmark-harness`: Add fine-tuned baseline checkpoint evaluation and comparative reporting scenarios.

## Impact

- Affected files: [`train_baseline_lm.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/train_baseline_lm.py), [`eval_benchmarks.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/eval_benchmarks.py), [`tests/test_small_lm_baselines.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/tests/test_small_lm_baselines.py), [`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md).
- Dependencies: Standard Hugging Face `transformers` models already vendored and cached locally; no external network access or C++/CUDA compilation required.
- Hardware: Peak GPU VRAM strictly bounded $< 6.0\text{ GB}$ on RTX 5060 Laptop GPU using `bfloat16`.
