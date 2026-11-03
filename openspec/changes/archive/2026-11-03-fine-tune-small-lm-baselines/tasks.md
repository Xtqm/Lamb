# Tasks: Fine-Tuning Small-LM Baselines (T5-Small & DistilGPT-2)

## 1. Training Pipeline Enhancement & Adaptor Unit Testing

- [x] 1.1 Enhance `train_baseline_lm.py` with validation loss evaluation, learning rate scheduling, gradient clipping, and robust directory checkpoint saving, verifying dry-run pass
- [x] 1.2 Update `eval_benchmarks.py` to load fine-tuned baseline weights from checkpoint directories (`checkpoints/baseline_t5_small`, `checkpoints/baseline_distilgpt2`) with distinct labeling
- [x] 1.3 Add unit tests in `tests/test_small_lm_baselines.py` verifying checkpoint serialization, directory loading, and loss computation

## 2. Baseline Model Fine-Tuning Execution

- [x] 2.1 Run fine-tuning for T5-Small on Mind2Web training manifest on CUDA and verify loss convergence and saved checkpoint in `checkpoints/baseline_t5_small`
- [x] 2.2 Run fine-tuning for DistilGPT-2 on Mind2Web training manifest on CUDA and verify loss convergence and saved checkpoint in `checkpoints/baseline_distilgpt2`
- [x] 2.3 Verify both fine-tuned checkpoint directories contain valid PyTorch/HuggingFace model weights and tokenizer configurations

## 3. Comparative Benchmark Sweep & Report Update

- [x] 3.1 Run full regression test suite (`pytest tests/ -q`) ensuring zero regressions
- [x] 3.2 Execute multi-checkpoint comparative evaluation across the 11 held-out test websites evaluating `jed_m2_sft_base`, `jed_m2_nano_pretrain_final`, `baseline_t5_small`, and `baseline_distilgpt2`
- [x] 3.3 Verify exported `BENCHMARK_REPORT.md` displays complete comparative metrics, parameter counts, latency, and step accuracy for fine-tuned baselines vs. JED-M2

## Workflow follow-up

- Validate specs with `openspec validate --specs`.
- Archive change with `openspec archive fine-tune-small-lm-baselines --yes`.
- Commit changes to git.
