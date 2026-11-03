# Tasks: Fine-Tuned Small-LM Baselines (T5-Small & DistilGPT-2)

## 1. Prompt Sequence Serialization and Action Parsing

- [x] 1.1 Implement `DOMActionSequenceFormatter` in `jed_m2/small_lm_baselines.py` converting turn inputs (goal, interactive DOM elements with normalized bounding boxes, action history) into structured sequence prompt strings and target action strings and verify with unit tests
- [x] 1.2 Implement `parse_action_string()` in `jed_m2/small_lm_baselines.py` extracting discrete ActionType, normalized coordinate pair (x, y), and argument strings with robust fallback handling and verify with unit tests
- [x] 1.3 Add unit tests in `tests/test_small_lm_baselines.py` verifying round-trip prompt formatting, coordinate token decoding, and regex parsing precision

## 2. Pretrained Small-LM Baseline Adaptors

- [x] 2.1 Implement `T5SmallActionAgent` in `jed_m2/small_lm_baselines.py` wrapping Hugging Face T5-Small with seq2seq forward loss computation and greedy sequence generation and verify with unit tests
- [x] 2.2 Implement `DistilGPT2ActionAgent` in `jed_m2/small_lm_baselines.py` wrapping Hugging Face DistilGPT-2 with causal next-token cross-entropy loss and prompt-masked training and verify with unit tests
- [x] 2.3 Add unit tests in `tests/test_small_lm_baselines.py` verifying forward passes, gradient flow, token shape compatibility, and generation without shape mismatches

## 3. Fine-Tuning Pipeline & Multi-Model Comparative Evaluation Harness

- [x] 3.1 Implement `train_baseline_lm.py` supporting fine-tuning T5-Small and DistilGPT-2 on the 51-website real Mind2Web training manifest with gradient accumulation and bfloat16 mixed precision and verify with unit tests
- [x] 3.2 Update `eval_benchmarks.py` and `jed_m2/benchmark.py` to support evaluating `T5SmallActionAgent` and `DistilGPT2ActionAgent` across identical held-out test splits and verify with unit tests
- [x] 3.3 Add unit tests verifying `eval_benchmarks.py` correctly scores baseline agent predictions and records comparative metrics without errors

## 4. End-to-End Verification & Comparative Baseline Report

- [x] 4.1 Run the full test suite (`pytest tests/`) ensuring zero regressions across existing JED-M2 tests and verifying all new baseline tests pass
- [x] 4.2 Execute a comparative benchmark sweep across the 11-website held-out test split, comparing JED-M2 against T5-Small and DistilGPT-2 on Action Accuracy, Coordinate MAE, Step Accuracy, Latency, and Memory Footprint in `BENCHMARK_REPORT.md`
