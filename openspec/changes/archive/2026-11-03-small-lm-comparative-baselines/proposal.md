# Proposal: Fine-Tuned Small-LM Baselines (T5-Small & DistilGPT-2)

## Why

A core critique identified in the independent publication readiness audit is the absence of external small pretrained language model baselines:
While JED-M2 was evaluated against internal architectural ablations (Baseline A Pure Transformer, Baseline B Hybrid SSM, Baseline C without JEPA), it has not been benchmarked against established, widely recognized open-source language models in the sub-100M parameter tier (specifically Google T5-Small at ~60M parameters and Hugging Face DistilGPT-2 at ~82M parameters). 
Without direct empirical comparisons on identical data splits and tasks, peer reviewers cannot verify whether JED-M2's hybrid Mamba-MLA-MoE-JEPA architecture delivers genuine efficiency, memory, or grounding advantages over commodity pretrained language models fine-tuned for web agent action prediction.

## What Changes

- **Text-to-Action Sequence Formatting**: Implement standardized token sequence serialization for browser agent interaction turns:
  - Input prompt: `[GOAL] <goal> [OBS] <flattened DOM candidates with bounding boxes> [HISTORY] <prior actions>`
  - Target output: `<thought> <rationale> </thought> [ACTION] <action_type> (<norm_x>, <norm_y>) [ARG] <arg> [/ARG]`
- **T5-Small (~60M) Fine-Tuning Adaptor**:
  - Implement a dedicated sequence-to-sequence fine-tuning harness for T5-Small using native Hugging Face `transformers` and pure PyTorch.
  - Implement an action parser converting generated sequence tokens into discrete `ActionType`, continuous pointer coordinates (x, y), and argument strings.
- **DistilGPT-2 (~82M) Fine-Tuning Adaptor**:
  - Implement an autoregressive causal LM fine-tuning harness for DistilGPT-2 using causal next-token cross-entropy.
  - Implement greedy/temperature generation with structured action output parsing.
- **Unified Training & Evaluation on Identical Splits**:
  - Fine-tune both baselines on the exact 51-website real Mind2Web training manifest (`data/manifests/real_mind2web_v1/manifest.json`).
  - Evaluate on the 11-website held-out test manifest across official Mind2Web generalization splits (`cross_task`, `cross_website`, `cross_domain`).
- **Standardized Multi-Model Comparative Table & Artifacts**:
  - Extend the benchmark evaluator to generate a unified comparative matrix reporting parameter count, active parameter count, $O(1)$ vs $O(N)$ KV-cache footprint, per-step latency (ms), Action Accuracy (%), Coordinate MAE, and Step Success Rate (%).

## Capabilities

### New Capabilities
- `small-lm-baselines`: Adaptors, tokenizers, prompt sequence formatters, training wrappers, and structured action prediction parsers for T5-Small and DistilGPT-2 on DOM interaction tasks.

### Modified Capabilities
- `evaluation-benchmark-harness`: Extend benchmark reporting to support external sequence-to-sequence and causal language model baselines alongside native LAM checkpoints.

## Impact

- Affected files:
  - `jed_m2/small_lm_baselines.py`: T5 and DistilGPT-2 model adaptors, prompt sequence serializer, structured action string parser.
  - `train_baseline_lm.py`: Fine-tuning runner for T5-Small and DistilGPT-2 on real Mind2Web manifests.
  - `eval_benchmarks.py` & `jed_m2/benchmark.py`: Support evaluating Hugging Face baseline models within the standard benchmark runner.
  - `tests/test_small_lm_baselines.py`: Unit tests verifying prompt serialization, action parsing, loss computation, and output format conformance.
- Hardware Envelope: Peak VRAM remains strictly $< 6.0$ GB on NVIDIA RTX 5060 Laptop GPU in `bfloat16`.
