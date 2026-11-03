# Design: Fine-Tuning Small-LM Baselines (T5-Small & DistilGPT-2)

## Context

In our initial evaluation ([`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md)), JED-M2 Nano achieved an 84.55% Step SR and 0.8468 Action F1 on the 11 site-held-out test websites. The small-LM baselines evaluated were raw off-the-shelf checkpoints (`t5-small` and `distilbert/distilgpt2`) with newly initialized sequence heads or zero-shot generation.

To provide a publication-grade scientific baseline comparison, we must fine-tune both T5-Small (~60.5M) and DistilGPT-2 (~81.9M) on the identical Mind2Web supervised training split (`data/manifests/real_mind2web_v1/manifest.json`), evaluate their post-fine-tuning performance, and establish whether JED-M2's hybrid Mamba/MLA architecture provides structural advantages over fine-tuned standard transformers.

## Goals / Non-Goals

**Goals:**
- Enhance [`train_baseline_lm.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/train_baseline_lm.py) with validation loss tracking, learning rate warmup, gradient clipping, and robust checkpoint serialization.
- Fine-tune `T5-Small` on the Mind2Web training manifest under `bfloat16` precision and save weights to `checkpoints/baseline_t5_small`.
- Fine-tune `DistilGPT-2` on the Mind2Web training manifest under `bfloat16` precision and save weights to `checkpoints/baseline_distilgpt2`.
- Update [`eval_benchmarks.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/eval_benchmarks.py) to load fine-tuned baseline checkpoints from directory paths and label them distinctly (e.g. `T5-Small Fine-Tuned (~60.5M)`).
- Execute the 11 held-out test website benchmark sweep and export the complete side-by-side comparative table to [`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md).

**Non-Goals:**
- Altering JED-M2 architecture, weights, or hyperparameters.
- Altering the 11 site-held-out test website split.
- Internet downloads during training (uses cached local model assets).

## Decisions

### 1. Training Recipe and Budget
- **Decision**: 50 optimization steps with batch size 4 and grad accumulation 2 (effective batch size 8), using AdamW with $\text{lr}=3\times 10^{-4}$ and weight decay 0.01.
- **Rationale**: 50 optimization steps covers ~400 interaction turns, sufficient to adapt the small LMs' generative heads to the DOM action syntax without overfitting or excessive compute time (~1-2 minutes per baseline on the RTX 5060 Laptop GPU).
- **Alternatives Considered**: 1,000 steps (would take > 30 minutes and risk severe catastrophic forgetting of language priors on small models).

### 2. Validation Loss Sanity Check
- **Decision**: Compute evaluation loss on the validation split (`real_mind2web_v1` val split) before and after fine-tuning, verifying that validation loss drops monotonically.
- **Rationale**: Confirms genuine sequence learning and prevents deploying degenerated or diverged baseline weights.

### 3. Checkpoint Directory Handling in `eval_benchmarks.py`
- **Decision**: When `checkpoint_path` is a directory containing `config.json` and `model.safetensors` / `pytorch_model.bin`, inspect directory name or config to instantiate `T5SmallActionAgent` or `DistilGPT2ActionAgent` with `model_name_or_path=checkpoint_path`.
- **Rationale**: Seamlessly integrates into existing comparative evaluation runner without requiring separate CLI flags.

## Risks / Trade-offs

- **[Risk] Small-LM Context Window Overflow** → DOM prompts with hundreds of elements exceed T5's 512-token limit.
  - *Mitigation*: [`DOMActionSequenceFormatter`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/small_lm_baselines.py) restricts DOM elements to the 24 most relevant candidates, keeping prompt length safely within 512 tokens.
- **[Risk] VRAM Allocation Ceiling** → Training language models could spike VRAM over 6.0 GB.
  - *Mitigation*: Run in `bfloat16` with micro-batch size 4, gradient accumulation 2, and `torch.cuda.empty_cache()` between runs. Peak VRAM will remain $< 3.5\text{ GB}$.
