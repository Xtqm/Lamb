# Design: Small-LM Comparative Baselines (T5-Small & DistilGPT-2)

## Context

JED-M2 employs a hybrid architecture stacking Mamba SSMs, Multi-Head Latent Attention (MLA), and Sparse Top-2 Mixture-of-Experts (MoE) with a connected JEPA world model, operating at ~53M total / ~24M active parameters per token. 
While internal architectural ablations (Baseline A Transformer, Baseline B Hybrid SSM, Baseline C without JEPA) were conducted in Phase 10, an essential peer-review question remains: *How does JED-M2 compare against standard, off-the-shelf small language models fine-tuned for browser action prediction?*

The host environment has Hugging Face `transformers` and `torch` available. We introduce two canonical sub-100M baseline models:
1. **Google T5-Small (~60M parameters)**: Standard sequence-to-sequence encoder-decoder architecture.
2. **Hugging Face DistilGPT-2 (~82M parameters)**: Standard autoregressive causal language model.

Both baselines will be trained and evaluated on the exact same 51-train / 11-test website-held-out manifest partitions as JED-M2.

## Goals / Non-Goals

**Goals:**
- Provide a clean sequence serialization format converting multi-turn browser DOM interactions into input-output text strings.
- Implement `T5SmallActionAgent` and `DistilGPT2ActionAgent` wrappers supporting tokenization, loss computation, and action string parsing into discrete `ActionType`, coordinates $(x, y)$, and argument strings.
- Implement `train_baseline_lm.py` supporting parameter-matched fine-tuning in `bfloat16` under $< 6.0$ GB VRAM.
- Integrate both baselines into `eval_benchmarks.py` and output side-by-side comparative metric tables in `BENCHMARK_REPORT.md`.
- Measure parameter counts, active parameters per token, per-step latency, Action Accuracy, continuous Coordinate MAE, and Step Success Rate.

**Non-Goals:**
- Large 7B+ models (e.g. LLaMA-3-8B, UI-TARS, SeeClick) are excluded; the study strictly evaluates the sub-100M parameter regime.
- No modifications to the standard T5 or DistilGPT2 architectures; they serve as standard, unaugmented baselines.

## Decisions

### 1. Unified Text-to-Action Sequence Grammar
To allow standard LMs to emit actions without custom architectural heads, we define a compact, deterministic string grammar:
- **Input Prompt**:
  `[GOAL] {goal} [OBS] {element_0_desc} [{x0},{y0},{x1},{y1}] ; {element_1_desc} ... [HISTORY] {prev_actions}`
- **Target Output**:
  `<thought> {rationale} </thought> [ACTION] {ACTION_TYPE} ({x:.3f}, {y:.3f}) [ARG] {argument} [/ARG]`

*Alternative considered*: Adding separate regression heads on top of T5/DistilGPT2.  
*Rationale*: Standard practice for language models in web/GUI agents (e.g., Mind2Web seq2seq baselines, WebN-T5) is generating textual action primitives and normalized coordinate tokens directly. This provides a fair comparison testing the LM's native representation.

### 2. Robust Action Parsing with Fallbacks
Generated strings are parsed via regular expressions:
`r"\[ACTION\]\s*([A-Z_]+)\s*\(\s*([\d\.]+)\s*,\s*([\d\.]+)\s*\)(?:\s*\[ARG\](.*?)\[/ARG\])?"`
If coordinate strings fail to parse, the parser falls back to the center of candidate elements matching the text or default $(0.5, 0.5)$, recording a parse error flag.

### 3. Isolated Baseline Module & CLI
Create `jed_m2/small_lm_baselines.py` housing model classes, sequence formatters, and parsers, keeping JED-M2 core modules decoupled from Hugging Face dependencies.

## Risks / Trade-offs

- **[Risk] Sequence Truncation**: Serialized DOM elements can exceed maximum context length (512 or 1024 tokens).  
  *Mitigation*: Truncate candidate list to top 24-32 salient interactive elements, matching JED-M2's candidate pool capacity.
- **[Risk] Generation Latency**: Autoregressive decoding token-by-token in Python can be slower than JED-M2's parallel discrete heads.  
  *Mitigation*: Use short generation limits (max 32 target tokens per action) and greedy decoding (`do_sample=False`).
- **[Risk] VRAM Allocation**: DistilGPT2 and T5 during training might exceed GPU memory if batch sizes are too large.  
  *Mitigation*: Enforce gradient accumulation with batch size 4 or 8 in `bfloat16`, peak VRAM $< 3.5$ GB.
