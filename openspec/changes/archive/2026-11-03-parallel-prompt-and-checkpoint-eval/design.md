# Design: Parallel Prompt Prefill & Multi-Checkpoint Comparative Evaluation

## Context

In our initial baseline evaluation on 11 held-out test websites ([`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md)), JED-M2 exhibited a 22,007 ms decision latency per step, whereas T5-Small and DistilGPT-2 executed in 201–368 ms. Profiling identified that [`JEDM2Agent._feed_token_sequence`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/agent.py#L210) feeds observation prompt tokens sequentially using a Python `for tok in token_ids:` loop calling `model.step(...)`. Real Mind2Web DOM traces contain 1,500–2,000 tokens per prompt, leading to thousands of serialized CUDA kernel launches per interaction turn.

Additionally, our benchmark runner only evaluated `jed_m2_sft_base.pt` (an early synthetic checkpoint) on real DOM traces, while the pre-trained Byte-Level BPE model ([`jed_m2_nano_pretrain_final.pt`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt)) and post-RL policy ([`jed_m2_grpo_step_40.pt`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/checkpoints/grpo_run2/jed_m2_grpo_step_40.pt)) have not yet been evaluated side-by-side against the Small-LM baselines.

## Goals / Non-Goals

**Goals:**
- Implement parallel prompt prefill (`model.prefill(...)`) in [`JEDM2Model`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/model.py) and [`JEDM2Agent`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/agent.py) to process observation token sequences in a single parallel tensor pass using parallel associative Blelloch scan and parallel attention.
- Reduce JED-M2 prompt prefill latency from 22,000 ms to < 100 ms on CUDA.
- Seamlessly transition from parallel prompt prefill into $O(1)$ constant-memory recurrent `model.step(...)` for System 2 autoregressive thought generation.
- Add automatic tokenizer selection in [`eval_benchmarks.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/eval_benchmarks.py) to instantiate [`ActionBPETokenizer`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/bpe_tokenizer.py) for BPE checkpoints and [`SimpleLAMTokenizer`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/dataset.py) for legacy checkpoints.
- Run a multi-checkpoint comparative evaluation across the 11 held-out test websites comparing:
  - `jed_m2_sft_base`
  - `jed_m2_nano_pretrain_final`
  - `jed_m2_grpo_step_40`
  - `T5-Small (~60.5M)`
  - `DistilGPT-2 (~81.9M)`
- Update [`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md) with complete comparative empirical metrics.

**Non-Goals:**
- Modifying model architecture, parameter count, or weights.
- Running long-horizon multi-hour RL retraining sweeps (evaluation and prompt prefill optimization only).
- Introducing external compiled C++/CUDA/Triton binaries.

## Decisions

### 1. Unified Parallel Prefill Method in `JEDM2Model` (`model.prefill`)
- **Decision**: Implement `prefill(token_ids, visual_tokens=None)` on `JEDM2Model` and per-block `prefill` on `JEDM2Block`.
  - In `MambaBlock`: Compute parallel forward pass via parallel associative scan, extract the final state $h_{L-1} \in \mathbb{R}^{B \times d_{\text{inner}} \times d_{\text{state}}}$ and the last $d_{\text{conv}}$ elements as the `conv_state` buffer.
  - In `MultiHeadLatentAttention`: Compute compressed KV latents $c_{KV} \in \mathbb{R}^{B \times L \times d_c}$ across the prompt sequence to seed the MLA dynamic KV cache.
  - Return the final position's LM logits, ActionOutput, and per-layer state tuples `[(ssm_state, conv_state, kv_cache), ...]`.
- **Alternatives Considered**:
  - *Keep token-by-token loop*: Completely untenable due to 22-second per-step latency on real DOM prompts.
  - *Discard state after parallel forward*: Would break autoregressive System 2 thought generation and action gating.

### 2. Auto-Detecting Tokenizer in Benchmark Runner
- **Decision**: Inspect checkpoint metadata / config. If vocab size matches BPE (or if state dict indicates BPE tokenizer compatibility), instantiate `ActionBPETokenizer(vocab_size=32000)`; otherwise fallback to `SimpleLAMTokenizer`.
- **Alternatives Considered**: Manual CLI flag `--tokenizer {bpe,simple}`. Auto-detection is cleaner, less error-prone, and works automatically when passing multiple heterogeneous checkpoints.

### 3. Native PyTorch Purity & Memory Constraints
- **Decision**: Maintain 100% native PyTorch 2.x without external packages. Bound peak VRAM allocation to $< 6.0$ GB on the NVIDIA GeForce RTX 5060 Laptop GPU.

## Risks / Trade-offs

- **[Risk] State Equivalence between Parallel Prefill and Recurrent Step** → Ensure the state produced by parallel scan at sequence boundary $L-1$ matches the state that would be computed by sequential `step(...)`. Add unit tests asserting numerical equivalence ($< 10^{-4}$ tolerance).
- **[Risk] Sequence Length Truncation on Oversized DOMs** → DOM trees can exceed max sequence length (e.g., > 2048 tokens). Ensure prompt prefill safely clamps or truncates long prompts without crashing or creating NaN/Inf values.
- **[Risk] Checkpoint Weight Compatibility** → Earlier checkpoints might have different state dict keys. `eval_benchmarks.py` already includes key remapping logic, which will be preserved and tested.
