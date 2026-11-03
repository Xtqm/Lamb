# Proposal: Parallel Prompt Prefill & Multi-Checkpoint Comparative Evaluation

## Why

In the Phase 4 comparative baseline evaluation ([`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md)), JED-M2 exhibited two major issues:
1. **Severe Inference Latency (~22,000 ms/step)**: The agent runtime sequentially ingested observation prompts in a Python `for tok in token_ids:` loop calling `model.step(...)` token-by-token. Real Mind2Web DOM observations contain 1,500–2,000 tokens, resulting in thousands of serialized CUDA kernel launches per step. In contrast, T5-Small and DistilGPT-2 ingest the prompt in a single parallel tensor pass (200–368 ms).
2. **Evaluation Gap on Trained Checkpoints**: Only the early, synthetic `jed_m2_sft_base.pt` was evaluated on real DOM traces (yielding 0.0% step accuracy), while the unified pretrain checkpoint ([`jed_m2_nano_pretrain_final.pt`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt), trained with Byte-Level BPE on real Mind2Web + WebArena) and the post-RL checkpoint ([`jed_m2_grpo_step_40.pt`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/checkpoints/grpo_run2/jed_m2_grpo_step_40.pt)) have not yet been evaluated side-by-side against the T5-Small and DistilGPT-2 baselines.

Executing parallel prompt prefill via JED-M2's parallel associative Blelloch scan and parallel attention will collapse prompt ingestion latency to < 50 ms. Benchmarking our best trained checkpoints against T5-Small and DistilGPT-2 on identical held-out test splits will produce the empirical results required for publication.

## What Changes

1. **Parallel Observation Prompt Prefill in `JEDM2Agent`**:
   - Refactor [`JEDM2Agent._feed_token_sequence`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/agent.py) to pass observation prompt tokens as a single 2D tensor `[1, seq_len]` into `model.forward(...)` rather than looping token-by-token through `model.step(...)`.
   - Seed recurrent Mamba SSM states, convolution buffers, and MLA KV-cache from the parallel forward pass so subsequent System 2 thought generation continues seamlessly using $O(1)$ constant-memory `model.step(...)`.
2. **Tokenizer Alignment in Benchmark Evaluator**:
   - Update [`eval_benchmarks.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/eval_benchmarks.py) to automatically detect and load [`ActionBPETokenizer`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/bpe_tokenizer.py) for checkpoints trained with BPE vocabularies (e.g. `jed_m2_nano_pretrain_final.pt` and `jed_m2_grpo_step_40.pt`), ensuring byte-level text invertible encoding.
3. **Multi-Checkpoint Comparative Sweep**:
   - Execute side-by-side benchmark sweeps across the 11-website held-out test split, comparing:
     - `jed_m2_sft_base.pt` (initial synthetic SFT base)
     - `jed_m2_nano_pretrain_final.pt` (unified pretrain with Byte-Level BPE)
     - `jed_m2_grpo_step_40.pt` (online GRPO post-trained)
     - `t5_small` (~60.5M baseline)
     - `distilgpt2` (~81.9M baseline)
   - Export updated comprehensive metrics (ASR %, Step SR, Macro F1, Coord MAE, Latency ms, Peak VRAM) to `BENCHMARK_REPORT.md`.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `interactive-agent-runtime`: Update requirement to mandate parallel observation prompt prefill using parallel scan and parallel attention, bounding prompt ingestion latency to sub-100 ms on CUDA.
- `evaluation-benchmark-harness`: Update comparative benchmarking requirement to evaluate unified pretrain (`jed_m2_nano_pretrain_final.pt`) and RL (`jed_m2_grpo_step_40.pt`) checkpoints with aligned BPE tokenization alongside baselines on identical held-out splits.

## Impact

- **Codebase**: Modifies [`jed_m2/agent.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/agent.py), [`jed_m2/model.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/jed_m2/model.py) (if state export is needed from forward pass), and [`eval_benchmarks.py`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/eval_benchmarks.py).
- **Inference Latency**: Drops JED-M2 per-step latency from ~22,000 ms to < 100 ms, beating or matching baseline LM inference times.
- **Reporting**: Regenerates [`BENCHMARK_REPORT.md`](file:///c:/Users/caleb/Documents/Coding/AI/Meta-Experiment-AiByAi/Lamb/BENCHMARK_REPORT.md) with authentic multi-checkpoint comparison.
