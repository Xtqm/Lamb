# Tasks: Parallel Prompt Prefill & Multi-Checkpoint Comparative Evaluation

## 1. Parallel Prompt Prefill Architecture in `JEDM2Model` & `JEDM2Block`

- [x] 1.1 Implement parallel prompt prefill in `jed_m2/mamba.py` and `jed_m2/model.py` returning final SSM hidden state, conv buffer, and MLA KV cache alongside final-step logits and verify with unit tests
- [x] 1.2 Implement parallel prefill in `JEDM2Block` and `JEDM2Model.prefill(...)` to ingest token sequences in a single batched tensor pass and verify state alignment with unit tests
- [x] 1.3 Add unit tests in `tests/test_parallel_prefill.py` asserting numerical parity between parallel prefill and sequential rollout states and verifying sub-100ms latency on CUDA

## 2. Agent Runtime Integration & Tokenizer Auto-Detection

- [x] 2.1 Refactor `JEDM2Agent._feed_token_sequence` in `jed_m2/agent.py` to use `model.prefill(...)` for observation prompts and verify seamless transition into autoregressive System 2 thought generation
- [x] 2.2 Update `eval_benchmarks.py` to auto-detect tokenizer type (`ActionBPETokenizer` vs `SimpleLAMTokenizer`) from checkpoint metadata or vocab size and verify with unit tests
- [x] 2.3 Add unit tests in `tests/test_parallel_prefill.py` verifying `JEDM2Agent` rollout on real DOM strings with parallel prefill

## 3. Multi-Checkpoint Benchmark Sweeps & Comparative Report Generation

- [x] 3.1 Run full regression suite (`pytest tests/`) ensuring zero regressions across all modules
- [x] 3.2 Execute comprehensive comparative benchmark sweep across the 11 held-out test websites evaluating `jed_m2_sft_base`, `jed_m2_nano_pretrain_final`, `jed_m2_grpo_step_40`, `t5_small`, and `distilgpt2`
- [x] 3.3 Verify exported `BENCHMARK_REPORT.md` displays complete comparative metrics, latency gains (< 100 ms), and generalization breakdowns

## Workflow follow-up

- Validate specs with `openspec validate --specs`.
- Archive change using `openspec archive parallel-prompt-and-checkpoint-eval --yes`.
- Commit changes to git.
