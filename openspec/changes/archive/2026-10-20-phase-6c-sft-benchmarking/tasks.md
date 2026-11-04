# Tasks

## 1. Trainer & Training Harness Enhancement (`jed_m2/trainer.py` and `train.py`)

- [x] 1.1 Implement `evaluation_step(batch)` and `evaluate(dataset, collator, max_eval_steps)` in `JEDM2Trainer` (`jed_m2/trainer.py`) executing under `torch.no_grad()` to compute validation losses across all 6 loss heads, coordinate MAE, and primitive accuracy without updating model weights. Verify with a standalone micro-batch evaluation call asserting valid metric outputs.
- [x] 1.2 Enhance `train.py` CLI parser with `--eval_interval`, `--eval_steps`, `--eval_dataset_size`, `--export_history`, and `--baseline_checkpoint` arguments. Verify CLI arguments via `python train.py --help`.
- [x] 1.3 Implement held-out evaluation dataset streaming and periodic validation passes in `train.py`, aggregating multi-head validation metrics, logging evaluation summaries to the console, and exporting structured JSON convergence history upon completion. Verify execution with a short 2-step evaluation dry run.

## 2. Benchmark Orchestrator Implementation (`benchmark_sft.py`)

- [x] 2.1 Implement `benchmark_sft.py` CLI script orchestrating the full 1,000-step SFT pretraining run on synthetic WebArena/Mind2Web trajectories with micro-batch size 2, gradient accumulation 8 (effective batch size 16), and sequence length 2048. Verify script initialization and CLI argument parsing via `python benchmark_sft.py --help`.
- [x] 2.2 Implement multi-head convergence logging in `benchmark_sft.py` tracking loss curves for reasoning, action gate, primitive classification, coordinate regression, JEPA latent consistency, and MoE load balancing, alongside coordinate MAE and primitive accuracy. Verify metric extraction and real-time formatting.
- [x] 2.3 Implement hardware profiling and ASCII convergence summary rendering in `benchmark_sft.py`, recording token throughput (tokens/sec), gradient norm stability, peak GPU VRAM allocation, and exporting structured metrics to `benchmark_sft_metrics.json`. Verify telemetry collection and JSON report generation.
- [x] 2.4 Integrate baseline checkpoint persistence in `benchmark_sft.py` saving and verifying the final baseline production checkpoint at `checkpoints/jed_m2_sft_base.pt`. Verify file presence and state dict contents.

## 3. Production Verification Suite & Context Update

- [x] 3.1 Implement standalone verification suite `verify_phase6c.py` asserting baseline checkpoint integrity, held-out validation loss evaluation, parameter gradient stability, and peak VRAM allocation (< 6.0 GB ceiling). Verify by executing `python verify_phase6c.py`.
- [x] 3.2 Update `PROJECT_CONTEXT.md` roadmap marking Phase 6C as completed with benchmark results and hardware memory profiles. Verify markdown formatting and phase completion status.
