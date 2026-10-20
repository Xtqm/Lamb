# Proposal

## Why

Phases 1 through 6B established and verified the core components of the JED-M2 architecture, including the Mamba SSM recurrent backbone, MLA attention, Top-2 MoE, JEPA latent world model, dual-system action heads, simulated Web/OS environment, and RL policy optimization. However, the model has only undergone short smoke benchmarks (e.g. 50 steps in Phase 5 and 25 steps in Phase 6B); it lacks a comprehensive, full-scale supervised fine-tuning (SFT) pretraining run that demonstrates multi-head convergence, stable token throughput, gradient norm stability, and validation generalization on held-out trajectories within the RTX 5060 Laptop GPU hardware envelope (8GB VRAM ceiling). 

Executing Phase 6C delivers `benchmark_sft.py` and enhances `train.py` and `jed_m2/trainer.py` to run a 1,000-step pretraining run on synthetic WebArena/Mind2Web trajectories, tracking convergence across all 6 loss heads and spatial/primitive metrics, and saving the foundational baseline production checkpoint `checkpoints/jed_m2_sft_base.pt`.

## What Changes

1. **Enhance `jed_m2/trainer.py` with Evaluation & Validation Support**:
   - Add `evaluation_step(batch)` and `evaluate(dataloader, max_eval_steps)` to `JEDM2Trainer` running in `torch.no_grad()` inference mode.
   - Aggregate validation metrics across all 6 loss heads (`loss_reason`, `loss_gate`, `loss_action_type`, `loss_coord`, `loss_arg`, `loss_jepa`, `loss_aux`), primitive accuracy, and coordinate MAE on held-out trajectories.
   - Support structured metrics history tracking (recording training and validation telemetry per step/interval).

2. **Enhance `train.py` CLI Training Harness**:
   - Add CLI arguments for evaluation: `--eval_interval` (e.g. every 100 steps), `--eval_steps` (e.g. 20 micro-batches), `--eval_dataset_size` (e.g. 1000 held-out trajectories), `--export_history` (JSON path for convergence telemetry), and `--baseline_checkpoint` (explicit target output path for production baseline).
   - Instantiate independent held-out evaluation dataset and collator with a distinct evaluation seed (e.g. seed + 1000) ensuring zero data leakage.
   - Run periodic validation passes during training, logging evaluation loss alongside training loss.
   - Automatically curate and export the production baseline checkpoint `checkpoints/jed_m2_sft_base.pt`.

3. **Implement `benchmark_sft.py` Orchestrator**:
   - Automated benchmarking script orchestrating the full 1,000-step SFT run with effective batch size 16 (micro-batch size 2, gradient accumulation 8) and sequence length 2048.
   - Monitors training dynamics: logs loss curves across all 6 loss heads, coordinate MAE, and action primitive classification accuracy.
   - Captures hardware telemetry: token throughput (tokens/sec), parameter gradient norm stability, and peak GPU VRAM allocation on the NVIDIA RTX 5060 Laptop GPU.
   - Generates structured convergence logs (`benchmark_sft_metrics.json`) and ASCII convergence summaries.

4. **Phase 6C Verification Suite (`verify_phase6c.py`)**:
   - Self-contained verification script asserting that:
     - The 1,000-step baseline checkpoint `checkpoints/jed_m2_sft_base.pt` exists, loads cleanly, and contains all expected parameter state dictionaries, optimizer states, and configuration metadata.
     - Validation loss evaluation functions correctly on held-out trajectories.
     - Coordinate MAE and primitive classification accuracy show monotonic or stable learning dynamics.
     - Parameter gradient norms remain stable with zero NaNs and Infs.
     - Peak allocated VRAM remains strictly below the 6.0 GB ceiling on CUDA in `bfloat16`.
   - Update `PROJECT_CONTEXT.md` roadmap marking Phase 6C as completed.

## Capabilities

### New Capabilities
- `sft-convergence-benchmarking`: Orchestrates comprehensive supervised fine-tuning (SFT) pretraining runs, logs multi-head loss curves (reasoning, gate, primitive, coordinates, JEPA, MoE load balancing) and spatial/action accuracy metrics, evaluates held-out validation trajectories, tracks hardware efficiency (tokens/sec, VRAM), and curates baseline production checkpoints.

### Modified Capabilities
- `production-training-harness`: Extends the production training harness with validation evaluation passes, held-out trajectory metrics aggregation, periodic evaluation checkpointing, and structured training metric history export.

## Impact

- **Affected Code**: `jed_m2/trainer.py`, `train.py`, `benchmark_sft.py` (new), `verify_phase6c.py` (new), `jed_m2/__init__.py`.
- **APIs & Dependencies**: 100% pure PyTorch; no external compiled packages or pretrained models.
- **Hardware & VRAM**: Operates on NVIDIA GeForce RTX 5060 Laptop GPU in `bfloat16`, utilizing gradient accumulation (effective batch size 16) and activation checkpointing within the 6.0 GB VRAM limit.
- **Breaking Changes**: None. All changes to `train.py` and `JEDM2Trainer` are backward-compatible with default fallback parameters.
