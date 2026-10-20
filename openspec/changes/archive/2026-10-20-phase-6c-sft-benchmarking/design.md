# Design

## Context

Phases 1 through 6B established the complete architectural stack of JED-M2:
- Recurrent backbone stacking 3 `JEDM2Block` layers combining pure PyTorch Mamba SSM, low-rank MLA attention, and Top-2 SwiGLU MoE (~50.97M total parameters, ~22.26M active parameters per token).
- Dual-process System 1 / System 2 action decoding (ActionGate, ActionTypeHead, CoordinateHead, ArgumentHead) and JEPA world model.
- Dynamic batch collation, composite multi-task loss harness, and memory-calibrated trainer supporting mixed-precision `torch.bfloat16`, fused AdamW, and activation checkpointing.

While Phase 5 verified 50 training steps and Phase 6B verified 25 RL iterations, Phase 6C executes the full 1,000-step supervised fine-tuning (SFT) pretraining run on synthetic WebArena/Mind2Web trajectories. The training run must adhere strictly to the 6.0 GB VRAM ceiling on the host NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7, sm_120 Blackwell architecture), evaluate generalization on held-out trajectories, track multi-head convergence and spatial/discrete accuracy, and curate the production baseline checkpoint `checkpoints/jed_m2_sft_base.pt`.

## Goals / Non-Goals

**Goals:**
- Implement `benchmark_sft.py` orchestrating the 1,000-step SFT pretraining run using `train.py` with effective batch size 16 (micro-batch size 2, gradient accumulation 8) and sequence length 2048.
- Extend `JEDM2Trainer` and `train.py` with evaluation passes (`evaluation_step`, `evaluate`), validation loss evaluation on held-out synthetic trajectories, and structured history export (`benchmark_sft_metrics.json`).
- Log convergence curves across all 6 loss heads: reasoning cross-entropy, action gate BCE, action primitive cross-entropy, coordinate Huber loss, JEPA latent consistency, and MoE routing penalty.
- Track coordinate Mean Absolute Error (MAE) and action primitive classification accuracy across training steps.
- Profile and document token throughput (tokens/sec), gradient norm stability, and peak GPU VRAM allocation.
- Save and verify the production baseline checkpoint `checkpoints/jed_m2_sft_base.pt`.
- Provide self-contained verification suite `verify_phase6c.py`.

**Non-Goals:**
- Multimodal pixel/screenshot visual patch encoding (reserved for Phase 6D: Vision-LAM extension).
- Reinforcement learning policy updates (GRPO / DPO completed in Phase 6B).
- Multi-GPU Distributed Data Parallel (DDP) scaling (target is single-GPU local hardware calibration).

## Decisions

### 1. Batch Sizing and Gradient Accumulation Architecture
- **Decision**: Configure training with micro-batch size 2, sequence length 2048, and gradient accumulation steps 8, yielding an effective batch size of 16 ($2 \times 8 = 16$).
- **Rationale**: An effective batch size of 16 provides smooth, low-variance gradient estimates across multi-turn trajectories with varying action densities. With sequence length 2048 and activation checkpointing across the 3 JEDM2Block layers, a micro-batch of 2 occupies ~750 MB to 1.2 GB of active GPU VRAM, comfortably below the 6.0 GB ceiling while maximizing GPU tensor core utilization.
- **Alternatives Considered**:
  - *Micro-batch size 4 with accumulation 4*: Increases activation memory pressure toward 2.5 GB without improving mathematical update frequency.
  - *Micro-batch size 1 with accumulation 16*: Lowers activation memory to ~400 MB but incurs higher PyTorch loop overhead and underutilizes Blackwell Tensor Cores.

### 2. Held-Out Evaluation Dataset & Zero-Leakage Validation
- **Decision**: Separate the training trajectory stream from evaluation using an independent `AgentTrajectoryDataset` instance initialized with an evaluation seed offset (e.g. `seed + 1000` or `seed=999`) and held-out virtual dataset size (e.g. 1,000 trajectories). Validation passes execute periodically (e.g. every 100 or 250 steps) inside `torch.no_grad()` across a fixed batch of held-out samples.
- **Rationale**: Real-time evaluation on held-out trajectories measures true out-of-distribution generalization for coordinate regression and action primitive classification, verifying that the model is not memorizing synthetic prompt templates.
- **Alternatives Considered**:
  - *Evaluating on training batches*: Flawed validation methodology; produces artificially low loss curves without detecting overfitting.
  - *Static pre-rendered validation disk files*: Unnecessary disk overhead; deterministic procedural generation with independent seeds guarantees identical reproducible evaluation trajectories without storing gigabytes of synthetic data on disk.

### 3. Multi-Head Loss & Convergence Metric Aggregation
- **Decision**: Track all component losses and diagnostic metrics at each logging interval:
  1. `loss_reason`: Autoregressive thought cross-entropy.
  2. `loss_gate`: Action emission vs reasoning binary cross-entropy.
  3. `loss_action_type`: Discrete action primitive cross-entropy.
  4. `loss_coord`: Spatial coordinate Smooth L1 loss.
  5. `loss_jepa`: JEPA latent consistency (Smooth L1 + cosine distance).
  6. `loss_aux`: MoE expert load-balancing auxiliary penalty.
  7. `loss_arg`: Action argument token cross-entropy.
  8. `loss_total`: Composite weighted multi-task loss.
  9. `coord_mae`: Continuous coordinate Mean Absolute Error ($L_1$ distance in normalized screen space $[0, 1]$).
  10. `acc_action_type`: Classification accuracy over the 6 discrete primitive classes.
  11. Hardware metrics: `tok_per_sec`, `grad_norm`, `peak_vram_mb`.
- **Rationale**: Tracking individual heads allows immediate detection of gradient dominance or starvation (e.g. if thought tokens overpower spatial regression or MoE routing collapses).
- **Alternatives Considered**:
  - *Tracking only composite total loss*: Hides internal dynamics (e.g., coordinate MAE could worsen while thought token loss decreases).

### 4. Modular Benchmark Orchestrator (`benchmark_sft.py`)
- **Decision**: Implement `benchmark_sft.py` as an orchestration layer that interfaces with `train.py` / `JEDM2Trainer`. It coordinates the 1,000-step training loop with validation checkpoints, collects time-series telemetry across all 6 loss heads, outputs real-time ASCII progress, exports structured JSON metrics (`benchmark_sft_metrics.json`), prints an ASCII summary table/curve, and guarantees baseline checkpoint curation at `checkpoints/jed_m2_sft_base.pt`.
- **Rationale**: Keeps `train.py` clean and production-oriented while providing a dedicated, repeatable benchmarking harness for Phase 6C requirements.
- **Alternatives Considered**:
  - *Embedding benchmark-specific logging entirely in `train.py`*: Blurs the line between general training harness and milestone benchmarking. Extending `train.py` with evaluation flags and having `benchmark_sft.py` orchestrate and document the run maintains separation of concerns.

### 5. Production Baseline Checkpoint Curation
- **Decision**: The baseline checkpoint `checkpoints/jed_m2_sft_base.pt` is atomically saved upon completion of the 1,000-step run. It includes complete model state dictionaries, optimizer state, learning rate scheduler state, configuration dataclasses, and final step metrics.
- **Rationale**: Serves as the official foundational pretraining checkpoint for subsequent phases (Phase 6D Vision-LAM fine-tuning, RL policy fine-tuning, and production deployment).

## Risks / Trade-offs

- **[Risk] Convergence Stalling or Head Imbalance**: Some loss heads (e.g., JEPA latent prediction or coordinate regression) might converge slower or with higher variance than thought tokens.
  - *Mitigation*: The composite loss harness uses calibrated weighting ($\lambda_{\text{reason}}=1.0, \lambda_{\text{gate}}=1.0, \lambda_{\text{type}}=1.0, \lambda_{\text{coord}}=5.0, \lambda_{\text{jepa}}=1.0, \lambda_{\text{aux}}=0.01$). Cosine learning rate scheduling with 100-step warmup and gradient clipping at 1.0 ensures stable parameter updates across all heads.
- **[Risk] VRAM Leakage During Long-Horizon 1,000-Step Run**: Accumulation of unreferenced tensors or dynamic compute graphs over 1,000 steps could lead to memory fragmentation and eventual OOM.
  - *Mitigation*: `training_step` calls `optimizer.zero_grad(set_to_none=True)` after each optimizer step; validation passes execute inside `torch.no_grad()`; intermediate metric tensors are detached (`.item()`) immediately; explicit cache clearance and peak memory monitoring ensure VRAM remains strictly below 6.0 GB throughout.
- **[Risk] Execution Time on Windows Host**: Executing 1,000 steps with micro-batch size 2 and gradient accumulation 8 processes 8,000 micro-batches.
  - *Mitigation*: Blackwell Tensor Cores in `bfloat16` achieve high throughput (~335+ tokens/sec). With sequence length 2048 and fast PyTorch fused AdamW, the benchmark script will provide progress bars and live metrics to monitor steady completion.
