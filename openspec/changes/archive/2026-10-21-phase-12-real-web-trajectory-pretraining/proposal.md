# Proposal

## Why
While synthetic trajectory generators and procedural environments proved essential for validating the architectural foundations, loss formulations, and reinforcement learning pipelines of JED-M2, synthetic data lacks the combinatorial complexity, irregular DOM hierarchies, dynamic layouts, and natural language variability of real-world websites. Furthermore, synthetic observations rely on artificial Gaussian noise rather than authentic visual-semantic web page structures.

To bridge the sim-to-real gap and prepare JED-M2 for production-grade web automation, Phase 12 introduces real-world web trajectory ingestion (from Mind2Web and WebArena offline traces) and executes a scaled pretraining run (5,000 to 10,000 steps) on consumer hardware. Training on genuine multi-turn interaction traces empowers the hybrid Mamba SSM + MLA + MoE architecture to ground user intentions in actual DOM elements, master continuous screen coordinate regression, and maintain latent state consistency across complex browser workflows, all within the 6.0 GB VRAM budget of the NVIDIA GeForce RTX 5060 Laptop GPU.

## What Changes
- **Real Trajectory Dataset & DOM Parser (`jed_m2/real_dataset.py`)**:
  - Replace synthetic Gaussian noise and mock DOM strings with offline web interaction traces based on Mind2Web / WebArena structures.
  - Implement an intelligent, lightweight DOM tree parser and pruning pipeline:
    - Prune raw HTML documents (often exceeding 10,000+ nodes) down to interactive candidate elements (e.g., `button`, `input`, `a`, `select`, `textarea`).
    - Extract and normalize bounding box coordinates $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}] \in [0, 1]^2$ and compute normalized center coordinates $(x, y)$.
    - Tokenize user task intent, pruned DOM accessibility representations, and System 2 rationale thoughts into unified sequence tensors.
  - Implement `RealWebTrajectoryDataset` with support for stratified train, validation, and test splits and persistent disk caching for high-throughput streaming.
- **Pure PyTorch Trajectory Collator (`jed_m2/real_dataset.py`)**:
  - Collate variable-length multi-turn web interaction episodes into padded batches with 4D causal attention masks.
  - Align multi-task loss masks for dual-process decoding:
    - Thought tokens: Autoregressive NLL reasoning loss mask.
    - Discrete action primitives: Multiclass CE mask over primitives (`CLICK`, `TYPE`, `SCROLL`, `NAVIGATE`, `FINISH`).
    - Continuous spatial coordinates: Smooth L1 / Huber regression loss mask over normalized $(x, y)$ targets.
    - Argument strings: Masked CE over typed text, URLs, and key sequences.
    - Connected JEPA world model: Aligned transition tuples $(s_t, a_t, s_{t+1})$ for next-state latent consistency.
- **Scaled Pretraining Engine (`train_pretrain.py`)**:
  - Production CLI training script supporting both JED-M2-Nano (51M) and JED-M2-Medium (175M).
  - Mixed-precision `torch.bfloat16` with fused/8-bit AdamW optimizer, cosine learning rate decay, and linear warmup.
  - Gradient accumulation enabling effective batch size 16 to 32 with micro-batch size 2 within the 6.0 GB VRAM ceiling.
  - Real-time telemetry logging: Loss decomposition ($L_{\text{reason}}$, $L_{\text{type}}$, $L_{\text{coord}}$, $L_{\text{arg}}$, $L_{\text{jepa}}$, $L_{\text{aux}}$), token throughput (tok/s), gradient norms, and live GPU VRAM tracking.
  - Periodic checkpointing with automated evaluation passes on the held-out validation split.
- **Target Convergence Criteria**:
  - Discrete action primitive classification accuracy $> 70\%$ on held-out real web tasks.
  - Continuous coordinate Mean Absolute Error (MAE) $< 0.05$ (matching typical button and input hitbox tolerances).
  - JEPA latent predictive cosine similarity $> 0.85$ between predicted and target representations.
- **Verification Suite (`verify_phase12.py`)**:
  - Test 1: DOM parser & bounding box extraction unit tests on realistic HTML pages.
  - Test 2: Dataset loading, persistent caching, and batch collation tensor shapes.
  - Test 3: Convergence test on 100 sample real trajectories verifying stable multi-task loss decrease.
  - Test 4: Peak VRAM compliance verification ($< 6.0\text{ GB}$ on RTX 5060 Laptop GPU).

## Capabilities

### New Capabilities
- `real-web-trajectory-dataset`: Ingestion of real-world offline web interaction traces (Mind2Web / WebArena format), lightweight DOM tree parsing and HTML pruning (filtering 10,000+ node trees to interactive candidates), normalized bounding box and center coordinate extraction, sequence tokenization, train/val/test splits, persistent disk caching, and variable-length trajectory collation with aligned dual-process loss masks and JEPA transition pairs.

### Modified Capabilities
- `production-training-harness`: Extends CLI training orchestration and evaluation to support scaled pretraining on real web interaction traces via `train_pretrain.py`, supporting both JED-M2-Nano (51M) and JED-M2-Medium (175M), fused AdamW optimizer with cosine LR decay and warmup, gradient accumulation (effective batch size 16 to 32) within 6.0 GB VRAM limit, comprehensive telemetry logging (loss decomposition, token throughput, gradient norms, live VRAM tracking), and automated periodic validation against convergence targets.

## Impact
- **Codebase**:
  - `jed_m2/real_dataset.py`: New module containing DOM parser, element extractor, `RealWebTrajectoryDataset`, and `RealTrajectoryCollator`.
  - `jed_m2/__init__.py`: Export new dataset classes, collator, and DOM parsing utilities.
  - `train_pretrain.py`: New production CLI pretraining entrypoint with real-data streaming, gradient accumulation, live telemetry, and validation checkpoints.
  - `verify_phase12.py`: Verification script covering DOM parsing, batch shapes, sample trajectory convergence, and peak VRAM adherence.
- **APIs & Backward Compatibility**: Fully backward compatible. Existing `AgentTrajectoryDataset` and `SyntheticTrajectoryGenerator` in `jed_m2/dataset.py` and `train.py` remain intact for synthetic benchmarking.
- **Hardware & VRAM**: Calibrated specifically for the NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) to maintain peak memory usage strictly $< 6.0\text{ GB}$ in `torch.bfloat16`.
