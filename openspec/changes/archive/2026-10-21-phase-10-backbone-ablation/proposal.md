# Proposal

## Why
To substantiate publication claims for JED-M2 as a next-generation Large Action Model (LAM), rigorous scientific evidence is required to isolate and quantify the empirical contributions of its core architectural innovations: linear SSM recurrence (Mamba), Multi-Head Latent Attention (MLA), Sparse Top-2 Mixture-of-Experts (MoE), and contextual JEPA latent world modeling. Without controlled parameter-matched baselines, performance gains cannot be definitively attributed to individual architectural components versus general scaling.

Now that Phase 9 has established a hardened backbone, true MLA latent caching with weight absorption, procedural scenario generation, and joint coordinate GRPO, Phase 10 executes a scientific 4-way ablation study. By calibrating all four models to an identical ~50.0M parameter budget, this study empirically demonstrates JED-M2's $O(1)$ constant memory scaling versus Transformer $O(N)$ memory blowout, profiles step latency in native `bfloat16` on the RTX 5060 GPU, measures multi-task convergence stability, and automatically generates publication-ready artifacts (tables, ASCII scaling charts, and LaTeX snippets).

## What Changes
- **Modular Architectural Toggles in `jed_m2/config.py`**: Add configuration switches:
  - `attn_type: Literal["mla", "mha"]`
  - `ffn_type: Literal["moe", "dense"]`
  - `use_mamba: bool`
  - `use_jepa: bool`
  - `dense_intermediate_dim: int` (calibrated for parameter budget parity in dense FFN models)
- **Baseline Modules in `jed_m2/baselines.py`**:
  - `StandardMHA`: Standard Multi-Head Attention with decoupled Q/K/V linear projections, rotary positional embeddings (RoPE), standard per-head KV caching, and scaled dot-product attention.
  - `DenseSwiGLUFFN`: Dense SwiGLU feed-forward network matching LLaMA-style gated FFN architectures.
  - Parameter-matched factory helper functions to instantiate Baselines A, B, C, and D.
- **Configurable `JEDM2Block` and `JEDM2Model` in `jed_m2/model.py`**:
  - Conditionally construct and execute Mamba SSM based on `config.use_mamba`.
  - Conditionally instantiate attention module (`MLA` vs `StandardMHA`) based on `config.attn_type`.
  - Conditionally instantiate FFN module (`SparseTop2MoE` vs `DenseSwiGLUFFN`) based on `config.ffn_type`.
  - Conditionally instantiate and update JEPA latent world model based on `config.use_jepa`.
  - Update `step(...)` recurrent rollout to support recurrent updates for both MHA (standard uncompressed KV cache) and MLA (compressed latent cache) with or without Mamba state buffers.
  - Update `count_parameters()` to accurately report total, trainable, and active parameters across both MoE and Dense FFN configurations.
- **Calibrated 4-Way Model Configurations (~50.0M Parameter Target)**:
  - **Baseline A (Pure Transformer)**: Standard MHA + Dense SwiGLU FFN (4 layers, ~50.5M params).
  - **Baseline B (Hybrid Attention-SSM)**: Mamba SSM + MLA + Dense SwiGLU FFN (3 layers, ~50.2M params).
  - **Baseline C (JED-M2 without JEPA)**: Mamba SSM + MLA + Sparse Top-2 MoE (3 layers, ~50.9M params).
  - **Baseline D (Full JED-M2 with Connected JEPA)**: Mamba SSM + MLA + Sparse MoE + Connected JEPA (3 layers, ~53.0M params).
- **Ablation Benchmark CLI (`run_ablation.py`)**:
  - **Experiment 1 (Memory Scaling Curve)**: Measure allocated and reserved VRAM and KV cache memory footprint across 16, 32, 64, 128 recurrent turns for all four baselines, contrasting $O(1)$ constant-memory recurrence against standard Transformer $O(N)$ growth.
  - **Experiment 2 (Inference Step Latency)**: Measure per-step latency (mean, median, p95 in ms/step) across 100 warm iterations in native `bfloat16` on CUDA.
  - **Experiment 3 (Controlled Convergence Run)**: Train all 4 baselines on identical procedural trajectory batches for 300 steps under identical seeds, logging loss curves, action primitive accuracy, and coordinate MAE.
  - **Experiment 4 (Paper Artifacts Export)**: Export comprehensive evaluation metrics to `ABLATION_REPORT.md` including markdown comparison tables, ASCII scaling plots, and publication-ready LaTeX table snippets.
- **Standalone Verification Suite (`verify_phase10.py`)**:
  - Verify initialization, parameter counts (all within 5% of 50M target), and forward/backward passes for all 4 baselines.
  - Verify memory profiling and latency benchmark execution.
  - Assert zero NaNs/Infs and strict compliance with the 6.0 GB VRAM limit.

## Capabilities

### New Capabilities
- `backbone-ablation-suite`: End-to-end scientific ablation benchmarking framework supporting memory scaling profiling across recurrent rollout horizons, per-step latency measurement in `bfloat16`, controlled multi-task trajectory convergence training across parameter-matched architectures, and automated paper artifact reporting (`ABLATION_REPORT.md` with ASCII scaling curves and LaTeX tables).

### Modified Capabilities
- `unified-lam-backbone`: Modifies backbone configuration and block execution to support modular architectural toggles (`attn_type`, `ffn_type`, `use_mamba`, `use_jepa`), standard baseline components (`StandardMHA`, `DenseSwiGLUFFN`), and parameter budget parity calibration (~50.0M total params).

## Impact
- **Codebase**:
  - `jed_m2/config.py`: Adds architectural configuration flags and defaults.
  - `jed_m2/baselines.py`: New module with `StandardMHA`, `DenseSwiGLUFFN`, and baseline model factory functions (`create_baseline_a`, `create_baseline_b`, `create_baseline_c`, `create_baseline_d`).
  - `jed_m2/model.py`: Updated `JEDM2Block` and `JEDM2Model` to support modular attention, FFN, SSM, and JEPA toggles.
  - `jed_m2/__init__.py`: Exports new baseline classes and factory helpers.
  - `run_ablation.py`: New CLI execution script for the 4-way ablation benchmark.
  - `verify_phase10.py`: New verification script.
  - `ABLATION_REPORT.md`: Generated ablation report and publication artifact.
- **APIs & Backward Compatibility**: Fully backward compatible. Default `JEDM2Config` retains all full JED-M2 settings (`attn_type="mla"`, `ffn_type="moe"`, `use_mamba=True`, `use_jepa=True`), ensuring existing models, checkpoints, and tests function without modification.
- **Hardware & VRAM**: All experiments and verification passes remain strictly constrained within the 6.0 GB VRAM envelope on the NVIDIA RTX 5060 Laptop GPU.
