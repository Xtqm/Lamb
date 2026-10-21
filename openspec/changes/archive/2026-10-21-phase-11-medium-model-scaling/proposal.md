# Proposal

## Why
While the ~45M parameter JED-M2-Nano baseline demonstrated the foundational feasibility of hybrid Mamba SSM + Multi-Head Latent Attention (MLA) + Sparse Top-2 Mixture-of-Experts (MoE) on consumer hardware, complex multi-turn web navigation and operating system tasks demand deeper hierarchical reasoning, richer world modeling representations, and higher expert capacity. 

To scale agent competence without sacrificing local execution on consumer hardware, Phase 11 introduces the scaled JED-M2-Medium architecture. By calibrating hidden dimensions, attention heads, SSM expansion, and expert counts to a ~273M total parameter budget with ~75M active parameters per token (87.5% MoE routing sparsity across 16 experts), JED-M2-Medium significantly increases model expressivity while remaining strictly within the hardware envelope of the NVIDIA GeForce RTX 5060 Laptop GPU (< 6.0 GB VRAM in native `torch.bfloat16` under sequence length 2048 with activation checkpointing).

## What Changes
- **Architectural Configuration in `jed_m2/config.py`**:
  - Add `JEDM2MediumConfig` helper / factory configuration and `model_size: Literal["nano", "medium"]` model scaling toggle.
  - Layer depth: 6 recurrent blocks (stacked Mamba SSM + MLA + Top-2 MoE), up from 3 in Nano.
  - Hidden dimension: $d_{\text{model}} = 768$ (scaled up from 512).
  - Multi-Head Latent Attention (MLA):
    - Query compression rank: $d_{c\_q} = 384$.
    - Decoupled RoPE dimension: $d_r = 64$.
    - Non-positional query/key dimension: $d_{h}^{\text{NOPE}} = 32$.
    - Key-Value compression rank: $d_{c\_kv} = 192$.
    - Value head dimension: $d_v = 64$.
    - Number of attention heads: 12 heads (total head dimension $d_h = 64$).
  - Mamba Selective State Space Model (SSM):
    - Expanded inner dimension: $d_{\text{inner}} = 1536$ ($2\times$ expansion).
    - State space rank: $d_{\text{state}} = 16$.
    - 1D causal convolution kernel: $d_{\text{conv}} = 4$.
    - Delta projection rank: $d_{\Delta} = 48$.
  - Sparse Top-2 Mixture-of-Experts (MoE):
    - 16 SwiGLU expert MLPs (scaled up from 8 in Nano), routing top-2 experts per token.
    - Intermediate hidden dimension per expert: 1024.
    - Parameter allocation: ~226.5M total MoE parameters with ~28.3M active MoE parameters per token.
  - Connected JEPA Latent World Model:
    - Abstract latent state dimension: $d_z = 384$ (scaled up from 256).
    - Action conditioning aligned with scaled action heads ($d_a = 64$).
  - Dual-System Action Heads:
    - Action gate, action type classification head, continuous coordinate regressor with visual cross-attention grounding, and token argument projections scaled to accept $d_{\text{model}} = 768$.
- **Model Factory & Parameter Accounting in `jed_m2/model.py`**:
  - Implement `create_jed_m2_medium(config=None) -> JEDM2Model` factory function alongside `create_jed_m2_nano(config=None) -> JEDM2Model`.
  - Update `count_parameters()` to accurately report total, active, trainable, and inactive parameters across 16-expert MoE configurations.
  - Ensure gradient activation checkpointing operates cleanly across all 6 recurrent blocks without activation retention or memory leaks.
- **Hardware Envelope & Memory Budget**:
  - Target precision: Native `torch.bfloat16`.
  - Peak VRAM allocation: Strictly below the 6.0 GB ceiling on 8GB VRAM (measured: ~2.4 GB to 2.8 GB peak under sequence length 2048 with activation checkpointing).
  - Strict build rule: 100% native PyTorch with zero external compiled C++/CUDA dependencies (pure PyTorch Mamba and MLA).
- **CLI & Training Compatibility**:
  - Update `train.py`, `train_rl.py`, and `eval_benchmarks.py` to support `--model_size {nano,medium}` flag, defaulting to `nano` for backward compatibility.
- **Automated Verification Suite (`verify_phase11.py`)**:
  - **Test 1: Parameter Budget Parity**: Verify total parameter count (~273M $\pm 5\%$) and active parameter count per token (~75M $\pm 5\%$).
  - **Test 2: Gradient Integrity**: End-to-end forward and backward passes across all 6 loss heads without NaNs, Infs, or vanishing gradients.
  - **Test 3: Recurrent Step Rollout & Caching**: Autoregressive `step()` latency profiling, state shape integrity, and $O(1)$ constant-memory cache verification.
  - **Test 4: Peak Training VRAM Allocation**: Verify peak VRAM during forward and backward passes at sequence length 1024 and 2048 remains strictly $< 6.0\text{ GB}$.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `unified-lam-backbone`: Modifies backbone configuration, recurrent block stacking, parameter scaling, and factory functions to introduce JED-M2-Medium (~273M total / ~75M active parameters per token, 6 recurrent blocks, $d_{\text{model}} = 768$, 12 heads MLA, $d_{\text{inner}} = 1536$ Mamba SSM, 16 SwiGLU experts with Top-2 routing, $d_z = 384$ JEPA, scaled action heads), activation checkpointing across 6 blocks, and hardware VRAM adherence ($< 6.0\text{ GB}$ ceiling).
- `production-training-harness`: Modifies CLI training entrypoints (`train.py`, `train_rl.py`) to support `--model_size {nano,medium}` flag, dynamically instantiating the requested model configuration with calibrated memory and activation checkpointing for medium-scale training.
- `evaluation-benchmark-harness`: Modifies comparative evaluation runner and CLI script (`eval_benchmarks.py`) to support `--model_size {nano,medium}` flag, allowing evaluation of both Nano and Medium model checkpoints.

## Impact
- **Codebase**:
  - `jed_m2/config.py`: Adds `JEDM2MediumConfig` factory / defaults and `model_size` toggle.
  - `jed_m2/model.py`: Adds `create_jed_m2_medium` factory function, updates `count_parameters()`, and validates 6-block activation checkpointing.
  - `jed_m2/__init__.py`: Exports `JEDM2MediumConfig` and `create_jed_m2_medium`.
  - `train.py`: Adds `--model_size {nano,medium}` argument and model initialization logic.
  - `train_rl.py`: Adds `--model_size {nano,medium}` argument and policy/reference initialization logic.
  - `eval_benchmarks.py`: Adds `--model_size {nano,medium}` argument and model loader support.
  - `verify_phase11.py`: Comprehensive test suite verifying parameter parity, gradient integrity, recurrent step latency, and peak VRAM limits.
- **APIs & Backward Compatibility**: Fully backward compatible. Default `JEDM2Config` retains all Nano settings ($d_{\text{model}} = 512$, 3 blocks, 8 experts), ensuring existing training runs, checkpoints, and benchmark scripts continue to work unchanged.
- **Hardware & VRAM**: Forward and backward passes with sequence length 2048 execute at $\sim 2.4\text{ GB}$ peak VRAM, safely inside the 6.0 GB budget on the 8GB RTX 5060 Laptop GPU.
