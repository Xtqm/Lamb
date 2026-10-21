# Design

## Context
See `proposal.md` for background and motivation. 

Currently, JED-M2 operates with a default "Nano" configuration calibrated at $d_{\text{model}} = 512$, 3 stacked recurrent layers, 8 MoE experts, and $d_z = 256$ JEPA latent representation, yielding ~53M total parameters (~24M active). 

Phase 11 scales this architecture to "Medium" with $d_{\text{model}} = 768$, 6 stacked recurrent blocks, 16 MoE experts, and $d_z = 384$ JEPA representation. When compiled, the scaled architecture contains ~273M total parameters and ~75M active parameters per token (87.5% MoE sparsity). Empirical profiling on the local NVIDIA GeForce RTX 5060 Laptop GPU confirms that sequence length 2048 execution with activation checkpointing in native `bfloat16` consumes only ~2.42 GB peak VRAM, well beneath the 6.0 GB VRAM safety limit.

## Goals / Non-Goals

**Goals:**
- Provide a clean, declarative configuration interface for JED-M2-Medium in `jed_m2/config.py` via `JEDM2MediumConfig` and `model_size: Literal["nano", "medium"]`.
- Implement factory functions `create_jed_m2_medium()` and `create_jed_m2_nano()` in `jed_m2/model.py`.
- Generalize `count_parameters()` to dynamically account for 16-expert MoE configurations with arbitrary `top_k` and `num_experts`.
- Validate activation checkpointing across all 6 recurrent blocks under native `bfloat16` without memory leakage or gradient degradation.
- Provide uniform `--model_size {nano,medium}` CLI parameter across `train.py`, `train_rl.py`, and `eval_benchmarks.py`.
- Implement a standalone verification test harness `verify_phase11.py` executing the 4 verification gates (parameter parity, gradient sanity, recurrent step caching, and VRAM limits).

**Non-Goals:**
- Modifying underlying block execution math (Mamba SSM, MLA, or Top-2 MoE routing algorithms remain identical; only dimensions and layer depths scale).
- Introducing external CUDA extensions or compiled packages (maintaining the strict 100% native PyTorch build rule).
- Retraining production checkpoints within this planning change.

## Decisions

### Decision 1: Architecture Scale Parity vs. MoE Hyperparameter Calibration
- **Options Considered**:
  1. *Preserve exact architecture (16 SwiGLU experts, intermediate dim 1024, $d_{\text{model}} = 768$, 6 layers)*: Total parameters = 273.3M, Active parameters = 75.0M. Peak VRAM at sequence length 2048 with checkpointing is ~2.42 GB on the RTX 5060.
  2. *Artificially compress MoE intermediate dimension to 576*: Total parameters = 174.2M, Active parameters = 61.7M, but reduces expert capacity by 44%.
  3. *Reduce expert count to 8*: Total parameters = 160.0M, Active parameters = 46.8M, but drops routing diversity from 16 experts to 8.
- **Decision**: Option 1 (Confirmed via user alignment). The full 16-expert capacity and 1024 intermediate dimension are preserved, giving a robust ~273M total parameter model that operates with extraordinary memory efficiency on consumer GPUs (~2.4 GB peak VRAM).

### Decision 2: Factory Function Pattern and Config Tiers
- **Structure**:
  - In `jed_m2/config.py`:
    - `JEDM2Config` retains Nano defaults for backward compatibility.
    - `JEDM2MediumConfig(max_seq_len: int = 2048, **kwargs) -> JEDM2Config` returns a pre-calibrated `JEDM2Config` populated with Medium hyperparameters:
      - `hidden_dim = 768`
      - `num_heads = 12`
      - `q_lora_rank = 384`
      - `kv_lora_rank = 192`
      - `qk_rope_head_dim = 64`
      - `qk_nope_head_dim = 32`
      - `v_head_dim = 64`
      - `mamba_d_state = 16`
      - `mamba_d_conv = 4`
      - `mamba_expand = 2`
      - `mamba_dt_rank = 48`
      - `num_experts = 16`
      - `top_k = 2`
      - `moe_intermediate_dim = 1024`
      - `num_layers = 6`
      - `jepa_latent_dim = 384`
      - `model_size = "medium"`
  - In `jed_m2/model.py`:
    - `create_jed_m2_medium(config: Optional[JEDM2Config] = None) -> JEDM2Model`
    - `create_jed_m2_nano(config: Optional[JEDM2Config] = None) -> JEDM2Model`
- **Rationale**: Keeps configuration declarative and centralized. Existing code expecting `JEDM2Config()` continues to run unchanged.

### Decision 3: Parameter Accounting Generalization
- **Methodology**:
  - `total_params`: `sum(p.numel() for p in self.parameters())`.
  - `trainable_params`: `sum(p.numel() for p in self.parameters() if p.requires_grad)`.
  - `inactive_moe_params`: Calculated per block by inspecting `block.moe.num_experts` and `block.moe.top_k`. Each expert has `expert_params // num_experts` parameters, with `(num_experts - top_k)` inactive experts per token.
  - `target_encoder_params`: Calculated from `self.jepa.target_encoder.parameters()`.
  - `active_params`: `total_params - inactive_moe_params - target_encoder_params`.
- **Validation**:
  - Nano: ~53.0M total, ~24.1M active.
  - Medium: ~273.3M total, ~75.0M active.

### Decision 4: Activation Checkpointing Across 6 Layers
- In `JEDM2Model.forward()`:
  - Gradient activation checkpointing uses `torch.utils.checkpoint.checkpoint(block, x, attention_mask, use_reentrant=False)` inside `for block in self.blocks`.
  - Non-reentrant checkpointing is compatible with PyTorch autograd and avoids memory leaks from preserved frame references.
  - Because each `JEDM2Block` returns `(x, aux_loss)`, checkpointing seamlessly recomputes both activations and routing logits during backward pass without retaining 6 layers of intermediate activations in VRAM.

### Decision 5: CLI Flags in Training and Evaluation Scripts
- `train.py`, `train_rl.py`, and `eval_benchmarks.py` will accept:
  ```python
  parser.add_argument(
      "--model_size",
      type=str,
      default="nano",
      choices=["nano", "medium"],
      help="Model scaling tier: nano (~53M params) or medium (~273M params)",
  )
  ```
- When `--model_size medium` is specified, the script initializes `JEDM2MediumConfig(max_seq_len=args.seq_len)`.
- If loading from a checkpoint file containing `model_config` in its payload, `eval_benchmarks.py` and `train_rl.py` auto-detect the configuration dynamically, falling back to `--model_size` only when loading bare weights or initializing from scratch.

## Risks / Trade-offs

- **[Risk] Backward pass activation memory blowout at sequence length 2048 without checkpointing** → **Mitigation**: Gradient checkpointing is enabled by default in `train.py` and `train_rl.py` for Medium configurations, maintaining peak memory at ~2.4 GB - 2.8 GB.
- **[Risk] Slower per-step rollout latency with 6 stacked layers** → **Mitigation**: MLA true latent caching with absorbed weights and Mamba $O(1)$ recurrent buffers maintain sub-linear inference scaling. Latency benchmarks in `verify_phase11.py` profile and verify smooth step execution.
- **[Risk] Checkpoint schema incompatibility between Nano and Medium** → **Mitigation**: Checkpoint payloads store `model_config` metadata dictionary, ensuring loaders reconstruct the appropriate model tier automatically.

## Migration Plan

1. Update `jed_m2/config.py` with `JEDM2MediumConfig` and `model_size` field.
2. Update `jed_m2/model.py` with `create_jed_m2_medium`, `create_jed_m2_nano`, and generalized `count_parameters()`.
3. Export factory functions and configs in `jed_m2/__init__.py`.
4. Update CLI scripts (`train.py`, `train_rl.py`, `eval_benchmarks.py`) with `--model_size` support.
5. Author `verify_phase11.py` and run verification tests.
6. Verify 100% backward compatibility with existing Nano checkpoints and test suites.
