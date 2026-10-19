# Proposal

## Why

With Phases 1-4 completed and verified (Backbone architecture, action heads, synthetic trajectory dataset, and multi-task loss harness), the JED-M2 Large Action Model requires a production-grade training harness and policy optimization scaffolding calibrated strictly for the 8GB VRAM envelope of an RTX 5060 Laptop GPU.

Training 45M-parameter recurrent LAM models at sequence length 2048 typically risks out-of-memory (OOM) errors during backpropagation if optimizer states and activation footprints are unmanaged. Phase 5 introduces a specialized trainer (`jed_m2/trainer.py`) with fused 8-bit / fused AdamW optimization, selective activation checkpointing across the 3 JEDM2Block layers, mixed-precision `torch.bfloat16`, gradient accumulation, and model checkpointing, alongside a CLI training entrypoint (`train.py`), policy optimization scaffolding (`jed_m2/rl.py`) supporting DPO and GRPO for trajectory outcome rewards, and an end-to-end 50-step benchmark (`verify_phase5.py`) proving peak VRAM stays strictly below 6.0 GB.

## What Changes

- Implement `jed_m2/trainer.py`:
  - `JEDM2TrainerConfig`: Dataclass specifying training hyperparameters (learning rate, min_lr, warmup_steps, max_steps, weight_decay, grad_clip_norm, grad_accumulation_steps, micro_batch_size, checkpoint_dir, use_8bit_adam, gradient_checkpointing).
  - Optimizer integration: Fused 8-bit AdamW (via `bitsandbytes.optim.AdamW8bit` when available) with seamless fallback to PyTorch native CUDA hardware-accelerated `torch.optim.AdamW(..., fused=True)`.
  - Learning rate scheduling: Linear warmup followed by cosine annealing decay to `min_lr`.
  - Selective gradient checkpointing support: Integrates activation checkpointing across `JEDM2Block` layers to constrain activation memory to ~3.5GB - 4.5GB at sequence length 2048.
  - Mixed-precision training: Native `torch.bfloat16` context (`torch.cuda.amp.autocast(dtype=torch.bfloat16)`) with gradient norm clipping and accumulation.
  - Checkpoint manager: Atomic saving and resumption of model weights, optimizer states, lr scheduler, global step, and training metrics.
- Modify `jed_m2/model.py`:
  - Add `gradient_checkpointing_enable()` / `gradient_checkpointing_disable()` methods and `gradient_checkpointing` attribute to `JEDM2Model`.
  - When enabled, `JEDM2Model.forward` evaluates each `JEDM2Block` using `torch.utils.checkpoint.checkpoint(block, x, attention_mask, use_reentrant=False)`.
- Implement `train.py`:
  - Production CLI training script accepting flags for `--epochs`, `--max_steps`, `--batch_size`, `--micro_batch_size`, `--lr`, `--warmup_steps`, `--weight_decay`, `--grad_clip`, `--seq_len`, `--checkpoint_dir`, `--resume`, and `--log_interval`.
  - Live console metric logging profiling loss components, learning rate, throughput (tokens/sec), and allocated GPU VRAM.
- Implement `jed_m2/rl.py`:
  - Policy optimization scaffolding for Large Action Models:
  - `DPOTrainer` / `calculate_dpo_loss`: Direct Policy Optimization comparing chosen vs rejected multi-turn trajectories using implicit reward estimation:
    $\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(\tau_w, \tau_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(\tau_w)}{\pi_{\text{ref}}(\tau_w)} - \beta \log \frac{\pi_\theta(\tau_l)}{\pi_{\text{ref}}(\tau_l)}\right)\right]$
  - `GRPOTrainer` / `calculate_grpo_loss`: Group Relative Policy Optimization evaluating groups of $G$ sampled rollout trajectories for a given environment task, computing normalized group advantage:
    $A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\}) + \epsilon}$
    and optimizing clipped surrogate objectives with token-level KL divergence penalty against a reference policy.
- Expose training and RL modules in `jed_m2/__init__.py`.
- Implement `verify_phase5.py`:
  - 50-step end-to-end training execution benchmark at sequence length 2048 with synthetic trajectories on CUDA in `bfloat16`.
  - Profiles tokens/sec throughput, loss descent, gradient stability (zero NaNs/Infs), and strictly asserts peak allocated GPU VRAM remains below the 6.0 GB ceiling.

## Capabilities

### New Capabilities
- `production-training-harness`: Memory-calibrated training harness with 8-bit / fused AdamW, activation checkpointing, BF16 mixed-precision, gradient accumulation, checkpoint resumption, and CLI script.
- `trajectory-policy-optimization`: Direct Policy Optimization (DPO) and Group Relative Policy Optimization (GRPO) scaffolding on multi-turn agent trajectory outcome rewards (success vs failure).

### Modified Capabilities
- `unified-lam-backbone`: Added selective gradient checkpointing support to `JEDM2Model` across stacked `JEDM2Block` layers for bounded activation memory at sequence length 2048.

## Impact

- New modules: `jed_m2/trainer.py`, `jed_m2/rl.py`, `train.py`, `verify_phase5.py`.
- Modified modules: `jed_m2/model.py`, `jed_m2/__init__.py`.
- Hardware compliance: Specifically calibrated for 8GB VRAM (RTX 5060 Laptop GPU), ensuring peak VRAM strictly < 6.0 GB under sequence length 2048 backpropagation.
- Dependencies: 100% native PyTorch, optional `bitsandbytes` when present with seamless fallback to native `torch.optim.AdamW(fused=True)`.
