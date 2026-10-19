# Tasks

## 1. Backbone Activation Checkpointing

- [x] 1.1 Add `gradient_checkpointing_enable()`, `gradient_checkpointing_disable()`, and checkpointing execution logic in `JEDM2Model.forward` in `jed_m2/model.py`, verifying backward pass runs with reduced activation memory.

## 2. Production Training Harness Implementation

- [x] 2.1 Implement `JEDM2TrainerConfig` and optimizer factory in `jed_m2/trainer.py` supporting fused 8-bit AdamW (`bitsandbytes`) with automatic fallback to native PyTorch CUDA-fused AdamW (`torch.optim.AdamW(..., fused=True)`), verifying optimizer initialization and step execution.
- [x] 2.2 Implement learning rate warmup, cosine decay schedule, mixed-precision `torch.bfloat16` context, gradient norm clipping, and gradient accumulation loop in `jed_m2/trainer.py`.
- [x] 2.3 Implement atomic checkpoint saving and resumption (`save_checkpoint`, `load_checkpoint`) in `jed_m2/trainer.py` persisting model weights, optimizer states, scheduler state, and training metrics.
- [x] 2.4 Implement production CLI script `train.py` with argument parsing for epochs, batch sizes, learning rate, sequence length, and live console metric logging.

## 3. Trajectory Policy Optimization (RL) Scaffolding

- [x] 3.1 Implement Direct Policy Optimization (`calculate_dpo_loss`, `DPOTrainer`) in `jed_m2/rl.py` comparing chosen vs rejected trajectories with implicit reward margin computation.
- [x] 3.2 Implement Group Relative Policy Optimization (`calculate_grpo_loss`, `GRPOTrainer`) in `jed_m2/rl.py` evaluating groups of rollouts with normalized relative advantages and reference policy KL penalties.

## 4. Public API Exposure & Standalone 50-Step Benchmark Verification

- [x] 4.1 Re-export trainer, RL, and checkpointing classes in `jed_m2/__init__.py`.
- [x] 4.2 Create and execute `verify_phase5.py` running a 50-step end-to-end training benchmark at sequence length 2048 on CUDA, profiling throughput (tokens/sec), gradient stability (zero NaNs/Infs), checkpoint save/reload, and strictly asserting peak allocated VRAM stays below the 6.0 GB ceiling.
