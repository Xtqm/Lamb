# Design: Production Training Harness & Policy Optimization (Phase 5)

## Context

Phases 1 through 4 established and verified the core components of JED-M2:
- Low-rank Multi-Head Latent Attention (MLA) and Sparse Top-2 MoE (`mla.py`, `moe.py`)
- Mamba Selective State Space Model (SSM) and JEPA Latent World Model (`mamba.py`, `jepa.py`)
- Dual-Process Action Decoupling Heads and 3-block unified LAM backbone (`heads.py`, `model.py`)
- Synthetic Agent Trajectory Dataset and Unified Composite Loss Harness (`dataset.py`, `losses.py`)

To train this ~45M parameter Large Action Model on our target hardware envelope (NVIDIA GeForce RTX 5060 Laptop GPU with 8GB GDDR7 VRAM, sm_120 Blackwell architecture), the training pipeline must satisfy strict physical memory constraints during active backpropagation at long sequence lengths (2048 tokens).

Physical Memory Budget:
- Total VRAM: 8.0 GB
- Strict training ceiling: $\le 6.0\text{ GB}$
- Parameters (BF16): ~102 MB
- Optimizer States: ~102 MB (8-bit) / ~408 MB (fused FP32)
- Gradients: ~102 MB
- Activation Footprint: $\le 4.5\text{ GB}$ with selective gradient checkpointing at sequence length 2048

## Goals / Non-Goals

**Goals:**
- Implement `jed_m2/trainer.py` providing `JEDM2Trainer` and `JEDM2TrainerConfig` with:
  - Fused 8-bit AdamW optimizer integration (`bitsandbytes.optim.AdamW8bit` when available) with seamless fallback to native PyTorch hardware-accelerated `torch.optim.AdamW(..., fused=True)`.
  - Selective gradient checkpointing across the 3 `JEDM2Block` layers to constrain activation memory within ~3.5GB - 4.5GB at sequence length 2048.
  - Native `torch.bfloat16` mixed precision with gradient clipping (`clip_grad_norm_`).
  - Gradient accumulation support for effective batch sizes of 16-32 with micro-batch size 2-4.
  - Robust checkpoint saving and resumption preserving model weights, optimizer states, lr scheduler, and metrics.
- Extend `JEDM2Model` in `jed_m2/model.py` with `gradient_checkpointing_enable()` / `disable()`.
- Implement `train.py` CLI script with configurable epochs, learning rate warmup, cosine decay schedule, and live metric logging.
- Implement `jed_m2/rl.py` providing Direct Policy Optimization (DPO) and Group Relative Policy Optimization (GRPO) scaffolding for trajectory outcome rewards.
- Implement `verify_phase5.py` executing a 50-step end-to-end training benchmark at sequence length 2048 on CUDA, profiling throughput (tokens/sec), gradient stability (zero NaNs/Infs), and strictly proving peak allocated VRAM stays below 6.0 GB.

**Non-Goals:**
- Full multi-node distributed training (DDP / FSDP / DeepSpeed): Designed for single-GPU 8GB VRAM envelope.
- External compiled C++/CUDA packages (strictly preserving 100% pure PyTorch portability).
- Pretrained Hugging Face weights or pipelines.

## Decisions

### 1. Dual-Path Optimizer Strategy (8-bit AdamW / Fused AdamW)
- **Decision**: Provide automatic detection: if `bitsandbytes` is installed and `use_8bit_adam=True`, use `bitsandbytes.optim.AdamW8bit`. Otherwise, use PyTorch's native `torch.optim.AdamW(..., fused=is_cuda)`.
- **Rationale**: PyTorch 2.14 includes native hardware-accelerated `fused=True` for AdamW on CUDA. Both 8-bit AdamW (~102 MB states) and fused AdamW (~408 MB states) fit well within our 8GB VRAM envelope. Giving priority to 8-bit while falling back cleanly to native fused AdamW guarantees zero-crash execution across any environment without compiling external C++ wheels.
- **Alternatives Considered**:
  - *Strict bitsandbytes dependency*: Fails on environments lacking prebuilt Windows sm_120 wheels.
  - *Standard non-fused AdamW*: Incurs multiple kernel launch overheads and higher memory thrashing.

### 2. Selective Gradient Checkpointing in `JEDM2Model`
- **Decision**: Add `gradient_checkpointing: bool = False` to `JEDM2Model` along with `gradient_checkpointing_enable()` and `gradient_checkpointing_disable()`.
  During the sequence forward pass, if enabled and training:
  `x, aux_loss = torch.utils.checkpoint.checkpoint(block, x, attention_mask, use_reentrant=False)`
- **Rationale**: The activation footprint for 3 stacked blocks (each with Mamba SSM, MLA low-rank attention, and 8-expert Top-2 SwiGLU MoE) at sequence length 2048 and batch size 2 exceeds 4.5 GB without checkpointing. Selective checkpointing at the block boundary discards intermediate forward activations and recomputes them on backward pass, slashing activation memory to ~1.5 - 2.5 GB while adding less than 15-20% computational overhead.
- **Alternatives Considered**:
  - *Sub-layer checkpointing (checkpointing only MoE or only MLA)*: Added code complexity with marginal benefit over clean block-level checkpointing.
  - *Reentrant checkpointing (`use_reentrant=True`)*: Deprecated and incompatible with tuple outputs without extra wrappers.

### 3. Mixed-Precision and Gradient Scaling
- **Decision**: Use native `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`.
  Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)` before optimizer steps.
- **Rationale**: NVIDIA Blackwell (sm_120) has dedicated Bfloat16 Tensor Cores. Bfloat16 has an 8-bit dynamic exponent range identical to FP32, eliminating underflow risks without requiring loss scale hunting. Gradient norm clipping (default 1.0) ensures stability during early training and across MoE routing perturbations.

### 4. Gradient Accumulation Mechanics
- **Decision**:
  The trainer loops over `grad_accumulation_steps` micro-batches.
  For each micro-batch:
  - Compute loss: `loss = composite_loss(...)`
  - Scale loss: `scaled_loss = loss / grad_accumulation_steps`
  - Backward pass: `scaled_loss.backward()`
  - Every `grad_accumulation_steps`, apply gradient clipping and `optimizer.step()`, then `optimizer.zero_grad()`.
- **Rationale**: An effective batch size of 16-32 is optimal for stable MoE expert load balancing and JEPA representation learning. Micro-batch size 2-4 fits sequence length 2048 inside the 8GB VRAM envelope. Accumulation decouples memory footprint from effective optimization batch size.

### 5. Policy Optimization Scaffolding (DPO and GRPO)
- **Decision**:
  - `calculate_dpo_loss`: Evaluates action token log probabilities $\log \pi_\theta(\tau_w)$, $\log \pi_{\text{ref}}(\tau_w)$, $\log \pi_\theta(\tau_l)$, and $\log \pi_{\text{ref}}(\tau_l)$.
    $$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[ \log \frac{\pi_\theta(\tau_w)}{\pi_{\text{ref}}(\tau_w)} - \log \frac{\pi_\theta(\tau_l)}{\pi_{\text{ref}}(\tau_l)} \right]\right)$$
  - `calculate_grpo_loss`: Given a group of $G$ sampled rollout trajectories with outcome rewards $\{r_1, \dots, r_G\}$:
    Computes group-normalized advantages: $A_i = (r_i - \mu) / (\sigma + \epsilon)$
    Evaluates clipped surrogate policy gradient with token-level KL divergence penalty against reference policy:
    $$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|T_i|}\sum_{t \in T_i} \left[ \min(r_{i,t} A_i, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) A_i) - \beta_{\text{KL}} D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]$$
- **Rationale**: Large Action Models act in environments where outcomes (task success vs failure) provide clean trajectory-level reward signals. DPO allows direct offline preference learning from paired trajectories; GRPO provides online group-relative policy gradient optimization without an explicit critic network (saving VRAM).

## Risks / Trade-offs

- **[Risk]** Sequence length 2048 memory spike if collator padding creates unnecessarily large tensors during accumulation.
  - **Mitigation**: Trajectory batches are dynamically padded to the maximum sequence length in that specific batch (bounded by `max_seq_len=2048`), and `torch.cuda.empty_cache()` is called between epochs/validation.
- **[Risk]** Gradient checkpointing interaction with Mamba SSM internal operations.
  - **Mitigation**: MambaBlock in pure PyTorch consists of standard tensor operations (`F.conv1d`, `torch.einsum`, `F.silu`) that are 100% differentiable and support activation recomputation without custom backward hooks.
- **[Risk]** Memory ceiling violation during optimizer step if optimizer states allocate extra temporary buffers.
  - **Mitigation**: Verification benchmark strictly monitors `torch.cuda.max_memory_allocated()` across 50 consecutive forward-backward-optimizer steps, asserting that the peak never exceeds 6.0 GB (6144 MB).
