# JED-M2: Native Large Action Model (LAM) Project Context

## 1. System Overview & Architecture
JED-M2 is a custom, from-scratch Large Action Model (LAM) engineered to execute actions as first-class output primitives rather than generating raw conversational text. 
Target scale: ~45M total parameters / ~14M active parameters per token.

### Architectural Stack:
1. Multi-Head Latent Attention (MLA): Low-rank compressed Key-Value representation ($c_{KV}$) with decoupled Rotary Position Embeddings (RoPE).
2. Sparse Mixture of Experts (MoE): Top-2 gating across 8 SwiGLU expert feed-forward networks with auxiliary load-balancing loss.
3. Linear Sequential Recurrence: Pure PyTorch Mamba Selective State Space Model (SSM) for constant-memory long-horizon state tracking.
4. JEPA Latent World Model: Joint-Embedding Predictive Architecture mapping digital observations to abstract state $z_t$ and predicting $\hat{z}_{t+1}$.
5. Dual-System Action Decoders: System 1 discrete action classifier + continuous spatial coordinate regressor (normalized $[0, 1]$ coordinates).

---

## 2. Hardware Constraints & Environment Rules
- Host GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, sm_120 Blackwell architecture).
- Host System: AMD Ryzen 9, 32GB RAM, Windows 11.
- Execution Precision: `torch.bfloat16`.
- STRICT HARD RULE: No pretrained models or Hugging Face pretrained pipelines (`AutoModelForCausalLM.from_pretrained` is forbidden). All layers are natively constructed in PyTorch.
- STRICT BUILD RULE: No external compiled C++/CUDA packages (e.g., official `mamba-ssm` or `causal-conv1d`). All convolution and recurrence operations must remain pure PyTorch to avoid Windows MSVC / sm_120 compilation failures.
- Maximum Target VRAM Ceiling: Never exceed 6.0 GB during active trajectory training.

---

## 3. Repository Structure & Module Status
```text
Lamb/
├── PROJECT_CONTEXT.md          # Core architecture & progress tracker
├── verify_phase1.py            # Phase 1 standalone hardware & tensor test
├── verify_phase2.py            # Phase 2 SSM + JEPA verification test
└── jed_m2/
    ├── __init__.py             # Exports public modules
    ├── config.py               # Global JEDM2Config dataclass
    ├── mla.py                  # Multi-Head Latent Attention + RoPE + RMSNorm
    ├── moe.py                  # Sparse Top-2 MoE with SwiGLU & Aux Loss
    ├── mamba.py                # Pure PyTorch Mamba SSM block
    ├── jepa.py                 # Latent State Encoder, EMA Target, and Latent Predictor
    ├── heads.py                # Discrete Action & Coordinate Regression Heads
    └── model.py                # Full JED-M2 Unified Transformer-SSM Model
```
---

## 4. Phase Roadmap & Verification Status

### [DONE] Phase 1: Core Attention & Routing Primitives
- [x] Global configuration dataclass (`JEDM2Config`): `hidden_dim=512`, `num_heads=8`, `q_lora_rank=256`, `kv_lora_rank=128`, `num_experts=8`, `top_k=2`.
- [x] Low-rank Multi-Head Latent Attention (`mla.py`): ~0.70M parameters.
- [x] Sparse Top-2 MoE (`moe.py`): ~12.59M parameters.
- [x] Combined Layer Footprint: ~13.29M parameters.
- [x] Verification Output: Passes `bfloat16` forward + backward pass on `cuda:0` at 158.11 MB peak VRAM.

### [DONE] Phase 2: Sequential Recurrence & Latent World Model
- [x] Implement `MambaBlock` in pure PyTorch (`mamba.py`): ~1.69M parameters, pure PyTorch causal 1D conv + selective scan ZOH discretization.
- [x] Autoregressive `step(...)` equivalence: O(1) constant-memory recurrent inference matching sequence forward mode (< 0.001 max diff in `bfloat16`).
- [x] Implement `JEPAWorldModel` with EMA target network and Smooth L1 + cosine loss (`jepa.py`): ~1.61M total parameters (~1.22M trainable, 0.39M frozen EMA).
- [x] Phase 2 footprint: ~3.31M parameters; Cumulative architecture (Phases 1-2): ~16.20M parameters.
- [x] Verification Output: Passes forward, recurrent step equivalence, backward gradient checks (zero NaNs/Infs), and target network gradient isolation on `cuda:0` in `bfloat16` at 922.45 MB peak VRAM.

### [PENDING] Phase 3: Action Heads & Backbone Integration
- [ ] Implement `NativeActionHead` (classification logits + spatial coordinate regressor).
- [ ] Assemble unified `JEDM2Model` combining JEPA encoder, Mamba-MLA-MoE blocks, and action decoders.
- [ ] Implement `verify_phase3.py` profiling multi-step rollout memory.

### [PENDING] Phase 4: Trajectory Dataset & Loss Harness
- [ ] Synthetic trajectory generation pipeline (DOM/GUI states, discrete tools, target coordinates).
- [ ] Composite loss function ($\mathcal{L}_{\text{total}} = \lambda_{\text{jepa}}\mathcal{L}_{\text{JEPA}} + \lambda_{\text{aux}}\mathcal{L}_{\text{aux}} + \lambda_{\text{action}}\mathcal{L}_{\text{CE}} + \lambda_{\text{coord}}\mathcal{L}_{\text{Huber}}$).

### [PENDING] Phase 5: Training Loop & VRAM Optimization
- [ ] Implement 8-bit AdamW / fused optimizer training loop with gradient accumulation.
- [ ] Profile memory with batch size 2–4 and sequence length 2048 under active backpropagation.

---

## 5. Coding Agent Operational Directives
When implementing any task in this codebase:
1. Type Annotations: Annotate every function signature (`torch.Tensor`, `Optional`, `Tuple`).
2. Tensor Shape Comments: Document the expected input and output tensor shapes at every major projection step (e.g., `# [B, S, D] -> [B, S, q_lora_rank]`).
3. Standalone Verification: Every phase must be accompanied by a self-contained `verify_phaseX.py` script that verifies:
   - Forward pass correctness
   - Backward pass integrity (assert no `NaN` or `Inf` in parameter gradients)
   - Peak allocated GPU VRAM using `torch.cuda.max_memory_allocated()`
4. Do not delete, rewrite, or rename working components from previous phases without explicit instruction.