# JED-M2: Native Large Action Model (LAM) Project Context

## 1. System Overview & Architecture
JED-M2 is a custom, from-scratch Large Action Model (LAM) engineered to execute actions as first-class output primitives rather than generating raw conversational text. 
Target scale: ~45M total parameters / ~14M active parameters per token, constructed by stacking **3 core recurrent blocks** over a shared latent observation and world model.

### Key Architectural Stack:
1. **JEPA Latent World Model (`jepa.py`)**:
   - Maps digital environment observations (DOM, GUI states, text) to abstract latent state $z_t \in \mathbb{R}^{256}$ via `LatentStateEncoder`.
   - Maintains an exponential moving average (EMA) momentum target encoder ($\tau=0.99$) with strictly detached gradients.
   - Action-conditioned transition predictor $\hat{z}_{t+1} = \text{Predictor}(z_t, a_t)$ with residual dynamics and composite Smooth L1 + cosine distance loss.
2. **JED-M2 Core Recurrent Stack (x3 Stacked Blocks)**:
   - **Linear Sequential Recurrence (`mamba.py`)**: Pure PyTorch Mamba Selective State Space Model (SSM) for constant-memory $O(1)$ long-horizon state tracking via rolling state buffers (`conv_state`, `ssm_state`).
   - **Multi-Head Latent Attention (`mla.py`)**: Low-rank compressed Key-Value representation ($c_{KV}=128$) and Query compression ($c_Q=256$) with decoupled Rotary Position Embeddings (RoPE $d_R=32$) and `F.scaled_dot_product_attention`.
   - **Sparse Mixture of Experts (`moe.py`)**: Top-2 gating across 8 SwiGLU expert feed-forward networks with DeepSeek/Switch-style auxiliary load-balancing loss.
3. **Dual-Process Action Decoupling (`heads.py`)**:
   - **System 2 (Latent Reasoning & Rationale)**: Autoregressive latent thought tokens / scratchpad representations for multi-step planning before action commitment.
   - **System 1 (Action Decoder Heads)**:
     - *Action Gate*: Binary/scalar classifier predicting whether to emit an executable action or continue reasoning.
     - *Discrete Primitive Head*: Softmax distribution over discrete primitives (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH).
     - *Spatial Coordinate Regressor*: Bounded regression head predicting normalized $(x, y) \in [0, 1]$ screen coordinates using a Sigmoid activation.
     - *Argument Head*: Constrained token decoding for payload arguments (e.g., keystroke text, query strings).

---

## 2. Hardware Constraints & Training Memory Budget
- **Host GPU**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7 VRAM, sm_120 Blackwell architecture).
- **Host System**: AMD Ryzen 9, 32GB System RAM, Windows 11.
- **Execution Precision**: `torch.bfloat16` natively supported by Blackwell Tensor Cores.
- **STRICT HARD RULE**: No pretrained models or Hugging Face pretrained pipelines (`AutoModelForCausalLM.from_pretrained` is forbidden). All layers are natively constructed from scratch in PyTorch.
- **STRICT BUILD RULE**: No external compiled C++/CUDA packages (e.g., official `mamba-ssm` or `causal-conv1d`). All convolution and recurrence operations must remain 100% pure PyTorch to avoid Windows MSVC / sm_120 compilation failures.
- **Target VRAM Ceiling**: Never exceed 6.0 GB during active trajectory training.

### Scratch Training Memory Budget (8GB VRAM Envelope)
| Component | Budget (45M Nano-LAM) | Notes |
| :--- | :--- | :--- |
| **Model Parameters** | ~90 MB | 45M parameters at BF16 |
| **Optimizer States** | ~180 MB | Fused 8-bit AdamW (`bitsandbytes.optim.AdamW8bit` / fused AdamW) |
| **Gradients** | ~90 MB | BF16 parameter gradients |
| **Activation Footprint** | ~3.5 GB – 4.5 GB | Sequence length 2048, batch size 2–4 with selective gradient checkpointing |
| **Headroom / CUDA Context** | ~1.5 GB – 2.0 GB | Buffer for PyTorch CUDA runtime context and dynamic tensor allocations |
| **Total Peak Allocation** | **~5.5 GB – 6.8 GB** | Comfortably within the 8.0 GB physical hardware limit |

---

## 3. Mathematical & Loss Formulations

The model is trained end-to-end on multi-turn trajectory tuples $\tau = (s_0, a_0, r_0, s_1, a_1, \dots, s_T)$ using a composite loss function:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{reason}}\mathcal{L}_{\text{NLL}} + \lambda_{\text{type}}\mathcal{L}_{\text{CE}} + \lambda_{\text{coord}}\mathcal{L}_{\text{Huber}} + \lambda_{\text{arg}}\mathcal{L}_{\text{arg}} + \lambda_{\text{jepa}}\mathcal{L}_{\text{JEPA}} + \lambda_{\text{aux}}\mathcal{L}_{\text{aux}}$$

1. **Reasoning Loss ($\mathcal{L}_{\text{NLL}}$)**: Cross-entropy over autoregressive strategic rationale/thought tokens.
2. **Action Primitive Loss ($\mathcal{L}_{\text{CE}}$)**: Multi-class cross-entropy over discrete action categories ($K$ action types).
3. **Coordinate Regression Loss ($\mathcal{L}_{\text{Huber}}$)**: Smooth L1 / Huber loss over normalized continuous coordinates $\hat{c} = (\hat{x}, \hat{y}) \in [0, 1]$ to penalize spatial errors without gradient explosion.
4. **Argument Decoding Loss ($\mathcal{L}_{\text{arg}}$)**: Masked cross-entropy over argument token sequences.
5. **JEPA Latent Consistency Loss ($\mathcal{L}_{\text{JEPA}}$)**:
   $$\mathcal{L}_{\text{JEPA}} = \text{SmoothL1}(\hat{z}_{t+1}, z_{t+1}) + \lambda_{\text{cos}}(1 - \text{cosine\_similarity}(\hat{z}_{t+1}, z_{t+1}))$$
6. **MoE Load-Balancing Loss ($\mathcal{L}_{\text{aux}}$)**:
   $$\mathcal{L}_{\text{aux}} = N \sum_{i=1}^N f_i \cdot P_i$$
   where $f_i$ is the fraction of tokens routed to expert $i$, and $P_i$ is the average routing probability.

---

## 4. Repository Structure & Module Status
```text
Lamb/
├── .gitignore                  # Python, PyTorch, and deep learning artifacts filter
├── PROJECT_CONTEXT.md          # Core architecture, specs & progress tracker
├── verify_phase1.py            # Phase 1 standalone hardware & tensor test (MLA + MoE)
├── verify_phase2.py            # Phase 2 SSM + JEPA verification test
├── verify_phase3.py            # Phase 3 Action heads & backbone rollout test
├── verify_phase4.py            # Phase 4 Dataset collation & composite loss test
├── openspec/                   # Spec-driven development framework & archived changes
└── jed_m2/
    ├── __init__.py             # Exports public modules
    ├── config.py               # Global JEDM2Config dataclass
    ├── mla.py                  # Multi-Head Latent Attention + RoPE + RMSNorm (Sequence + Step)
    ├── moe.py                  # Sparse Top-2 MoE with SwiGLU & Aux Loss
    ├── mamba.py                # Pure PyTorch Mamba SSM block (Parallel + O(1) Step)
    ├── jepa.py                 # Latent State Encoder, EMA Target, and Latent Predictor
    ├── heads.py                # Dual-System Action Gate, Primitive Classifier, Coords & Arg Head
    ├── model.py                # Full JED-M2 Unified Transformer-SSM Model (3x Stacked Blocks)
    ├── dataset.py              # WebArena/Mind2Web synthetic trajectory generator & collator
    └── losses.py               # Calibrated multi-task composite loss harness
```

---

## 5. Phase Roadmap & Verification Status

### [DONE] Phase 1: Core Attention & Routing Primitives
- [x] Global configuration dataclass (`JEDM2Config`): `hidden_dim=512`, `num_heads=8`, `q_lora_rank=256`, `kv_lora_rank=128`, `num_experts=8`, `top_k=2`.
- [x] Low-rank Multi-Head Latent Attention (`mla.py`): ~0.70M parameters.
- [x] Sparse Top-2 MoE (`moe.py`): ~12.59M parameters.
- [x] Combined Layer Footprint: ~13.29M parameters.
- [x] Verification Output: Passes `bfloat16` forward + backward pass on `cuda:0` at 158.68 MB peak VRAM.

### [DONE] Phase 2: Sequential Recurrence & Latent World Model
- [x] Implement `MambaBlock` in pure PyTorch (`mamba.py`): ~1.69M parameters, pure PyTorch causal 1D conv + selective scan ZOH discretization.
- [x] Autoregressive `step(...)` equivalence: O(1) constant-memory recurrent inference matching sequence forward mode (< 0.001 max diff in `bfloat16`).
- [x] Implement `JEPAWorldModel` with EMA target network and Smooth L1 + cosine loss (`jepa.py`): ~1.61M total parameters (~1.22M trainable, 0.39M frozen EMA).
- [x] Phase 2 footprint: ~3.31M parameters; Cumulative architecture (Phases 1-2): ~16.20M parameters.
- [x] Verification Output: Passes forward, recurrent step equivalence, backward gradient checks (zero NaNs/Infs), and target network gradient isolation on `cuda:0` in `bfloat16` at 922.45 MB peak VRAM.

### [DONE] Phase 3: Action Heads & Backbone Integration
- [x] Implement `heads.py` featuring dual-system action decoding:
  - `ActionGate`: `nn.Linear(hidden_dim, 1)` predicting action emission vs latent reasoning.
  - `ActionTypeHead`: MLP (`hidden_dim -> hidden_dim // 2 -> GELU -> Dropout -> num_action_types`) covering 6 primitives (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH).
  - `CoordinateHead`: Spatial pointer regressor (`hidden_dim -> hidden_dim // 4 -> GELU -> 2 -> Sigmoid`) for normalized $[0, 1]$ screen coordinates.
  - `ArgumentHead`: Linear projection (`hidden_dim -> vocab_size`) for tokenized arguments.
  - `NativeActionHead`: Composite module coordinating all four heads with inference decoding `predict_action(...)`.
- [x] Implement unified `JEDM2Model` in `model.py`:
  - `JEDM2Block`: Composite recurrent block interleaving RMSNorm -> MambaBlock -> RMSNorm -> MLA -> RMSNorm -> SparseTop2MoE with residual connections.
  - `JEDM2Model`: Full LAM backbone stacking 3 `JEDM2Block` layers (~44.96M params) over observation embeddings and `JEPAWorldModel`, feeding `NativeActionHead` and autoregressive LM reasoning head. Total parameters: ~50.97M total / ~22.26M active per token.
  - Autoregressive recurrent rollout `step(...)` method preserving layer-wise Mamba SSM buffers, conv buffers, and MLA KV-cache across all 3 layers.
- [x] Implement `verify_phase3.py` profiling multi-step rollout, dual action outputs, gradient backpropagation, parameter budget, and memory:
  - Passes forward pass, dual action output shapes, and bounded coordinates.
  - Passes 16-step recurrent rollout with state buffer integrity.
  - Passes backward pass asserting zero NaNs/Infs and JEPA EMA target network gradient isolation.
  - Peak GPU VRAM allocated: **1159.73 MB** (< 2048.00 MB / 2.0 GB target limit) on CUDA in `bfloat16` on NVIDIA GeForce RTX 5060 Laptop GPU.

### [DONE] Phase 4: Trajectory Dataset & Loss Harness
- [x] Synthetic multi-turn agent trajectory generator (`jed_m2/dataset.py`):
  - Trajectory tuples $\tau = (s_0, a_0, r_0, s_1, a_1, \dots, s_T)$ modeled after WebArena and Mind2Web.
  - Realistic DOM element trees, UI bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0, 1]$, and continuous pointer targets $(x, y) \in [0, 1]$.
  - Discrete action primitives (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH), string payload arguments, and System 2 reasoning thoughts.
  - Streaming PyTorch Dataset (`AgentTrajectoryDataset`) yielding trajectories on-demand without memory bottlenecks.
  - Dynamic batch collator (`TrajectoryCollator`) constructing 4D causal & padding attention masks, action token index mapping, component loss masks, and aligned JEPA state transition pairs.
- [x] Unified Composite Multi-Task Loss Harness (`jed_m2/losses.py`):
  - Reasoning loss ($\mathcal{L}_{\text{reason}}$): Autoregressive cross-entropy on thought tokens.
  - Action gate loss ($\mathcal{L}_{\text{gate}}$): Binary cross-entropy on action emission vs continued reasoning.
  - Action primitive loss ($\mathcal{L}_{\text{type}}$): Multi-class cross-entropy on discrete tool categories.
  - Coordinate regression loss ($\mathcal{L}_{\text{coord}}$): Smooth L1 / Huber loss on predicted $(x, y) \in [0, 1]$ screen coordinates.
  - Argument payload loss ($\mathcal{L}_{\text{arg}}$): Masked autoregressive cross-entropy on argument tokens.
  - JEPA latent loss ($\mathcal{L}_{\text{jepa}}$): Smooth L1 + cosine distance representation consistency.
  - MoE load-balancing loss ($\mathcal{L}_{\text{aux}}$): Auxiliary routing penalty from SparseTop2MoE.
  - Detailed logging metric dictionary reporting detached floats for all loss components, primitive accuracy, and coordinate MAE.
- [x] Public API exposure in `jed_m2/__init__.py`.
- [x] Implement and execute `verify_phase4.py`:
  - Passes synthetic trajectory generation and DOM structure assertions.
  - Passes dynamic batch collation and 4D causal lower-triangular mask checks.
  - Passes full model forward pass and composite multi-task loss evaluation.
  - Passes full backward pass asserting zero NaNs/Infs across all 161 active trainable parameters.
  - Passes JEPA EMA target network gradient isolation check (`grad is None`).
  - Passes zero-mask guardrails robustness check on sparse batches.
  - Peak GPU VRAM allocated: **797.33 MB** (< 2048.00 MB / 2.0 GB target limit) on CUDA in `bfloat16` on NVIDIA GeForce RTX 5060 Laptop GPU.

### [PENDING] Phase 5: Training Loop & VRAM Optimization
- [ ] Implement 8-bit AdamW / fused optimizer training loop with gradient accumulation.
- [ ] Activation checkpointing across recurrent blocks.
- [ ] Profile memory with batch size 2–4 and sequence length 2048 under active backpropagation within 6.0 GB ceiling.
- [ ] Direct Policy Optimization: Outline GRPO / DPO on trajectory outcome rewards.

---

## 6. Coding Agent Operational Directives
When implementing any task in this codebase:
1. **Type Annotations**: Annotate every function signature (`torch.Tensor`, `Optional`, `Tuple`, `Dict`).
2. **Tensor Shape Comments**: Document the expected input and output tensor shapes at every major projection step (e.g., `# [B, S, D] -> [B, S, q_lora_rank]`).
3. **Pure PyTorch Build Rule**: Do NOT import external compiled C++/CUDA packages (`mamba-ssm`, `causal-conv1d`).
4. **Standalone Verification**: Every phase must be accompanied by a self-contained `verify_phaseX.py` script that verifies:
   - Forward pass correctness
   - Backward pass integrity (assert no `NaN` or `Inf` in parameter gradients)
   - Peak allocated GPU VRAM using `torch.cuda.max_memory_allocated()`
   - Pure ASCII console output (avoid multi-byte Unicode box drawing characters to prevent Windows CP1252 `UnicodeEncodeError`).
5. **Preservation**: Do not delete, rewrite, or rename working components from previous phases without explicit instruction.