# Technical Design: Phase 2 - Sequential Recurrence & Latent World Model

## Context

Phase 1 established the baseline attention and routing stack: low-rank Multi-Head Latent Attention (MLA, ~0.70M params) and Sparse Top-2 Mixture of Experts (MoE, ~12.59M params), operating at 162.39 MB peak VRAM on an NVIDIA GeForce RTX 5060 Laptop GPU in `bfloat16`. 

Phase 2 introduces the sequential recurrence engine and the latent world model. Per project guidelines in `PROJECT_CONTEXT.md`, all components must remain pure PyTorch without external compiled C++/CUDA extensions to guarantee reliable operation on Windows 11 with Blackwell sm_120 architecture, while staying strictly within our parameter and VRAM budgets.

## Goals / Non-Goals

**Goals:**
- Implement `MambaBlock` in `jed_m2/mamba.py` in 100% native PyTorch.
- Support dual modes in `MambaBlock`: sequence-parallel `forward(...)` and constant-memory iterative `step(...)`.
- Guarantee mathematical equivalence between sequence `forward(...)` and iterative `step(...)` within `bfloat16` precision bounds.
- Implement `JEPAWorldModel` in `jed_m2/jepa.py` with `LatentStateEncoder`, momentum-updated `EMATargetEncoder` (stop-gradient), `LatentPredictor`, and composite Smooth L1 + cosine distance loss.
- Extend `JEDM2Config` in `jed_m2/config.py` with validated defaults for Mamba and JEPA.
- Expose all new modules in `jed_m2/__init__.py`.
- Implement `verify_phase2.py` verifying forward, backward (zero NaNs/Infs), step equivalence, parameter counts, and peak VRAM on CUDA.

**Non-Goals:**
- Custom C++/CUDA kernel compilation (`causal-conv1d` or `mamba-ssm`).
- Full unified model assembly (`jed_m2/model.py`), which is reserved for Phase 3.
- Action generation heads (`jed_m2/heads.py`), which is reserved for Phase 3.

## Decisions

### 1. Pure PyTorch Selective State Space Model (SSM)
- **Rationale**: External packages such as `mamba-ssm` require MSVC compilation and lack pre-built binary wheels for PyTorch on Blackwell sm_120 under Windows. Pure PyTorch tensor operations guarantee full cross-platform portability.
- **Formulation**:
  - Input $x \in \mathbb{R}^{B \times L \times D}$.
  - In-projection creates expanded representations $u, z \in \mathbb{R}^{B \times L \times D_{in}}$ where $D_{in} = \text{expand} \times D$ (e.g. $2 \times 512 = 1024$).
  - Causal 1D convolution on $u$ across sequence dimension with SiLU activation.
  - Linear projection of convolved features to input-dependent parameters $\Delta \in \mathbb{R}^{B \times L \times D_{in}}$, $B \in \mathbb{R}^{B \times L \times N}$, and $C \in \mathbb{R}^{B \times L \times N}$ ($N = d_{state} = 16$).
  - Discretization via zero-order hold: $\bar{A}_t = \exp(\Delta_t \odot A)$, $\bar{B}_t = \Delta_t \odot B_t$.
  - State update: $h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot u_t$.
  - Output projection: $y_t = (C_t \cdot h_t) + D \odot u_t$, gated by $\text{SiLU}(z_t)$, followed by out-projection back to $D$.
- **Alternatives Considered**: Parallel scan associative algorithms vs sequential loop. The sequential scan loop in PyTorch is straightforward, numerically stable, supports easy autograd differentiation, and mirrors the single-step recurrent update.

### 2. State Caching and Autoregressive `step(...)` Equivalence
- **Mechanism**:
  - `conv_state`: Buffer of shape `[B, D_in, d_conv]`. In step mode, new token features are appended, oldest are shifted out, and 1D depthwise convolution is computed via dot-product with conv weights and bias.
  - `ssm_state`: Buffer of shape `[B, D_in, d_state]`. In step mode, the recurrent update $h_t = \bar{A}_t h_{t-1} + \bar{B}_t u_t$ is executed for $t=1$.
  - Mathematically, rolling out $L$ sequential calls to `step(...)` with an initialized state reproduces the output of `forward(x)` within floating-point tolerance.

### 3. JEPA Architecture & Momentum Target Updates
- **Components**:
  - `LatentStateEncoder`: Maps observation vectors $o_t \in \mathbb{R}^{D_{obs}}$ (or token embeddings $D$) into normalized latent space $z_t \in \mathbb{R}^{D_{latent}}$ using an MLP with LayerNorm/RMSNorm and SwiGLU activations.
  - `EMATargetEncoder`: Cloned architecture from the state encoder. Forward passes run inside `torch.no_grad()`. Parameters are updated asynchronously via Polyak averaging:
    $$\theta_{\text{target}} \leftarrow \tau \theta_{\text{target}} + (1 - \tau) \theta_{\text{online}}$$
    with default decay $\tau = 0.99$.
  - `LatentPredictor`: Residual MLP taking concatenated $[z_t, a_t]$ and predicting $\hat{z}_{t+1} \in \mathbb{R}^{D_{latent}}$.
  - Loss function: Composite loss:
    $$\mathcal{L}_{\text{JEPA}} = \text{SmoothL1}(\hat{z}_{t+1}, \text{sg}(z_{t+1})) + \lambda_{\text{cos}} (1 - \text{cosine\_similarity}(\hat{z}_{t+1}, \text{sg}(z_{t+1})))$$

### 4. Configuration and Parameter Budget
- Extend `JEDM2Config` with:
  - `mamba_d_state: int = 16`
  - `mamba_d_conv: int = 4`
  - `mamba_expand: int = 2`
  - `mamba_dt_rank: int = 32`
  - `jepa_latent_dim: int = 256`
  - `jepa_ema_decay: float = 0.99`
  - `jepa_loss_cos_weight: float = 0.5`
- Parameter sizing:
  - `MambaBlock`: ~1.68M parameters.
  - `JEPAWorldModel`: ~0.65M parameters (online encoder ~0.26M, target ~0.26M untracked/EMA, predictor ~0.39M).
  - Total Phase 2 footprint: ~2.33M parameters. Cumulatively with Phase 1 (~13.29M), total architecture remains ~15.62M, comfortably under the ~45M budget.

## Risks / Trade-offs

- **[Risk 1] Discretization Overflow or Underflow in bfloat16**:
  - *Mitigation*: Parameterize $A$ as $-\exp(A\_log)$ with $A\_log$ initialized via log-spaced values, ensuring strictly negative eigenvalues and contractive recurrence. Clamp $\Delta$ after `softplus` to minimum value $10^{-3}$ and maximum $10^2$.
- **[Risk 2] Sequence vs Step Numerical Discrepancy**:
  - *Mitigation*: Ensure causal padding in `F.conv1d` strictly matches the right-aligned buffer update in `conv_state`. The test in `verify_phase2.py` explicitly asserts max error $\le 10^{-2}$ in bfloat16.
- **[Risk 3] Target Network Gradient Leakage**:
  - *Mitigation*: Explicitly set `requires_grad=False` on all target encoder parameters upon initialization and wrap target forward evaluations in `torch.no_grad()`.
