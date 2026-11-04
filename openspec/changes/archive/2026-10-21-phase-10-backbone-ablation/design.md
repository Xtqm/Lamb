# Design

## Context
See `proposal.md` for the motivation and high-level scope of the Phase 10 scientific 4-way ablation study. JED-M2 incorporates four core architectural pillars:
1. Mamba Selective State Space Model (SSM) recurrence (`mamba.py`) for $O(1)$ constant-memory sequential state tracking.
2. Multi-Head Latent Attention (`mla.py`) with low-rank Key-Value compression ($c_{KV}=128, K_{\text{rope}}=32$) and $W_{UK}$ query absorption.
3. Sparse Top-2 Mixture-of-Experts (`moe.py`) across 8 SwiGLU experts for high model capacity with low active compute.
4. Contextual JEPA Latent World Model (`jepa.py`) predicting forward environment state transitions.

To rigorously isolate and demonstrate the empirical necessity of each pillar for academic publication, we construct four parameter-matched baseline architectures calibrated to the same total parameter budget (~50.0M total parameters). All implementations adhere to the project's strict constraints: 100% native PyTorch without external compiled C++/CUDA packages, native `torch.bfloat16` execution, and a strict 6.0 GB VRAM ceiling on the host NVIDIA RTX 5060 Laptop GPU.

## Goals / Non-Goals

**Goals:**
- Provide clean, modular architectural toggles in `JEDM2Config` (`attn_type`, `ffn_type`, `use_mamba`, `use_jepa`, `dense_intermediate_dim`) without breaking backward compatibility for the default full JED-M2 model.
- Implement reusable baseline modules in `jed_m2/baselines.py`: `StandardMHA` (standard multi-head attention with decoupled Q/K/V projections, RoPE, and per-head uncompressed KV-caching) and `DenseSwiGLUFFN` (dense gated SwiGLU feed-forward network).
- Implement factory functions for the 4 calibrated baselines with parameter budget parity ($\pm 5\%$ of 50.0M):
  - Baseline A (Pure Transformer): 4 layers, Standard MHA + Dense SwiGLU FFN (~50.5M params).
  - Baseline B (Hybrid Attention-SSM): 3 layers, Mamba SSM + MLA + Dense SwiGLU FFN (~50.2M params).
  - Baseline C (JED-M2 without JEPA): 3 layers, Mamba SSM + MLA + Sparse Top-2 MoE (~50.9M params).
  - Baseline D (Full JED-M2 with Connected JEPA): 3 layers, Mamba SSM + MLA + Sparse MoE + Connected JEPA (~53.0M params).
- Implement `run_ablation.py` CLI benchmark script running 4 controlled experiments:
  - Experiment 1: Memory Scaling Curve across 16, 32, 64, 128 rollout turns ($O(1)$ vs $O(N)$).
  - Experiment 2: Inference Step Latency in `bfloat16` (mean, median, p95).
  - Experiment 3: Controlled 300-step multi-task convergence training on identical batches.
  - Experiment 4: Automated export to `ABLATION_REPORT.md` (Markdown tables, ASCII charts, LaTeX table snippets).
- Implement `verify_phase10.py` verifying all 4 baseline architectures, parameter parity, forward/backward passes, and 6.0 GB VRAM ceiling compliance.

**Non-Goals:**
- External compiled C++/CUDA kernels (e.g. `flash-attn`, `causal-conv1d`, `mamba-ssm`) — pure PyTorch is strictly maintained for Windows MSVC / Blackwell compatibility.
- Training models to full convergence over millions of tokens — Experiment 3 is a controlled 300-step comparative convergence benchmark on identical seeded batches to establish relative loss trajectories and efficiency.
- Modifying the public inference interface of `JEDM2Model` — `forward()` and `step()` retain their existing signatures and output structures.

## Decisions

### 1. Architectural Toggles and Parameter Parity Calibration
**Decision**: Parameterize `JEDM2Config` with explicit component discriminators:
```python
attn_type: Literal["mla", "mha"] = "mla"
ffn_type: Literal["moe", "dense"] = "moe"
use_mamba: bool = True
use_jepa: bool = True
dense_intermediate_dim: int = 1024
```
**Rationale**: 
- Rather than maintaining 4 separate divergent model classes, `JEDM2Block` and `JEDM2Model` conditionally instantiate and invoke components based on these toggles. This ensures all baselines share the exact same embeddings, vision encoder, normalization layers, and action heads.
- In dense FFN models (Baselines A and B), `dense_intermediate_dim` is calibrated to absorb the parameter difference resulting from disabling MoE or Mamba:
  - Base overhead (embeddings, vision patch encoder, action heads, norm): ~5.88M params.
  - Baseline A (4 layers): Standard MHA (~1.05M) + Dense SwiGLU with $D_{\text{dense}}=6592$ (~10.13M) = ~11.18M per layer $\times 4 \approx 44.72\text{M} + 5.88\text{M} \approx 50.60\text{M}$ total params.
  - Baseline B (3 layers): Mamba SSM (~1.69M) + MLA (~0.70M) + Dense SwiGLU with $D_{\text{dense}}=8064$ (~12.39M) = ~14.78M per layer $\times 3 \approx 44.34\text{M} + 5.88\text{M} \approx 50.22\text{M}$ total params.
  - Baseline C (3 layers): Mamba SSM (~1.69M) + MLA (~0.70M) + Sparse Top-2 MoE (~12.59M) = ~14.98M per layer $\times 3 \approx 44.95\text{M} + 5.88\text{M} \approx 50.83\text{M}$ total params.
  - Baseline D (3 layers + JEPA): Baseline C + JEPAWorldModel (~1.61M) $\approx 52.44\text{M} - 53.0\text{M}$ total params.
- All models fall strictly within $\pm 5\%$ of the 50.0M target, guaranteeing a fair parameter-matched ablation study.
**Alternatives Considered**:
- Keeping $D_{\text{dense}}=1024$ without calibration: Would cause dense baselines to have only ~15M total params, confounding architectural advantages with parameter count deficit.

### 2. Standard Multi-Head Attention (`StandardMHA`) Design
**Decision**: Implement `StandardMHA` in `jed_m2/baselines.py` matching standard Transformer self-attention:
- Separate projections: $W_Q, W_K, W_V \in \mathbb{R}^{D \times D}$ and $W_O \in \mathbb{R}^{D \times D}$ with $H=8$ heads and $d_h=64$.
- Rotary Positional Embeddings (RoPE) applied to Q and K.
- Sequence forward pass using `torch.nn.functional.scaled_dot_product_attention`.
- Incremental `step(...)` method maintaining an uncompressed recurrent KV-cache: stores full $K_{\text{cache}}, V_{\text{cache}} \in \mathbb{R}^{B, H, S, d_h}$, growing linearly with turn horizon $T$.
**Rationale**:
- Providing uncompressed KV-caching directly exposes the theoretical and practical difference between standard MHA ($O(N)$ memory growth) and JED-M2's MLA ($O(1)$ compressed latent caching with $W_{UK}$ absorption).
**Alternatives Considered**:
- PyTorch `nn.MultiheadAttention`: Does not cleanly expose manual step-by-step KV-cache tensors or decoupled RoPE matching our token indexing.

### 3. Unified Recurrent Step State Representation
**Decision**: In `JEDM2Block.step(x_t, state, pos)`:
- If `use_mamba=True`, maintain `(ssm_state, conv_state)`. If `use_mamba=False`, pass `(None, None)`.
- If `attn_type="mla"`, `kv_cache` holds compressed latent tuple `(c_kv, k_rope)`.
- If `attn_type="mha"`, `kv_cache` holds standard uncompressed tuple `(k_cache, v_cache)`.
- Return format remains `(out_t, next_state)` where `next_state = (next_ssm_state, next_conv_state, next_kv_cache)`.
**Rationale**:
- Maintains full polymorphic compatibility across the entire stack (`JEDM2Model.step`, `JEDM2Agent`, and `run_ablation.py`).

### 4. Controlled Multi-Task Convergence Setup (Experiment 3)
**Decision**:
- Generate identical procedural trajectory batches using a fixed seed (`seed=42`) with `SyntheticTrajectoryGenerator`.
- Micro-batch size 2, gradient accumulation 2 (effective batch size 4), sequence length 512, 300 steps.
- Optimizer: Native PyTorch AdamW with fused implementation (`torch.optim.AdamW(..., fused=True, lr=3e-4)`).
- Log step-by-step composite loss, reasoning loss, action primitive accuracy, and coordinate MAE.
**Rationale**:
- 300 steps is sufficient to observe initial convergence velocity and gradient stability across all 4 architectures on the host GPU without exceeding reasonable runtime or the 6.0 GB VRAM limit.

### 5. Automated Publication Artifacts Export (Experiment 4)
**Decision**: Write `ABLATION_REPORT.md` dynamically with:
1. Executive Summary & Parameter Parity Table.
2. Memory Scaling Table (16, 32, 64, 128 steps) + ASCII scaling curves showing $O(1)$ vs $O(N)$ slopes.
3. Inference Latency Percentile Table (mean, median, p95).
4. 300-Step Convergence Metrics Table (final loss, primitive accuracy, coordinate MAE).
5. Self-contained LaTeX table snippets (`\begin{table}...\end{table}`) ready for direct inclusion in LaTeX paper manuscripts.
**Rationale**:
- Eliminates manual error-prone copying of experimental numbers and ensures reproducibility.

## Risks / Trade-offs

- **[Risk] Dense FFN parameter blowout in VRAM during sequence forward pass**:
  $D_{\text{dense}}=8064$ creates large intermediate activation tensors $[B, S, 8064]$.
  *Mitigation*: At sequence length 512 with batch size 2, intermediate activations require only $\sim 2 \times 512 \times 8064 \times 2 = 16.5\text{ MB}$ per layer in BF16, well within the 6.0 GB budget. We verify peak allocated memory in `verify_phase10.py`.
- **[Risk] MHA uncompressed KV-cache OOM at extreme horizons**:
  Standard MHA cache grows linearly with token count.
  *Mitigation*: Horizons up to 128 steps require $128 \times 1024 \times 2 = 256\text{ KB}$ per layer at batch size 1 in BF16, which is small in absolute bytes but easily measurable in growth slope relative to MLA's constant 160 floats/token.
- **[Risk] Windows terminal Unicode character corruption**:
  Tables or ASCII plots containing UTF-8 box characters can fail with `UnicodeEncodeError` on Windows CP1252 consoles.
  *Mitigation*: Use strictly 7-bit ASCII characters (`+`, `-`, `|`, `*`, `#`) for all console outputs and ASCII plots.
