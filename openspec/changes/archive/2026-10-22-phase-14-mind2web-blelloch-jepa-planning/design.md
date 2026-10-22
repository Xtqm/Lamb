# Design: Phase 14 - Real-World Mind2Web Corpus Ingestion, Pure-PyTorch Blelloch Associative Scan Acceleration, and Latent JEPA Rollout Planning

## Context

JED-M2-Medium (~273M Total / ~74M Active parameters, hybrid Mamba SSM + MLA + MoE) achieved 100% action accuracy and 0.0019 coordinate MAE on mock HTML templates at Step 2,500. However, three foundational bottlenecks constrain its deployment as an open-world web agent:
1. `RealWebTrajectoryDataset` is trapped in 5 synthetic mock HTML templates (`MOCK_TRACE_TEMPLATES`).
2. Mamba SSM's forward pass sequentially loops over sequence length $T$ (`for t in range(seq_len):`), dispatching 32,768 kernels across 16 layers and capping training throughput to ~200 tok/s.
3. The JEPA world model acts only as a passive auxiliary loss rather than an active model-predictive planner during inference.

This design introduces genuine Mind2Web ingestion, pure-PyTorch parallel prefix scan (Blelloch scan), and latent predictive planning while strictly adhering to the 6.0 GB VRAM ceiling on consumer RTX 5060 hardware without external C++/CUDA/Triton binaries.

## Goals / Non-Goals

**Goals:**
- Provide automated downloading and streaming of the official Mind2Web dataset (`osunlp/Mind2Web`) across 137 real domains with zero git leakage.
- Refactor `RealWebTrajectoryDataset` to parse genuine HTML/DOM snapshots, interactive candidates, and bounding boxes, with graceful offline template fallback.
- Replace the sequential Python loop in `MambaBlock.forward()` with a pure-PyTorch parallel associative prefix scan, achieving $>1,200$ tok/s throughput ($5\times$ to $8\times$ speedup) and maintaining $< 1\text{e-}5$ numerical parity.
- Implement active JEPA latent rollout planning in `JEDM2Agent`, re-ranking candidate actions based on predictive cosine distance to the goal representation.
- Provide end-to-end benchmarking in `eval_benchmarks.py` and a comprehensive verification suite in `verify_phase14.py`.

**Non-Goals:**
- Compiling external C++/CUDA or Triton kernels (`mamba-ssm`, `causal-conv1d` remain forbidden).
- Retraining the medium model from scratch in this planning phase (focus is architecture, data ingestion, planning algorithm, and verification).
- WebArena live server deployment (offline Mind2Web traces are prioritized for phase 14).

## Decisions

### Decision 1: Pure-PyTorch Logarithmic Associative Scan (Blelloch Prefix Scan)
- **Problem**: The sequential recurrence in `MambaBlock.forward()`:
  $$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot u_t$$
  causes $O(T)$ sequential GPU launches. For $T=2048$, this is 2048 kernel dispatches per layer.
- **Solution**: Implement an associative operator $(a_i, b_i) \bullet (a_j, b_j) = (a_j \odot a_i, a_j \odot b_i + b_j)$ executed via tree-based parallel prefix scan (Blelloch scan / log-step scan) in pure PyTorch:
  - In log-depth steps ($k = 1, 2, 4, \dots < L$), vectors are combined along the sequence dimension using tensor slicing and elementwise operations.
  - Number of kernel launches drops from $T$ to $\lceil \log_2 T \rceil$ (e.g. 11 launches for $T=2048$, a $>180\times$ reduction in dispatch overhead).
- **Alternatives Considered**:
  - *Triton or CUDA C++ kernel*: Violates the strict pure-PyTorch portability constraint on Windows/cross-platform environments.
  - *PyTorch `torch.cumsum`*: Cannot directly express state space decay $\bar{A}_t \in (0, 1)$ with variable input scaling.
  - *Chunked sequential scan*: Reduces dispatch by $K$ but remains sequential across chunks. Full logarithmic parallel scan achieves optimal GPU utilization.

### Decision 2: Mind2Web Dataset Ingestion Architecture with Offline Fallback
- **Problem**: Large web DOM trees (10,092 tasks, 137 websites) require heavy disk space and memory if loaded naively.
- **Solution**:
  - `data/download_mind2web.py`: Stream dataset from Hugging Face (`osunlp/Mind2Web`) in chunks, storing compressed task annotations and pruned HTML snapshots in `data/raw_mind2web/`.
  - `.gitignore`: Add strict exclusions for `data/raw_mind2web/`, `data/*.parquet`, and `data/*_cache/`.
  - `RealWebTrajectoryDataset`: When `data/raw_mind2web/` is present, parse real task instances, extracting interactive elements, normalized bounding box centers $(x, y) \in [0, 1]^2$, and candidate roles. If the directory is absent (e.g. fresh clone or offline environment), seamlessly fall back to procedural templates without failing tests or crashing.
- **Alternatives Considered**:
  - *Mandating local raw dataset download before running*: Breaks CI and offline verification scripts. Dual-mode fallback ensures reliability everywhere.

### Decision 3: JEPA Model-Predictive Rollout Planning in `JEDM2Agent`
- **Problem**: In `JEDM2Agent.act()`, System 1 decodes actions via greedy argmax over action heads without verifying whether the predicted transition leads toward the task objective.
- **Solution**:
  - Before committing to an action, extract current latent state $z_t = \text{Encoder}(o_t)$ and goal representation $z_{\text{goal}} = \text{Encoder}(o_{\text{goal}})$.
  - For top candidate actions $a_i \in \{\text{CLICK}, \text{TYPE}, \text{SCROLL}, \text{NAVIGATE}\}$:
    $$\hat{z}_{t+1}^{(i)} = \text{Predictor}(z_t, a_i)$$
    $$d_{\text{goal}}(a_i) = 1 - \frac{\hat{z}_{t+1}^{(i)} \cdot z_{\text{goal}}}{\|\hat{z}_{t+1}^{(i)}\|_2 \|z_{\text{goal}}\|_2}$$
  - Re-rank or select action:
    $$\text{Score}(a_i) = \text{logits}[a_i] - \lambda \cdot d_{\text{goal}}(a_i)$$
    where $\lambda \ge 0$ is the predictive planning weight.
- **Alternatives Considered**:
  - *Full multi-step Monte Carlo Tree Search (MCTS)*: Too computationally expensive for interactive real-time inference under 6.0 GB VRAM. 1-step latent rollout provides immediate model-predictive grounding with minimal latency overhead ($< 5$ ms).

## Risks / Trade-offs

- **[Risk: Numerical precision divergence in parallel scan]** → In `bfloat16`, repeated multiplications along sequence length can suffer numerical drift.
  - *Mitigation*: Perform the associative scan internal accumulators in `float32` (standard practice in Mamba) before casting back to input dtype, ensuring $< 1\text{e-}5$ difference against sequential recurrence in float32 and $< 10^{-2}$ in bfloat16.
- **[Risk: Mind2Web dataset size & download time]** → Full raw Mind2Web dataset is several gigabytes.
  - *Mitigation*: Provide configurable sample limits (e.g. `--num_samples 200` or `--split test`) and streaming support in `download_mind2web.py`, enabling fast verification and lightweight caching.
- **[Risk: Memory overhead during associative scan]** → Parallel prefix scan maintains intermediate states across power-of-two steps.
  - *Mitigation*: Operate in-place or reuse tensor buffers; verify peak VRAM remains strictly $< 6.0$ GB on RTX 5060.

## Migration Plan

1. Update `.gitignore` to prevent any data artifacts from leaking into git.
2. Implement `data/download_mind2web.py` for streaming Mind2Web task annotations and snapshots.
3. Implement `pure_torch_associative_scan` and integrate into `MambaBlock.forward()` in `jed_m2/mamba.py`, retaining `step()` for autoregressive decoding.
4. Refactor `RealWebTrajectoryDataset` in `jed_m2/real_dataset.py` with real Mind2Web ingestion and synthetic fallback.
5. Implement active JEPA rollout planning in `JEDM2Agent` (`jed_m2/agent.py`).
6. Update `eval_benchmarks.py` and implement `verify_phase14.py`.
