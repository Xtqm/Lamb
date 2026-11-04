# Proposal: Phase 14 - Real-World Mind2Web Corpus Ingestion, Pure-PyTorch Blelloch Associative Scan Acceleration, and Latent JEPA Rollout Planning

## Why

JED-M2-Medium (~273M Total / ~74M Active parameters) achieved 100% action accuracy and 0.0019 coordinate MAE on mock HTML templates at Step 2,500. However, the system faces three critical architectural and operational bottlenecks:
1. **Synthetic DOM Trap**: `RealWebTrajectoryDataset` iterates over 5 hard-coded in-memory mock HTML templates (`MOCK_TRACE_TEMPLATES`). It is not ingesting external real-world web corpuses, limiting generalization to genuine open-world web interaction.
2. **Mamba SSM Kernel Dispatch Bottleneck**: The selective state space recurrence in `jed_m2/mamba.py` is implemented as a sequential Python `for t in range(seq_len):` loop. At sequence length $T=2048$ across 16 layers, this causes 32,768 Python-to-GPU kernel dispatches per pass, taking ~65s on backward pass with checkpointing and limiting throughput to ~200 tok/s.
3. **Passive World Model**: The JEPA world model currently operates solely as an auxiliary regularization loss during training rather than an active model-predictive rollout planner in `jed_m2/agent.py`.

Phase 14 resolves these bottlenecks by ingesting the genuine Mind2Web corpus (10,092 tasks across 137 domains), implementing a native pure-PyTorch parallel associative prefix scan (Blelloch scan) in `MambaBlock` for 5x-8x throughput gains without external CUDA/Triton binaries, and activating JEPA as a model-predictive rollout planner in `JEDM2Agent`.

## What Changes

- **Automated Mind2Web Corpus Ingestion & Downloader**:
  - Implement `data/download_mind2web.py` to stream and unpack official Mind2Web (`osunlp/Mind2Web`) raw tasks and DOM annotation splits (train, val, test across 137 real-world domains).
  - Strictly configure `.gitignore` so that `data/raw_mind2web/`, `data/*.parquet`, `data/*_cache/`, and raw snapshot artifacts are never committed to git.
  - Refactor `RealWebTrajectoryDataset` in `jed_m2/real_dataset.py` to parse raw Mind2Web HTML snapshots, extracting real interactive candidate elements, accessibility roles, and normalized bounding box centers $(x, y) \in [0, 1]^2$, with robust fallback to procedural templates when offline.
- **Pure-PyTorch Parallel Prefix Scan (Blelloch Scan) in Mamba SSM**:
  - Replace the sequential Python `for t in range(seq_len):` loop in `MambaBlock.forward()` (`jed_m2/mamba.py`) with a parallel associative prefix scan algorithm implemented purely in native PyTorch tensor operations.
  - Maintain 100% pure-PyTorch portability without compiled C++/CUDA extensions, Triton, or external packages (`mamba-ssm`, `causal-conv1d` forbidden).
  - Reduce backward pass latency at $T=1024/2048$ by 5x-8x, boosting throughput from ~200 tok/s to $>1,200$ tok/s while maintaining numerical parity ($< 1\text{e-}5$ difference) with the sequential formulation.
- **Active Model-Predictive Planning with JEPA**:
  - In `JEDM2Agent` (`jed_m2/agent.py`), implement an active latent rollout mechanism:
    Before committing to an action, evaluate $\hat{z}_{t+1} = \text{Predictor}(z_t, a_i)$ across candidate discrete actions $a_i \in \{\text{CLICK}, \text{TYPE}, \text{SCROLL}, \text{NAVIGATE}\}$, selecting the action that minimizes latent cosine distance to the goal embedding $z_{\text{goal}}$.
  - Elevate JEPA from a passive loss to a functional predictive planning system.
- **Benchmarking and Verification**:
  - Update `eval_benchmarks.py` to support evaluating Step 2,500 baseline vs Phase 14 models on genuine held-out Mind2Web test splits.
  - Create `verify_phase14.py` verifying Blelloch scan parity, real Mind2Web streaming and collation under 6.0 GB VRAM, JEPA latent rollout action selection, and export of comparative evaluation reports.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `real-web-trajectory-dataset`: Requirements updated to mandate automated downloading, streaming, and parsing of official Mind2Web tasks and raw DOM snapshots across 137 domains, real candidate interactive element extraction, coordinate normalization, and strict zero-leakage cache git hygiene.
- `mamba-ssm`: Requirements updated to mandate pure-PyTorch parallel associative prefix scan (Blelloch scan) in forward sequence processing with logarithmic depth, achieving $>1,200$ tok/s throughput while preserving mathematical parity ($< 1\text{e-}5$) with step-level recurrent updates.
- `interactive-agent-runtime`: Requirements updated to implement active JEPA latent rollout predictive planning in `JEDM2Agent`, scoring candidate actions against goal embeddings using the action-conditioned predictor prior to environmental dispatch.
- `evaluation-benchmark-harness`: Requirements updated to support genuine Mind2Web dataset test evaluation splits, comparative checkpoint analysis, and latency profiling under the 6.0 GB VRAM limit.

## Impact

- **Affected Code**: `jed_m2/mamba.py`, `jed_m2/real_dataset.py`, `jed_m2/agent.py`, `eval_benchmarks.py`, `.gitignore`.
- **New Files**: `data/download_mind2web.py`, `verify_phase14.py`.
- **Dependencies**: 100% native PyTorch and Python standard library (no external C++/CUDA/Triton binaries). HuggingFace `datasets` or direct HTTP streaming for Mind2Web downloader.
- **Hardware Budget**: Strict adherence to the 6.0 GB VRAM limit on consumer NVIDIA RTX 5060 Laptop GPU (8GB VRAM).
