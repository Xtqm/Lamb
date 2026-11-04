# Design

## Context
See `proposal.md` for motivation and background.

Currently, JED-M2 relies on synthetic multi-turn trajectory generators (`jed_m2/dataset.py`) that construct synthetic DOM trees (6 to 10 elements) and synthetic screenshot images with Gaussian noise. While effective for initial architectural validation, these do not represent the complexity of real-world websites, where raw HTML trees contain over 10,000 DOM nodes, diverse CSS frameworks, nested iframe components, and varied bounding box coordinates.

The system must run pretraining on the NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) in pure PyTorch without external compiled C++/CUDA packages or heavy unvetted dependencies. Training must operate strictly under a 6.0 GB VRAM allocation ceiling in `torch.bfloat16` precision across both JED-M2-Nano (51M) and JED-M2-Medium (175M).

## Goals / Non-Goals

**Goals:**
- Provide a pure-PyTorch, dependency-light real web trajectory ingestion pipeline in `jed_m2/real_dataset.py`.
- Implement a fast streaming DOM parser using Python's standard `html.parser` to prune massive (10,000+ node) HTML pages down to salient interactive elements (`button`, `input`, `a`, `select`, `textarea`, ARIA interactive roles).
- Extract and normalize bounding boxes $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}] \in [0, 1]^2$ and compute normalized center pointer coordinates $(x, y)$.
- Support persistent on-disk binary caching (`.pt`) for pre-parsed trajectories to eliminate DOM parsing overhead during multi-epoch training.
- Collate variable-length multi-turn web interaction traces into unified `TrajectoryBatch` containers with aligned dual-process loss masks (reasoning NLL, action type CE, coordinate Huber, argument CE, and JEPA latent consistency).
- Provide a production-grade pretraining script (`train_pretrain.py`) supporting scaled runs (5,000 to 10,000 steps), mixed precision `torch.bfloat16`, gradient accumulation (effective batch 16-32), live telemetry logging, and validation evaluation.
- Implement a comprehensive verification suite (`verify_phase12.py`) covering DOM parsing, tensor collation, 100-sample convergence, and peak VRAM limits.

**Non-Goals:**
- Scraping live websites over the network during training (traces are ingested from offline benchmark datasets).
- Integrating external heavy rendering engines (e.g. headless Chrome) into the training loop; offline bounding boxes and accessibility trees are pre-extracted in the trajectory traces.
- Modifying the underlying model architectures (`JEDM2Model` and recurrent blocks) beyond utilizing their existing Nano and Medium configurations.

## Decisions

### Decision 1: Lightweight Standard Library DOM Tree Parser vs. Heavy Third-Party Parsers
- **Choice**: Use Python's built-in `html.parser.HTMLParser` enhanced with an interactive node collector and bounding box extractor in `jed_m2/real_dataset.py`.
- **Rationale**: Keeps the codebase 100% compliant with the project's strict rule: zero external non-PyTorch dependencies. `html.parser` processes megabyte-scale HTML strings in milliseconds without memory leaks, C-extension compilation issues on Windows, or external wheel dependencies.
- **Pruning Strategy**:
  - Keep tags: `button`, `input`, `a`, `select`, `textarea`, `option`.
  - Keep elements with interactive roles: `role="button"`, `role="link"`, `role="textbox"`, `role="checkbox"`, `role="combobox"`, `role="menuitem"`, `role="tab"`.
  - Extract text and bounding box attributes (`bbox`, `data-bbox`, or external bounding box metadata maps).
  - Cap maximum candidate elements per turn (e.g., top 64 to 128 elements) to prevent sequence length overflow.
- **Alternatives Considered**:
  - *BeautifulSoup4 / lxml*: Requires external C-libraries (libxml2) which can fail to build or introduce binary incompatibilities on Windows environments.
  - *Regex-only extraction*: Prone to malformed HTML failures and nested tag parsing errors.

### Decision 2: Normalized Coordinate System and Spatial Target Matching
- **Choice**: Standardize all bounding boxes to $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}] \in [0, 1]^2$ normalized by viewport dimensions $(W, H)$. Compute the target click center as $(x, y) = ((x_{\text{min}} + x_{\text{max}})/2, (y_{\text{min}} + y_{\text{max}})/2)$.
- **Rationale**: Aligns directly with JED-M2's continuous coordinate head which uses Sigmoid activations to emit $(x, y) \in [0, 1]^2$ and Smooth L1 / Huber loss.
- **Alternatives Considered**:
  - *Discrete pixel coordinate tokens*: Expands token sequence lengths and vocabulary size significantly.

### Decision 3: Persistent Disk Caching Architecture
- **Choice**: Implement `RealWebTrajectoryDataset` with two-tier loading:
  1. Parse raw offline dataset traces on first access or pre-build pass.
  2. Cache parsed, tokenized `TrajectorySample` objects in PyTorch binary format (`.pt`) indexed by split (`train`, `val`, `test`).
- **Rationale**: Real HTML traces can take several seconds to parse if done repeatedly across epochs. Disk caching ensures subsequent training runs and multi-epoch iterations achieve high token throughput (> 5,000 tok/s).
- **Alternatives Considered**:
  - *In-memory caching of all trajectories*: Risk of out-of-memory on system RAM for large trajectory collections.
  - *On-the-fly parsing without cache*: Becomes CPU-bound, causing GPU starvation.

### Decision 4: Trajectory Batch Collation and Loss Mask Alignment
- **Choice**: Build `RealTrajectoryCollator` that accepts variable-length turns, tokenizes them with `SimpleLAMTokenizer`, pads sequences to the micro-batch maximum sequence length (or `max_seq_len`), and constructs aligned masks:
  - `reason_loss_mask`: 1 for `<thought> ... </thought>` tokens, 0 elsewhere.
  - `action_indices` & `action_type_mask`: Pointers to action emission token positions.
  - `coord_targets` & `coord_loss_mask`: Normalized coordinates $(x, y)$ for spatial actions (`CLICK`).
  - `arg_targets` & `arg_loss_mask`: Token IDs for argument values in `TYPE` or `NAVIGATE`.
  - `jepa_obs_t`, `jepa_action_t`, `jepa_obs_next`: Latent observation embeddings across consecutive turns for JEPA consistency loss.
- **Rationale**: Maintains exact parity with the interface expected by `JEDM2CompositeLoss` in `jed_m2/losses.py`.

### Decision 5: Scaled Pretraining Engine Architecture (`train_pretrain.py`)
- **Choice**: Implement a standalone CLI script `train_pretrain.py` featuring:
  - Fused AdamW optimizer (`torch.optim.AdamW(..., fused=True)`) or 8-bit AdamW fallback.
  - Cosine learning rate decay with linear warmup.
  - Gradient accumulation with micro-batch size 2 and effective batch size 16 to 32.
  - Mixed precision `torch.bfloat16`.
  - Activation checkpointing enabled across recurrent blocks.
  - Real-time telemetry logging: Decomposed losses ($L_{\text{reason}}$, $L_{\text{type}}$, $L_{\text{coord}}$, $L_{\text{arg}}$, $L_{\text{jepa}}$, $L_{\text{aux}}$), token processing throughput (tok/s), gradient norms, and live GPU VRAM tracking.
  - Automated validation evaluation passes and checkpoint saving at configured intervals.
- **Rationale**: Isolates pretraining orchestration while leveraging existing `JEDM2Model` and `JEDM2CompositeLoss` modules.

## Risks / Trade-offs

- **[Risk] Massive HTML Documents Causing Parser Hang or Memory Spikes**:
  - *Mitigation*: The DOM parser enforces a maximum node depth, skips `<script>`, `<style>`, and `<svg>` blocks during tag scanning, and caps the maximum number of extracted interactive elements per turn to 128.
- **[Risk] Missing Bounding Boxes in Certain Web Traces**:
  - *Mitigation*: For traces where explicit bounding boxes are omitted, synthesize deterministic normalized bounding boxes based on DOM tree hierarchy order, or mask out coordinate loss for those turns (`coord_loss_mask = False`).
- **[Risk] VRAM Exceeding 6.0 GB during Medium Pretraining (175M Parameters)**:
  - *Mitigation*: Maintain micro-batch size of 2, enable activation checkpointing across all recurrent blocks, and enforce native `torch.bfloat16` precision.
- **[Risk] Slow Initial Cache Generation**:
  - *Mitigation*: Implement offline caching generation with progress reporting, caching per-sample or per-shard files so interrupted pre-processing can resume incrementally.
