# real-web-trajectory-dataset Specification

## Purpose
Provides ingestion, lightweight DOM pruning, spatial bounding box normalization, persistent caching, and dynamic PyTorch batch collation for real-world web interaction trajectories derived from Mind2Web and WebArena traces.

## Requirements

### Requirement: Offline Real-World Web Trajectory Ingestion
The system SHALL ingest real-world offline web interaction traces formatted after Mind2Web and WebArena datasets, representing multi-turn web navigation and task completion episodes across real websites, with graceful fallback to procedural templates when offline.

#### Scenario: Offline web interaction trace parsing
- **WHEN** raw offline web trajectory records are loaded
- **THEN** the system extracts task goals, multi-turn observations, interactive DOM elements, discrete actions, screen coordinates, and rationale thoughts

#### Scenario: Genuine Mind2Web raw snapshot parsing
- **WHEN** raw Mind2Web JSON files and raw DOM HTML files are supplied to `RealWebTrajectoryDataset`
- **THEN** the dataset parses candidate DOM elements from both positive target candidates (`pos_candidates`) and negative distractor candidates (`neg_candidates`), extracts bounding boxes for all identifiable candidates to construct realistic multi-element candidate pools, normalizes pointer coordinates to $[0, 1]^2$, and maps action operations to discrete action types

#### Scenario: Offline fallback when raw corpus is absent
- **WHEN** the raw Mind2Web directory is not downloaded or accessible
- **THEN** `RealWebTrajectoryDataset` logs an informational notice and falls back to deterministic procedural HTML trace templates without failing initialization

### Requirement: Intelligent DOM Tree Pruning and Interactive Candidate Filtering
The system SHALL parse raw HTML trees containing 10,000+ nodes and prune non-interactive noise, retaining only candidate interactive elements including buttons, inputs, links, selects, and textareas.

#### Scenario: Pruning large DOM tree
- **WHEN** a raw web page HTML string containing over 10,000 DOM nodes is processed
- **THEN** the parser prunes passive containers and outputs a filtered list of interactive candidate elements without memory exhaustion or stack overflow

### Requirement: Normalized Spatial Bounding Box and Coordinate Extraction
The system SHALL extract element bounding box coordinates $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ normalized in $[0, 1]^2$ and compute normalized center pointer coordinates $(x, y)$ for each interactive element, structured as canonical `UIElement` instances with `InteractiveElement` maintained as a backward-compatible alias. Dense observation features for JEPA and model conditioning SHALL be produced using the centralized `compute_obs_features` utility.

#### Scenario: Bounding box normalization
- **WHEN** raw pixel or viewport bounding boxes are provided
- **THEN** the system normalizes coordinates to $[0, 1]^2$ and computes center points matching target click locations

#### Scenario: InteractiveElement type aliasing and interoperability
- **WHEN** downstream code accesses elements via `InteractiveElement` or `UIElement`
- **THEN** both types resolve to the same underlying canonical class and provide equivalent attributes, centers, and `to_dom_string()` representations

### Requirement: Sequence Tokenization and Intent Representation
The system SHALL tokenize task intent strings, pruned accessibility DOM representations, and reasoning thoughts into standard sequence tensors matching model vocabulary configurations.

#### Scenario: Multimodal sequence tokenization
- **WHEN** task intent, DOM element descriptions, and thoughts are passed to the tokenizer
- **THEN** standard token ID sequences are emitted with dedicated delimiters for observations, thoughts, actions, and arguments

### Requirement: Persistent Disk Caching and Dataset Splitting
The system SHALL provide a dataset interface supporting stratified train, validation, and test splits with persistent disk caching for rapid replay and zero-recomputation loading, while ensuring cache directories are quarantined from git tracking.

#### Scenario: Disk cache hit and split partition
- **WHEN** the dataset is initialized with a cache path and split name
- **THEN** processed trajectories are retrieved from disk cache without re-parsing raw HTML, and disjoint samples are returned for train, validation, and test partitions

#### Scenario: Cache invalidation and rebuilding
- **WHEN** `force_rebuild=True` is passed to `RealWebTrajectoryDataset`
- **THEN** existing disk cache files are bypassed, raw trajectories are re-parsed and re-tokenized, and a refreshed cache file is atomically written

### Requirement: Dynamic Batch Collation and Multi-Task Loss Mask Alignment
The system SHALL collate variable-length multi-turn web episodes into padded batch tensors with 4D causal attention masks, aligned thought NLL masks, action type targets, coordinate regression masks, argument tokens, and JEPA transition pairs.

#### Scenario: Batch tensor collation
- **WHEN** a list of variable-length real trajectory samples is collated
- **THEN** the collator outputs a unified batch containing padded token sequences, loss masks for all dual-process decoding heads, and aligned next-state latent transition pairs

### Requirement: Automated Mind2Web Corpus Streaming Downloader
The system SHALL provide an automated corpus downloader script (`data/download_mind2web.py`) to stream, verify, and unpack official Mind2Web (`osunlp/Mind2Web`) task definitions and raw HTML/DOM annotations across 137 real-world domains.

#### Scenario: Streaming Mind2Web corpus subsets
- **WHEN** `data/download_mind2web.py` is executed with specified task splits (train, val, test) and maximum sample counts
- **THEN** the script downloads the raw dataset chunks, verifies content integrity, extracts task annotations and DOM snapshots to local disk, and logs progress without stalling

#### Scenario: Git hygiene and zero-leakage enforcement
- **WHEN** dataset download artifacts, raw HTML trees, or parquet files are written to the workspace
- **THEN** project git ignore rules prevent `data/raw_mind2web/`, `data/*.parquet`, and intermediate dataset caches from being tracked or committed to version control

### Requirement: Provenance-Aware Real-Only Trajectory Selection
The system SHALL distinguish captured real interaction traces from procedural or reconstructed synthetic examples and SHALL expose a real-only dataset mode for controlled experiments.

#### Scenario: Real-only dataset load
- **WHEN** a training or evaluation run requests real-only mode
- **THEN** every returned sample has trace provenance identifying its source and version, source record, website identity, upstream split, and content checksum, and no procedural or reconstructed synthetic sample is returned

#### Scenario: Missing authentic trace data
- **WHEN** real-only mode is requested but no usable authentic traces are available
- **THEN** dataset initialization fails with an actionable error and does not fall back to procedural or synthetic data

#### Scenario: Mixed source inventory
- **WHEN** authentic traces and synthetic examples are both present in the configured data roots
- **THEN** the real-only dataset includes only authentic traces and reports the excluded synthetic record count

### Requirement: Website-Held-Out Split Manifests
The system SHALL support deterministic train, validation, and test manifests grouped by normalized website identity so that a website cannot occur in more than one split.

#### Scenario: Website-disjoint manifest generation
- **WHEN** a site-grouped split is requested with a fixed seed
- **THEN** the system assigns whole websites to train, validation, or test, writes a reproducible manifest, and verifies that the three website sets are pairwise disjoint

#### Scenario: Mind2Web category is not used as website identity
- **WHEN** a Mind2Web record contains both a `website` field and broad `domain` or `subdomain` category fields
- **THEN** the split groups by the normalized `website` identity and retains category fields only for stratification and reporting

#### Scenario: Duplicate and snapshot leakage check
- **WHEN** manifests are generated from multiple source records
- **THEN** the system detects duplicate task records and shared snapshot identifiers across splits, excludes or reports leakage before training, and records the disposition in the manifest summary

#### Scenario: Insufficient site diversity
- **WHEN** a run requests minimum site counts that the available corpus cannot satisfy
- **THEN** preparation fails before training and reports the available and required counts for each split

### Requirement: Dataset Diversity Audit
The system SHALL produce a machine-readable and human-readable audit of the dataset selected for each experiment.

#### Scenario: Audit a prepared real-data corpus
- **WHEN** train, validation, and test manifests are prepared
- **THEN** the audit reports per-split trajectory counts, distinct website counts, source counts, task-category coverage, action-type coverage, exclusions, and manifest/content checksums

#### Scenario: Enforce the ablation diversity floor
- **WHEN** the backbone ablation requests its default real-data profile
- **THEN** preparation requires at least 50 training websites and at least 10 distinct validation and 10 distinct test websites, unless the experiment explicitly records different thresholds before training

### Requirement: Faithful Real-Trace Supervision
The system SHALL preserve recorded interaction labels and SHALL NOT fabricate rationale, target, or coordinate supervision for real-only samples.

#### Scenario: Parse recorded interaction labels
- **WHEN** a captured interaction trace is converted into a training sample
- **THEN** the original operation label and argument are retained, and any mapping into model action classes is explicit and counted in the dataset audit

#### Scenario: Preserve Mind2Web SELECT operations
- **WHEN** a captured Mind2Web action has operation `SELECT` and the ablation model uses the seven-action configuration
- **THEN** the sample target is the distinct SELECT class and is not converted to CLICK

#### Scenario: Missing rationale annotation
- **WHEN** a real trace contains no recorded rationale text
- **THEN** the sample has no rationale target and its rationale loss mask is false

#### Scenario: Missing or ambiguous target geometry
- **WHEN** a real trace has no unambiguous labeled target element or valid target bounding box
- **THEN** the sample marks the relevant target or coordinate label unavailable, masks its training loss and evaluation metric, and does not substitute a fabricated box or nearest element

#### Scenario: Missing observation snapshot
- **WHEN** a real trace action lacks a captured observation snapshot
- **THEN** the action is excluded or explicitly marked unusable with a recorded reason and is not filled with a procedural page template
