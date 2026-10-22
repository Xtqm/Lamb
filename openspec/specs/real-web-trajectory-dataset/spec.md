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
- **THEN** the dataset parses the actual candidate DOM elements, extracts target element bounding boxes, normalizes pointer coordinates to $[0, 1]^2$, and maps action operations to discrete action types

#### Scenario: Offline fallback when raw corpus is absent
- **WHEN** the raw Mind2Web directory is not downloaded or accessible
- **THEN** `RealWebTrajectoryDataset` logs an informational notice and falls back to deterministic procedural HTML trace templates without failing initialization

### Requirement: Intelligent DOM Tree Pruning and Interactive Candidate Filtering
The system SHALL parse raw HTML trees containing 10,000+ nodes and prune non-interactive noise, retaining only candidate interactive elements including buttons, inputs, links, selects, and textareas.

#### Scenario: Pruning large DOM tree
- **WHEN** a raw web page HTML string containing over 10,000 DOM nodes is processed
- **THEN** the parser prunes passive containers and outputs a filtered list of interactive candidate elements without memory exhaustion or stack overflow

### Requirement: Normalized Spatial Bounding Box and Coordinate Extraction
The system SHALL extract element bounding box coordinates $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}]$ normalized in $[0, 1]^2$ and compute normalized center pointer coordinates $(x, y)$ for each interactive element.

#### Scenario: Bounding box normalization
- **WHEN** raw pixel or viewport bounding boxes are provided
- **THEN** the system normalizes coordinates to $[0, 1]^2$ and computes center points matching target click locations

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
