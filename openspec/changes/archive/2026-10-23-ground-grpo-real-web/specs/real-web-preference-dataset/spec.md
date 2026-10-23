# Spec Delta: real-web-preference-dataset

## Purpose

Provides dataset abstraction and collation utilities to construct paired preference trajectories directly from authentic Mind2Web offline traces for Direct Policy Optimization (DPO).

## ADDED Requirements

### Requirement: Authentic Expert Demonstration Trajectory Extraction
The system SHALL extract winning trajectories $(\tau_w)$ from authentic human expert demonstrations in Mind2Web task definitions or disk-cached real-world web interaction traces.

#### Scenario: Successful expert demonstration loading
- **WHEN** raw Mind2Web trajectory traces are loaded into the preference dataset
- **THEN** the system extracts the expert action sequence, ground-truth target coordinates, and interactive element identifiers as the winning trajectory $(\tau_w)$

#### Scenario: Offline fallback when corpus is absent
- **WHEN** the raw Mind2Web directory is absent or inaccessible
- **THEN** the dataset logs an informational notice and constructs authentic demonstration trajectories from procedural web templates without failing initialization

### Requirement: Synthetic Negative and Perturbed Trajectory Synthesis
The system SHALL synthesize paired losing trajectories $(\tau_l)$ from winning demonstrations by injecting spatial coordinate perturbations or substituting sub-optimal action primitives while preserving initial DOM state.

#### Scenario: Spatial coordinate perturbation injection
- **WHEN** a losing trajectory is synthesized via coordinate perturbation
- **THEN** the system offsets ground-truth coordinates by displacement $(|\Delta x| > 0.15, |\Delta y| > 0.15)$, clamps them to $[0, 1]^2$, and flags the trajectory as negative $(\tau_l)$

#### Scenario: Sub-optimal action primitive substitution
- **WHEN** a losing trajectory is synthesized via action primitive degradation
- **THEN** the system substitutes an essential action with a spurious scroll, no-op, or non-interactable coordinate click

### Requirement: Paired Preference Sequence Collation for DPO
The system SHALL collate paired winning and losing trajectories into padded token tensors, attention masks, and action loss masks formatted for dual-policy DPO log-probability evaluation.

#### Scenario: Paired tensor batch collation
- **WHEN** a batch of paired preference samples is collated for training
- **THEN** the collator outputs aligned chosen and rejected token sequences, attention masks, and token loss masks with maximum sequence length truncation

#### Scenario: Coordinate and action loss mask alignment
- **WHEN** preference pairs contain continuous spatial coordinates and discrete action types
- **THEN** the collator extracts chosen and rejected coordinate targets and action type IDs for auxiliary margin verification

### Requirement: Persistent Disk Caching and Split Partitioning
The system SHALL persist pre-processed preference pairs to disk cache files and provide stratified train and validation split partitions quarantined from version control.

#### Scenario: Disk cache hit and replay
- **WHEN** the preference dataset initializes with an existing valid cache file
- **THEN** it loads pre-paired trajectories directly into memory without re-parsing raw DOM snapshots or regenerating negative perturbations

#### Scenario: Strict git quarantine of cached tensors
- **WHEN** cached preference dataset files are generated on disk
- **THEN** workspace git ignore rules ensure cache files and intermediate tensors are never tracked or committed to version control
