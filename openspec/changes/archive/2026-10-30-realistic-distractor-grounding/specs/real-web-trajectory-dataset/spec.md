# Spec Delta: real-web-trajectory-dataset

## MODIFIED Requirements

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
