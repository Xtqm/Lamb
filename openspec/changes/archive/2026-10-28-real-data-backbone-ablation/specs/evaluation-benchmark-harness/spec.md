# Spec Delta

## ADDED Requirements

### Requirement: Site-Held-Out Real Trajectory Evaluation
The system SHALL evaluate ablation checkpoints on real trajectory test manifests whose websites are disjoint from the corresponding training and validation manifests.

#### Scenario: Evaluate a held-out real-site manifest
- **WHEN** a checkpoint is evaluated on a real-data test manifest
- **THEN** the harness verifies site disjointness and reports action type accuracy, target-element hit rate, coordinate MAE and tolerance accuracy, step accuracy, and exact full-trajectory match rate for the complete test set and by website, with macro-average-by-website and micro-average results

#### Scenario: Invalid or leaking test manifest
- **WHEN** a test manifest is missing, contains synthetic examples in a real-only evaluation, or shares a website or snapshot with training or validation
- **THEN** the harness stops the real-data evaluation and identifies the manifest validation failure

### Requirement: Evaluation Provenance and Protocol Separation
The system SHALL identify the data source, split, website set, checkpoint, run identifier, and evaluation protocol for every reported result.

#### Scenario: Report offline trace replay
- **WHEN** predictions are scored by replaying recorded real trajectories
- **THEN** the result is labeled offline trace replay and is not presented as closed-loop task completion

#### Scenario: Compare different environment types
- **WHEN** results from real trace replay, synthetic tasks, or closed-loop browser environments appear in one report
- **THEN** each protocol is reported in a separate section with its own metrics and provenance, without combining the results into a single score
