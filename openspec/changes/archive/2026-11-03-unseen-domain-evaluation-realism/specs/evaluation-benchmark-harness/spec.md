# Spec Delta: evaluation-benchmark-harness

## MODIFIED Requirements

### Requirement: Site-Held-Out Real Trajectory Evaluation
The system SHALL evaluate policy and ablation checkpoints on real trajectory test manifests whose websites are disjoint from the corresponding training and validation manifests, and SHALL benchmark generalization across official cross-task, cross-website, and cross-domain splits without mock fixture contamination.

#### Scenario: Evaluate a held-out real-site manifest
- **WHEN** a checkpoint is evaluated on a real-data test manifest
- **THEN** the harness verifies site disjointness and reports action type accuracy, target-element hit rate, coordinate MAE and tolerance accuracy, step accuracy, and exact full-trajectory match rate for the complete test set and by website, with macro-average-by-website and micro-average results

#### Scenario: Invalid or leaking test manifest
- **WHEN** a test manifest is missing, contains synthetic examples in a real-only evaluation, or shares a website or snapshot with training or validation
- **THEN** the harness stops the real-data evaluation and identifies the manifest validation failure

#### Scenario: Generalization split reporting across cross-task, cross-website, and cross-domain
- **WHEN** benchmark evaluation is executed across official Mind2Web generalization splits
- **THEN** the harness computes and outputs distinct metric rows for Cross-Task (unseen task, seen website), Cross-Website (unseen website, seen domain), and Cross-Domain (unseen domain), reporting the out-of-distribution generalization degradation

#### Scenario: Elimination of mock templates from benchmark evaluations
- **WHEN** evaluating under `--real_only` mode
- **THEN** all environments and dataset loaders reject procedural mock fallbacks and assert 100% of evaluated turns originate from authentic web DOM snapshots
