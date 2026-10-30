# Spec Delta: evaluation-benchmark-harness

## MODIFIED Requirements

### Requirement: Dual-Mode Coordinate Benchmark Reporting
The offline evaluation harness SHALL support dual-mode coordinate evaluation, reporting both unconstrained continuous coordinate metrics and DOM candidate-snapped metrics across target element hit rate, coordinate tolerance accuracy, coordinate MAE, step accuracy, and exact trajectory match rate, ensuring candidate snapping is evaluated against multi-element candidate pools that include real page distractors.

#### Scenario: Reporting raw and candidate-snapped metrics in parallel
- **WHEN** the offline evaluation harness replays context windows from real-world trajectory datasets
- **THEN** it computes and logs both unconstrained continuous coordinate metrics (treated as primary unassisted baseline) and DOM candidate-snapped metrics side-by-side in evaluation summaries

#### Scenario: Evaluation against realistic multi-element distractor pools
- **WHEN** candidate bounding box snapping is evaluated on real web snapshots
- **THEN** the candidate pool contains all parsed interactive DOM elements (including negative distractors), measuring true element discrimination rather than single-candidate proximity gating

#### Scenario: Preserving backwards-compatible offline scoring
- **WHEN** offline evaluation is executed with snapping disabled or candidate elements omitted
- **THEN** scoring defaults to standard continuous coordinate tolerance hit-testing without regression
