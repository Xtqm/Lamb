# Spec Delta: evaluation-benchmark-harness

## MODIFIED Requirements

### Requirement: Standardized Action and Trajectory Metrics
The system SHALL compute formal action and trajectory evaluation metrics across agent execution runs, including Task Success Rate (SR %), Step Success Rate, Action Primitive F1, Spatial Coordinate Mean Absolute Error (MAE), Spatial Coordinate Tolerance Accuracy ($\epsilon \le 0.05$), Trajectory Efficiency Ratio, and Decision Latency.

#### Scenario: Metric computation on completed episode
- **WHEN** an evaluation episode finishes with predicted actions and ground truth steps
- **THEN** the system returns a metric record containing SR in $[0, 1]$, Step SR in $[0, 1]$, F1 in $[0, 1]$, MAE $\ge 0$, tolerance accuracy in $[0, 1]$, trajectory efficiency in $[0, 1]$, and decision latency in milliseconds per step.

#### Scenario: Metric numerical sanity guarantee
- **WHEN** metrics are computed across any number of evaluated episodes
- **THEN** all computed metric values are valid floating-point numbers containing zero NaNs and zero Infs.

#### Scenario: Native termination metric breakdown
- **WHEN** benchmark evaluation completes for a policy checkpoint
- **THEN** the metric summary explicitly reports Native FINISH Precision, Recall, and F1 score, separating autonomous unassisted success from referee-assisted completion

### Requirement: Online Closed-Loop Environment Evaluation
The system SHALL support online closed-loop multi-turn agent evaluation across interactive environments, executing live actions in `WebOSEnvironment` sandboxes and live Playwright browser sessions until task termination or maximum step limit.

#### Scenario: Closed-loop multi-turn simulated evaluation
- **WHEN** evaluating a model checkpoint on simulated Web/OS sandbox tasks
- **THEN** the agent interacts multi-turn with `WebOSEnvironment`, tracking step success and overall task success dynamically.

#### Scenario: Closed-loop live Playwright evaluation
- **WHEN** evaluating a model checkpoint on live Playwright browser tasks
- **THEN** the agent navigates real web pages, captures screenshots and DOM snapshots, dispatches browser primitives, and records task completion metrics.

#### Scenario: Unassisted closed-loop evaluation protocol
- **WHEN** evaluating under the standard unassisted benchmark protocol
- **THEN** execution proceeds without in-the-loop completion referee overrides, terminating exclusively on agent-emitted FINISH or maximum step timeout
