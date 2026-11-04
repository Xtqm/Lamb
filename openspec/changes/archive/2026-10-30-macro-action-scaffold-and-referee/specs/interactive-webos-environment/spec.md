# Spec Delta: interactive-webos-environment

## MODIFIED Requirements

### Requirement: Action Primitive Transition Handling
The environment SHALL execute state transitions for discrete action primitives including CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, and FINISH, executing deterministic macro-action behaviors such as input clearing and render settling, updating internal state and emitting transition observations.

#### Scenario: Click action transitions state
- **WHEN** the agent emits a CLICK action targeting an active interactive element
- **THEN** the environment updates element selection state and emits the post-click DOM observation

#### Scenario: Type action enters text argument
- **WHEN** the agent emits a TYPE action targeting an input field with string argument payload
- **THEN** the macro-action scaffold clears any existing value in the target input field before setting the new argument, updating the stored value and reflecting the typed text in the subsequent observation

#### Scenario: Navigation and scroll transitions
- **WHEN** the agent emits a NAVIGATE action with target URL or a SCROLL action with direction
- **THEN** the environment updates the current URL location or viewport scroll position and updates visible elements

### Requirement: Deterministic Reward and Goal Evaluation
The environment SHALL compute deterministic step rewards, evaluate task completion against goal criteria, support external referee inspection of goal criteria satisfaction, and signal episode termination upon receiving a FINISH action or exceeding maximum step horizon.

#### Scenario: Task successfully completed
- **WHEN** the agent executes a FINISH action after fulfilling all required task state transitions
- **THEN** the environment returns done=True, terminal success reward of 1.0, and task completion status True

#### Scenario: Referee verification of goal state
- **WHEN** the completion referee inspects the environment state after intermediate transitions
- **THEN** the environment exposes whether all terminal goal criteria are fulfilled prior to explicit FINISH emission

#### Scenario: Episode truncation on step limit
- **WHEN** the interaction sequence reaches the maximum allowable steps without completing the goal
- **THEN** the environment terminates the episode with done=True and task completion status False
