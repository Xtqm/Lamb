# interactive-webos-environment Specification

## Purpose
Provides a deterministic simulated Web/OS sandbox environment that models multi-turn GUI and browser interactions, element trees, spatial hit-testing, action execution, and reward evaluation.

## Requirements

### Requirement: Interactive UI Element Tree State
The environment SHALL maintain an interactable tree of UI elements where each element has an element ID, HTML/OS tag, role, accessible text, and normalized bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0, 1]$.

#### Scenario: Element tree initialization and query
- **WHEN** the environment is initialized or reset with a benchmark scenario
- **THEN** the active state exposes a list of interactable UI elements with normalized non-inverted bounding boxes and center coordinates in $[0, 1]$

### Requirement: Action Primitive Transition Handling
The environment SHALL execute state transitions for discrete action primitives including CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, and FINISH, updating internal state and emitting transition observations.

#### Scenario: Click action transitions state
- **WHEN** the agent emits a CLICK action targeting an active interactive element
- **THEN** the environment updates element selection state and emits the post-click DOM observation

#### Scenario: Type action enters text argument
- **WHEN** the agent emits a TYPE action targeting an input field with string argument payload
- **THEN** the environment updates the target element's stored value and reflects the typed text in the subsequent observation

#### Scenario: Navigation and scroll transitions
- **WHEN** the agent emits a NAVIGATE action with target URL or a SCROLL action with direction
- **THEN** the environment updates the current URL location or viewport scroll position and updates visible elements

### Requirement: Spatial Hit-Testing
The environment SHALL perform spatial hit-testing on normalized $(x, y) \in [0, 1]$ screen coordinates to determine whether pointer actions intersect target element bounding boxes.

#### Scenario: Coordinate hit-test intersects element
- **WHEN** a CLICK or TYPE action provides coordinates $(x, y)$ falling inside an element's bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$
- **THEN** the hit-test succeeds and links the action to that specific element

#### Scenario: Coordinate hit-test misses target
- **WHEN** a CLICK action provides coordinates $(x, y)$ that do not intersect any interactive element
- **THEN** the hit-test registers a background click with zero state mutation on interactive elements

### Requirement: Deterministic Reward and Goal Evaluation
The environment SHALL compute deterministic step rewards, evaluate task completion against goal criteria, and signal episode termination upon receiving a FINISH action or exceeding maximum step horizon.

#### Scenario: Task successfully completed
- **WHEN** the agent executes a FINISH action after fulfilling all required task state transitions
- **THEN** the environment returns done=True, terminal success reward of 1.0, and task completion status True

#### Scenario: Episode truncation on step limit
- **WHEN** the interaction sequence reaches the maximum allowable steps without completing the goal
- **THEN** the environment terminates the episode with done=True and task completion status False
