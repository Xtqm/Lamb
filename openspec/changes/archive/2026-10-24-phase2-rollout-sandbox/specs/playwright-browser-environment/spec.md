# Spec Delta: playwright-browser-environment

## MODIFIED Requirements

### Requirement: Programmatic Oracle Task Verification and Reward Shaping
The browser environment SHALL evaluate deterministic programmatic oracle verifiers against live page state to determine task success, award intermediate milestone rewards, and compute dense spatial IoU rewards against target element bounding boxes.

#### Scenario: URL state matching oracle
- **WHEN** an action completes and the oracle defines target URL patterns
- **THEN** the environment tests the current page URL against the expected regex or path and awards milestone rewards upon match

#### Scenario: DOM element presence and absence verification
- **WHEN** a task requires verifying UI mutations (e.g., item added to cart, confirmation message displayed)
- **THEN** the environment queries DOM selector state and awards milestone or terminal reward upon condition fulfillment

#### Scenario: Spatial IoU reward shaping
- **WHEN** the agent dispatches a CLICK action at coordinates $(cx, cy)$ in the presence of a known target element bounding box
- **THEN** the oracle awards intermediate shaped reward based on whether the coordinates fall inside or within proximity of the target bounding box

#### Scenario: Task goal satisfaction and episode termination
- **WHEN** all required oracle conditions are satisfied or a valid FINISH action is dispatched
- **THEN** the environment marks the episode as done, flags goal satisfied as True, and computes positive terminal reward
