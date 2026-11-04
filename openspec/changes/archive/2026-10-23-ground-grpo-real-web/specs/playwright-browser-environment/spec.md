# Spec Delta: playwright-browser-environment

## ADDED Requirements

### Requirement: Local Benchmark and Fixture Serving
The browser environment SHALL support loading local static HTML fixtures and connecting to locally served reproducible web benchmark applications across realistic domains.

#### Scenario: Local HTML fixture navigation
- **WHEN** the browser environment is reset with a local file URL or static HTML fixture path
- **THEN** Chromium loads the document from disk and awaits DOM content loaded status before returning initial state

#### Scenario: Locally hosted web application navigation
- **WHEN** the browser environment is reset with a locally served benchmark endpoint (e.g. Saleor e-commerce or Postmill forum)
- **THEN** Chromium navigates to the HTTP server address and establishes an interactive browsing session

### Requirement: Programmatic Oracle Task Verification and Reward Shaping
The browser environment SHALL evaluate deterministic programmatic oracle verifiers against live page state to determine task success and compute step-level shaped rewards.

#### Scenario: URL state matching oracle
- **WHEN** an action completes and the oracle defines target URL patterns
- **THEN** the environment tests the current page URL against the expected regex or path and awards milestone rewards upon match

#### Scenario: DOM element presence and absence verification
- **WHEN** a task requires verifying UI mutations (e.g., item added to cart, confirmation message displayed)
- **THEN** the environment queries DOM selector state and awards milestone or terminal reward upon condition fulfillment

#### Scenario: Task goal satisfaction and episode termination
- **WHEN** all required oracle conditions are satisfied or a valid FINISH action is dispatched
- **THEN** the environment marks the episode as done, flags goal satisfied as True, and computes positive terminal reward

### Requirement: Interactive Environment Interface Alignment
The browser environment SHALL implement the unified environment interface conforming to `reset()` and `step(action_type, coordinates, argument)` contracts for drop-in rollout compatibility.

#### Scenario: Reset contract compliance
- **WHEN** `reset()` is invoked with an optional starting URL and goal description
- **THEN** the environment re-initializes page state and returns an observation record containing observation text, JEPA feature representations, and viewport screenshots

#### Scenario: Step contract compliance
- **WHEN** `step(action_type, coordinates, argument)` is invoked with an action primitive
- **THEN** the environment dispatches the action into the browser session, executes oracle checks, and returns an observation record with step reward, termination flags, and element metadata
