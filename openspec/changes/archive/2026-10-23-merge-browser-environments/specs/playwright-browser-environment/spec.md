# Spec Delta: playwright-browser-environment

## MODIFIED Requirements

### Requirement: Interactive Environment Interface Alignment
The browser environment SHALL implement the unified environment interface conforming to `reset()` and `step(action_type, coordinates, argument)` contracts for drop-in rollout compatibility, provided by canonical `PlaywrightEnvironment` with backward-compatible aliases `PlaywrightBrowserEnvironment` and `PlaywrightWebEnvironment`. The environment SHALL support both live browser execution and offline/sandboxed mock mode without hanging.

#### Scenario: Reset contract compliance
- **WHEN** `reset()` is invoked with an optional starting URL and goal description
- **THEN** the environment re-initializes page state and returns an observation record containing observation text, JEPA feature representations, and viewport screenshots

#### Scenario: Step contract compliance
- **WHEN** `step(action_type, coordinates, argument)` is invoked with an action primitive
- **THEN** the environment dispatches the action into the browser session, executes oracle checks, and returns an observation record with step reward, termination flags, and element metadata

#### Scenario: Canonical class naming and legacy alias interoperability
- **WHEN** code imports or instantiates `PlaywrightEnvironment`, `PlaywrightBrowserEnvironment`, or `PlaywrightWebEnvironment`
- **THEN** all three symbols resolve to the same underlying canonical class, providing identical methods, viewport configurations, oracle verifiers, and screenshot generation capabilities

#### Scenario: Sandboxed mock mode execution
- **WHEN** the browser environment is initialized with `mock_mode=True` or when browser process spawning fails in a sandboxed runtime
- **THEN** the environment gracefully falls back to synthetic DOM tree generation and programmatic oracle evaluation without blocking process execution or raising unhandled IPC exceptions
