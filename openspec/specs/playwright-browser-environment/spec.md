# playwright-browser-environment Specification

## Purpose

Provides a live browser automation environment wrapping Playwright Chromium to navigate real websites, extract live DOM accessibility trees, capture real viewport screenshots, and dispatch native interaction primitives.

## Requirements

### Requirement: Browser Launch and Page Navigation
The browser environment SHALL initialize and manage an automated Chromium browser instance in headless or headed mode, navigate to requested URLs, and maintain persistent page sessions across interaction turns.

#### Scenario: Successful page navigation
- **WHEN** the browser environment is initialized or reset with a target URL
- **THEN** the Chromium browser launches, navigates to the URL, and waits for DOM network idle state

#### Scenario: Graceful handling of navigation failures
- **WHEN** the browser environment encounters an invalid URL, connection timeout, or DNS resolution failure
- **THEN** it intercepts the error without crashing, logs the error, and provides an informative error observation state

### Requirement: DOM Accessibility Tree and Element Extraction
The browser environment SHALL extract live DOM accessibility snapshots, interactable element candidates, normalized bounding boxes in $[0, 1] \times [0, 1]$, and serialized DOM text for model observation parsing.

#### Scenario: Accessibility snapshot and bounding boxes
- **WHEN** an observation is requested after page navigation or action execution
- **THEN** the environment extracts interactive elements (buttons, inputs, links), assigns integer identifiers, and computes normalized bounding box coordinates

#### Scenario: Minimal or dynamic DOM fallback
- **WHEN** a web page renders client-side dynamic content or sparse accessibility nodes
- **THEN** the environment returns all resolvable visible interactive nodes without raising exceptions

### Requirement: Viewport Screenshot Capture
The browser environment SHALL capture real viewport screenshots and format them as $[3, H, W]$ float tensors normalized to $[0, 1]$ or PIL images for multimodal visual action grounding.

#### Scenario: Viewport screenshot tensor generation
- **WHEN** the environment captures a visual observation of the current page
- **THEN** it produces an RGB image tensor of shape $[3, H, W]$ with values in $[0.0, 1.0]$ matching the configured resolution

### Requirement: Native Browser Primitive Dispatch
The browser environment SHALL dispatch discrete action primitives into the live browser session: CLICK at denormalized $(x, y)$ coordinates, TYPE into active or targeted elements, SCROLL via mouse wheel events, NAVIGATE to new URLs, and FINISH to complete the episode.

#### Scenario: Mouse click dispatch
- **WHEN** a CLICK action with normalized coordinates $(cx, cy) \in [0, 1] \times [0, 1]$ is received
- **THEN** the environment denormalizes coordinates to viewport pixels and triggers a native mouse click event

#### Scenario: Keyboard typing dispatch
- **WHEN** a TYPE action with string argument payload is dispatched
- **THEN** the environment types the text into the active input element via keyboard input events

#### Scenario: Viewport scrolling dispatch
- **WHEN** a SCROLL action with vertical/horizontal offset is dispatched
- **THEN** the environment dispatches mouse wheel delta events to scroll the viewport

#### Scenario: URL navigation dispatch
- **WHEN** a NAVIGATE action with target URL argument is received
- **THEN** the environment navigates the active page to the specified URL

#### Scenario: Episode termination dispatch
- **WHEN** a FINISH action is received
- **THEN** the environment flags the episode as done, computes final status, and releases turn resources

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

