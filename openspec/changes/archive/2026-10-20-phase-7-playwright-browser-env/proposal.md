# Proposal

## Why

Following Phase 6D (Multimodal visual action grounding, patch encoder, and synthetic vision trajectories), JED-M2 is capable of grounding spatial continuous actions in visual screenshot patch embeddings. However, to operate as a functional Large Action Model on real-world web applications, the model requires a live browser execution environment rather than relying solely on simulated sandbox state machines. 

Phase 7 introduces native live browser automation via Playwright Chromium, bridging JED-M2's recurrent multimodal backbone to real web pages, extracting live accessibility trees and real viewport screenshots, and executing physical browser actions (CLICK, TYPE, SCROLL, NAVIGATE, FINISH) with $O(1)$ constant-memory recurrence.

## What Changes

- **New Live Browser Environment (`jed_m2/browser.py`)**:
  - `PlaywrightBrowserEnvironment`: Wraps headless/headed Chromium browser via Playwright.
  - DOM Accessibility Tree & Interactive Element Extraction: Extracts live accessibility trees, interactive element bounding boxes normalized to $[0, 1]$, and viewport text representation.
  - Live Viewport Screenshot Capture: Captures real viewport screenshots as $[3, H, W]$ normalized RGB tensors or PIL images suitable for direct consumption by `VisionPatchEncoder`.
  - Native Interaction Primitives: Dispatches CLICK at continuous $(x, y) \in [0, 1]$ screen coordinates via `page.mouse.click`, TYPE at focused/target elements via `page.keyboard.type`, SCROLL via `page.mouse.wheel`, NAVIGATE via `page.goto`, and FINISH to terminate episodes.
  - Resilient Error Handling: Graceful handling of network timeouts, unreachable domains, and navigation failures without crashing the agent runtime.
- **Multimodal Agent Runtime Extension (`jed_m2/agent.py`)**:
  - Extends `JEDM2Agent` to synchronize live screenshots with `VisionPatchEncoder`, feeding visual patch tokens into `model.step(...)` to ground `CoordinateHead` predictions in real pixel observations.
  - Parses live accessibility trees into structured DOM token sequences.
  - Supports stepping both `WebOSEnvironment` (simulated) and `PlaywrightBrowserEnvironment` (live).
- **Interactive CLI Browser Selection (`interactive_agent.py`)**:
  - Adds `--browser {simulated,playwright}` flag.
  - Adds `--url` option to launch live browsing sessions on custom URLs or local HTML pages.
  - Visual canvas and action inspection adapted for live browser states.
- **Phase 7 Verification Suite (`verify_phase7.py`)**:
  - Standalone script verifying browser launch, local page navigation, screenshot tensor capture, DOM tree extraction, end-to-end agent action dispatch (CLICK, TYPE, SCROLL), memory stability, and graceful page failure handling under the 6.0 GB VRAM envelope.

## Capabilities

### New Capabilities
- `playwright-browser-environment`: Live browser automation environment wrapping Playwright Chromium to navigate real websites, extract accessibility trees and bounding boxes, capture real viewport screenshots, and dispatch native interaction primitives.

### Modified Capabilities
- `interactive-agent-runtime`: Adds support for live browser observation formatting, multimodal visual grounding synchronization with `VisionPatchEncoder`, and polymorphic action dispatching to live browser environments.

## Impact

- **Affected Code**:
  - `jed_m2/browser.py` (new module)
  - `jed_m2/agent.py` (extended for live browser stepping and visual grounding)
  - `jed_m2/__init__.py` (exposing `PlaywrightBrowserEnvironment`)
  - `interactive_agent.py` (adding `--browser` and `--url` CLI options)
  - `verify_phase7.py` (new standalone test script)
- **Dependencies**: Requires `playwright` Python package and Chromium browser binary (`playwright install chromium`). Gracefully reports guidance if Playwright is uninstalled.
- **Hardware & Build Rules**: Strict compliance with < 6.0 GB VRAM limit, pure PyTorch model architecture without compiled C++/CUDA packages, and pure ASCII console output for Windows CP1252 compatibility.
