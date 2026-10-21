# Tasks

## 1. Environment Dependencies & Setup

- [x] 1.1 Verify and install `playwright` dependency and Chromium browser binary (`playwright install chromium`), verifying clean import via `python -c "import playwright"`.

## 2. Playwright Live Browser Automation Module

- [x] 2.1 Implement `PlaywrightBrowserEnvironment` in `jed_m2/browser.py` managing synchronous Chromium browser instances (headless/headed), navigation (`page.goto`), viewport configuration (1280x800), and graceful session shutdown.
- [x] 2.2 Implement accessibility tree and interactive element extraction in `jed_m2/browser.py` combining `page.accessibility.snapshot()` and DOM client bounding box queries normalized to $[0, 1] \times [0, 1]$.
- [x] 2.3 Implement viewport screenshot capture in `jed_m2/browser.py` returning normalized $[3, H, W]$ float tensors and PIL images compatible with `VisionPatchEncoder`.
- [x] 2.4 Implement native browser action dispatch in `jed_m2/browser.py` for CLICK (denormalized pixel coordinates via `page.mouse.click`), TYPE (`page.keyboard.type`), SCROLL (`page.mouse.wheel`), NAVIGATE (`page.goto`), and FINISH.
- [x] 2.5 Implement resilient error handling in `jed_m2/browser.py` to intercept network timeouts, 404/DNS failures, and invalid URLs, returning structured error feedback without raising uncaught exceptions.
- [x] 2.6 Export `PlaywrightBrowserEnvironment` in `jed_m2/__init__.py`.

## 3. Multimodal Agent Runtime Extension

- [x] 3.1 Extend `JEDM2Agent` in `jed_m2/agent.py` to parse live browser observations and synchronize screenshot tensors with `VisionPatchEncoder`, feeding visual patch tokens into `model.step(..., visual_tokens=visual_tokens)`.
- [x] 3.2 Update `JEDM2Agent.step_environment(...)` in `jed_m2/agent.py` to support `PlaywrightBrowserEnvironment` alongside `WebOSEnvironment` with $O(1)$ constant-memory recurrent state persistence.

## 4. Interactive CLI Updates

- [x] 4.1 Update `interactive_agent.py` to add `--browser {simulated,playwright}` flag and `--url` option.
- [x] 4.2 Adapt interactive visual canvas and action telemetry in `interactive_agent.py` for live browser session inspection.

## 5. Phase 7 Standalone Verification Suite

- [x] 5.1 Implement `verify_phase7.py` orchestrating a local HTTP test server with interactive elements (buttons, inputs, links).
- [x] 5.2 Validate browser launching, navigation, real screenshot capture ($[3, H, W]$ tensor in $[0, 1]$), and DOM accessibility extraction in `verify_phase7.py`.
- [x] 5.3 Validate end-to-end multimodal agent action dispatch (CLICK, TYPE, SCROLL, NAVIGATE, FINISH) in a live Playwright session in `verify_phase7.py`.
- [x] 5.4 Validate graceful error handling in `verify_phase7.py` when attempting to load unresolvable or invalid URLs.
- [x] 5.5 Validate peak GPU VRAM allocation remains strictly below 6.0 GB (6144.0 MB) and assert zero NaNs/Infs during live visual rollout in `verify_phase7.py`.
- [x] 5.6 Run `python verify_phase7.py` and verify all tests pass with exit code 0.
