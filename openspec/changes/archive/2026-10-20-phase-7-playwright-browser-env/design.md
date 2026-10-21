# Design

## Context

Following Phase 6D (Multimodal visual action grounding, patch encoder, and synthetic vision trajectories), JED-M2 possesses an integrated `VisionPatchEncoder` that transforms $224 \times 224$ (or $448 \times 448$) screen images into visual patch tokens, grounding the continuous `CoordinateHead` via cross-attention.

Previously in Phase 6A, `WebOSEnvironment` provided a simulated discrete state machine with predefined scenarios. In Phase 7, we bridge the model to real digital web environments by introducing `jed_m2/browser.py`, which wraps Playwright Chromium to navigate live websites, extract accessibility trees, capture real screenshots, and execute physical browser actions.

Strict hardware constraints remain in effect:
- Host GPU: NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7 VRAM, < 6.0 GB ceiling).
- Pure PyTorch model architecture without compiled C++/CUDA packages.
- Pure ASCII console output (Windows CP1252 safe).

## Goals / Non-Goals

**Goals:**
- Provide `PlaywrightBrowserEnvironment` in `jed_m2/browser.py` with synchronous Playwright Chromium automation.
- Extract live DOM accessibility trees, interactive element candidates, normalized bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0, 1]$, and visible text representations.
- Capture real viewport screenshots as normalized $[3, H, W]$ float tensors ready for `VisionPatchEncoder`.
- Dispatch native browser primitives: CLICK (denormalized to viewport pixels), TYPE (keyboard input), SCROLL (mouse wheel events), NAVIGATE (`page.goto`), and FINISH.
- Extend `JEDM2Agent` to synchronize live screenshot capture with visual patch encoding and recurrent $O(1)$ step rollouts.
- Update `interactive_agent.py` to support `--browser {simulated,playwright}` and starting `--url`.
- Implement standalone `verify_phase7.py` validating live browser execution, navigation, screenshot capture, action dispatch, error handling, and VRAM budget.

**Non-Goals:**
- Multi-tab management or parallel browser clusters (single active viewport per agent session).
- Heavy external scraping frameworks (Selenium, Puppeteer).
- Architectural alterations to core transformer or SSM neural network layers.

## Decisions

### 1. Synchronous Playwright API (`sync_playwright`)
- **Choice**: Use `playwright.sync_api.sync_playwright` rather than the asynchronous API.
- **Rationale**: PyTorch training, recurrent rollout loops, and interactive terminal sessions run synchronously. Avoiding Python `asyncio` event loops eliminates nested loop conflicts and simplifies debugging in the CLI and test verification scripts.
- **Alternative considered**: `asyncio` with `async_playwright`, which requires running an event loop inside PyTorch stepping methods and complicates interactive input.

### 2. Dual Accessibility Snapshot + DOM Bounding Box Extraction
- **Choice**: Combine Playwright's `page.accessibility.snapshot()` with an evaluated JavaScript DOM query that inspects visible interactable elements (`button`, `input`, `a`, `select`, `textarea`, `[role]`) to retrieve client bounding boxes via `getBoundingClientRect()`.
- **Rationale**: Accessibility snapshots provide structured semantic roles and accessible names, but exact pixel-level bounding boxes are essential for normalized coordinate regression and spatial pointer hit-testing. Normalizing coordinates against `(viewport_width, viewport_height)` guarantees consistency with JED-M2's $[0, 1]$ spatial coordinate regression head.
- **Alternative considered**: Relying solely on raw HTML string parsing (exceeds token limits and lacks layout geometry) or accessibility snapshot alone (lacks pixel bounding boxes).

### 3. Screenshot Tensor Pipeline for Visual Patch Grounding
- **Choice**: Capture viewport screenshot bytes via `page.screenshot(type="png")`, decode with PIL (`PIL.Image.open`), resize to `(image_size, image_size)` (e.g. 224x224), convert to float tensor `[3, H, W]` normalized to $[0.0, 1.0]$, and transfer to model execution device.
- **Rationale**: Matches the exact input format required by `VisionPatchEncoder` in `jed_m2/vision.py`, allowing seamless early fusion and cross-attention grounding in `CoordinateHead`.
- **Alternative considered**: Capturing base64 screenshots and saving to disk (introduces unnecessary I/O latency).

### 4. Agent Polymorphic Environment Stepping
- **Choice**: Support both `WebOSEnvironment` and `PlaywrightBrowserEnvironment` in `JEDM2Agent.step_environment(...)` by detecting environment capabilities. When a screenshot is available, the agent calls `model.patch_encoder(screenshot)` to obtain `visual_tokens` and passes them to `model.step(..., visual_tokens=visual_tokens)`.
- **Rationale**: Preserves 100% backward compatibility with simulated benchmark scenarios from Phase 6A/6B while empowering live browser interaction.

### 5. Resilient Error Handling for Live Web Navigation
- **Choice**: Wrap page navigation and interactions in robust try-except blocks that catch `playwright._impl._errors.Error`, `TimeoutError`, and network errors, returning an informative error observation state rather than raising fatal runtime exceptions.
- **Rationale**: Real-world websites may fail, timeout, or block requests. The Large Action Model must observe failure states and formulate recovery actions without terminating the process.

## Risks / Trade-offs

- **[Risk] Missing Playwright or Chromium binary on host**
  → *Mitigation*: Graceful import checks with actionable error messages indicating `pip install playwright && playwright install chromium`. Verification script checks environment prerequisite cleanly.
- **[Risk] Network latency and variable page load times**
  → *Mitigation*: Configure standard navigation timeout (e.g., 10 seconds) with `wait_until="domcontentloaded"` rather than waiting indefinitely for network idle.
- **[Risk] High-DPI screen coordinate discrepancies**
  → *Mitigation*: Explicitly launch browser context with fixed viewport `{"width": 1280, "height": 800}` and `device_scale_factor=1.0` to guarantee deterministic normalized coordinate mapping $[0, 1]$.
- **[Risk] VRAM memory leakage during live visual stepping**
  → *Mitigation*: Run visual patch encoding under `torch.no_grad()`, detach visual tokens immediately, and enforce strict VRAM ceiling checks (< 6.0 GB).
