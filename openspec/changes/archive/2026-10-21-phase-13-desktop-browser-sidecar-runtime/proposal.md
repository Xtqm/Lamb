# Proposal

## Why

Autonomous Large Action Models (LAMs) are predominantly confined to synthetic sandboxes or isolated headless browser sessions, preventing direct assistance on real-world multi-application desktop workflows that span native Windows software, local files, terminal environments, and browser sessions. To realize true pair-programming and desktop-assistant capabilities on personal workstations, JED-M2 requires a persistent, ultra-lightweight background sidecar daemon running on Windows with full per-monitor DPI awareness, sub-100 MB idle memory footprint, real-time telemetry streaming, and human-in-the-loop safety interception, optimized for execution on local hardware such as the NVIDIA GeForce RTX 5060 Laptop GPU (< 6.0 GB VRAM).

## What Changes

- Add pure Python OS automation controller (`jed_m2/sidecar/os_controller.py`):
  - Windows per-monitor DPI awareness via `ctypes.windll.shcore.SetProcessDpiAwareness(2)` ensuring pixel-accurate mouse and keyboard control across 100%, 125%, 150%, and custom display scaling settings.
  - Coordinate translation mapping continuous normalized model predictions $(x, y) \in [0, 1]$ to physical screen pixels $(X_{screen}, Y_{screen})$ across single or multi-monitor configurations.
  - Mouse dispatch using `pyautogui` (with pure ctypes fallback) supporting left click, right click, double click, smooth human-like Bezier/interpolated cursor trajectories, drag-and-drop, and scroll.
  - Keyboard dispatch supporting text typing with randomized inter-key jitter, special keys (Enter, Tab, Backspace, Escape), and key combination hotkeys (Ctrl+C, Ctrl+V, Alt+Tab).
  - High-speed screen perception capturing full displays or active window regions as RGB tensors directly compatible with `VisionPatchEncoder`.
- Add lightweight background sidecar daemon (`jed_m2/sidecar/daemon.py`):
  - Asynchronous local daemon using FastAPI and asyncio with REST endpoints `POST /task`, `GET /status`, `POST /abort`, and real-time WebSocket telemetry streaming at `GET /stream`.
  - Minimal idle footprint maintaining < 100 MB RAM and 0% idle GPU utilization when no tasks are running.
  - Multi-turn stateful session manager persisting recurrent Mamba SSM states, rolling 1D causal conv buffers, and MLA KV-cache across interaction steps without sequence length explosion.
  - Dynamic model sizing supporting seamless loading of either JED-M2-Nano (51M) or JED-M2-Medium (273M) checkpoints via `--model_size {nano,medium}`.
- Add human-in-the-loop safety supervisor (`jed_m2/sidecar/safety.py`):
  - Rule-based interception for destructive file system actions (e.g., `rm`, `del`, `format`, `rmdir`), sensitive authentication/password input fields, and financial payment buttons/transactions.
  - Automatic policy execution pausing that prompts the user for explicit confirmation via Windows toast notifications or terminal CLI prompts before proceeding.
- Add interactive CLI runner and HUD (`sidecar_agent.py`):
  - Unified commands: `daemon` (start background service), `exec` (one-shot task execution), and `monitor` (attach to running daemon).
  - Real-time ASCII terminal HUD displaying System 2 latent rationales, committed action primitives, and target coordinate overlays.
- Add comprehensive verification test suite (`verify_phase13.py`):
  - 5 automated verification tests covering DPI coordinate translation accuracy, daemon lifecycle/API health, safety supervisor interception, perception-to-tensor memory leak checks, and RTX 5060 VRAM limits (< 6.0 GB).

## Capabilities

### New Capabilities

- `desktop-os-controller`: Windows per-monitor DPI-aware OS automation controller handling normalized-to-physical coordinate translation, mouse movement and click dispatch, keyboard typing with jitter, and screen observation capture directly into PyTorch visual patch tensors.
- `desktop-sidecar-runtime`: Asynchronous background sidecar daemon, stateful multi-turn Mamba/MLA session management, human-in-the-loop safety interception, interactive CLI runner with ASCII HUD, and local GPU resource management.

### Modified Capabilities

<!-- Existing capabilities whose REQUIREMENTS are changing. None in this change. -->

## Impact

- **New Files**:
  - `jed_m2/sidecar/__init__.py`
  - `jed_m2/sidecar/os_controller.py`
  - `jed_m2/sidecar/daemon.py`
  - `jed_m2/sidecar/safety.py`
  - `sidecar_agent.py`
  - `verify_phase13.py`
- **Dependencies**: `fastapi`, `uvicorn`, `websockets`, `pyautogui` (optional / fallback supported), `pillow`, `torch` (>= 2.0).
- **APIs**: Exposes local REST (`/task`, `/status`, `/abort`) and WebSocket (`/stream`) endpoints on port 8765.
- **Hardware Profile**: Verified for NVIDIA GeForce RTX 5060 Laptop GPU under 6.0 GB VRAM in bfloat16 inference.
