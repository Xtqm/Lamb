# Tasks

## 1. Pure Python OS Automation Controller

- [x] 1.1 Implement Windows DPI awareness and coordinate translation in `jed_m2/sidecar/os_controller.py` with `ctypes.windll.shcore.SetProcessDpiAwareness(2)` and bound clamping, and verify mapping precision across 100%, 125%, and 150% scales.
- [x] 1.2 Implement native mouse dispatching (left/right/double click, smooth human-like Bezier cursor trajectories, drag-and-drop, scroll) with pyautogui and ctypes fallback, verifying dispatch execution and cursor movement.
- [x] 1.3 Implement keyboard keystroke typing with natural inter-key jitter, special keys (Enter, Tab, Backspace, Escape), and modifier hotkeys (Ctrl+C, Ctrl+V, Alt+Tab), and verify key sequence generation.
- [x] 1.4 Implement high-speed screen perception capturing active windows or full displays as RGB PyTorch tensors for `VisionPatchEncoder`, and verify tensor dimensions `[1, 3, H, W]` and memory leak freedom.

## 2. Human-in-the-Loop Safety Supervisor

- [x] 2.1 Implement `SafetySupervisor` in `jed_m2/sidecar/safety.py` with pattern matching for destructive file operations (`rm`, `del`, `format`, `rmdir`) and sensitive target keywords (password, credentials, payment checkout), and verify rule interception on synthetic test inputs.
- [x] 2.2 Implement user confirmation prompts via Windows toast notifications (with CLI fallback) and atomic pause/resume control, and verify confirmation blocking and resolution.

## 3. Asynchronous Sidecar Daemon and Session Manager

- [x] 3.1 Implement stateful session manager in `jed_m2/sidecar/daemon.py` preserving recurrent Mamba SSM states, rolling 1D conv buffers, and MLA KV-cache across interaction steps, and verify $O(1)$ memory consumption across multi-turn rollouts.
- [x] 3.2 Implement FastAPI daemon with endpoints `POST /task`, `GET /status`, `POST /abort`, and WebSocket telemetry streaming at `GET /stream`, and verify async API responses and event streaming under idle and active loads.
- [x] 3.3 Add dynamic model sizing supporting `--model_size {nano,medium}` for JED-M2-Nano (51M) and JED-M2-Medium (273M) checkpoints, and verify parameter counts and GPU memory residency under 6.0 GB VRAM.

## 4. Interactive CLI Runner and Terminal HUD

- [x] 4.1 Implement `sidecar_agent.py` CLI supporting subcommands `daemon`, `exec`, and `monitor`, with argument parsing for port, device, model size, and model path, and verify CLI help and command routing.
- [x] 4.2 Build real-time ASCII terminal HUD rendering step-by-step System 2 thoughts, committed action primitives, gate confidence, and 2D target coordinate crosshairs, and verify terminal visualization against mock and live streams.

## 5. Verification Suite and Integration Testing

- [x] 5.1 Implement comprehensive test suite in `verify_phase13.py` implementing Test 1 (coordinate precision), Test 2 (daemon lifecycle and abort), Test 3 (safety supervisor interception), Test 4 (screen capture to tensor memory stability), and Test 5 (RTX 5060 VRAM ceiling check < 6.0 GB).
- [x] 5.2 Execute `python verify_phase13.py` and verify all 5 test suites pass with zero errors, zero memory leaks, and full compliance with the 6.0 GB VRAM ceiling.
