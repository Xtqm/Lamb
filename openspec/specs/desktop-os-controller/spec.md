# desktop-os-controller Specification

## Purpose
Provides low-level Windows OS automation primitives including per-monitor DPI-aware coordinate mapping, mouse and keyboard dispatching with natural timing, and high-speed screen observation encoding for vision patch tokens.

## Requirements

### Requirement: Windows Per-Monitor DPI Awareness
The OS controller SHALL register per-monitor DPI awareness with the Windows window manager to ensure mouse pointer coordinates and screen captures align accurately across 100%, 125%, 150%, and multi-monitor scaling configurations.

#### Scenario: High DPI coordinate alignment
- **WHEN** the host operating system is configured with non-100% display scaling (such as 125% or 150%)
- **THEN** the OS controller sets process DPI awareness mode 2 (per-monitor DPI aware) and queries physical pixel resolutions without scaling distortion.

### Requirement: Continuous Normalized Coordinate Translation
The OS controller SHALL translate continuous normalized coordinate predictions $(x, y) \in [0, 1] \times [0, 1]$ into discrete physical pixel coordinates $(X_{screen}, Y_{screen})$ clamped within the target display or window bounds.

#### Scenario: Bound-safe coordinate projection
- **WHEN** a normalized coordinate tuple $(0.5, 0.5)$ is provided for a display of resolution $1920 \times 1080$
- **THEN** the controller resolves physical pixel target $(960, 540)$ within valid screen bounds.

### Requirement: Native Mouse Dispatching
The OS controller SHALL dispatch native mouse events including left click, right click, double click, drag-and-drop, scroll actions, and smooth human-like cursor trajectories.

#### Scenario: Smooth cursor movement and click execution
- **WHEN** a mouse click action at target coordinates is commanded
- **THEN** the controller moves the cursor along an interpolated trajectory to the target position and executes the specified mouse button down and up sequence.

### Requirement: Keystroke and Hotkey Dispatching
The OS controller SHALL emit keyboard typing sequences with randomized inter-keystroke timing jitter, special control keys (Enter, Tab, Backspace, Escape), and modifier hotkey combinations (Ctrl+C, Ctrl+V, Alt+Tab).

#### Scenario: Jittered text typing and modifier combination
- **WHEN** an action specifies typing text or triggering a shortcut hotkey
- **THEN** the controller emits keystrokes with natural inter-key delay variance and handles modifier key press and release transitions cleanly.

### Requirement: High-Speed Screen Perception to Vision Patch Tensors
The OS controller SHALL capture the primary screen or target window as RGB image data and convert it into normalized PyTorch observation tensors formatted for direct ingestion by the visual patch encoder.

#### Scenario: Screen observation tensor generation
- **WHEN** a visual observation frame is requested by the agent loop
- **THEN** the controller produces an RGB PyTorch tensor of shape `[1, 3, H, W]` in bfloat16 or float32 with minimal capture latency and zero memory leaks.
