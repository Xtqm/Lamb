# Proposal

## Why

With Phases 1-5 completed and verified (backbone architecture, dual-system action heads, synthetic trajectory generation, composite multi-task loss harness, memory-calibrated production trainer, and RL policy optimization scaffolding), JED-M2 requires a closed-loop execution environment and an interactive agent runtime. Prior to Phase 6A, the model operated primarily in sequence-parallel training and offline loss evaluation modes. To validate live agency, enable interactive human-in-the-loop task execution, and benchmark constant-memory recurrent rollouts over long horizons, we need:
1. A deterministic simulated Web/OS sandbox environment modeling multi-turn web and desktop interactions.
2. An interactive agent execution engine wrapping `JEDM2Model` with true $O(1)$ constant-memory autoregressive recurrent rollout via `step(...)`, preserving Mamba SSM states, conv buffers, and MLA KV-cache across multiple turns.
3. An interactive CLI tool providing live terminal execution with step-by-step System 2 rationale inspection and ASCII UI visual canvas rendering.

## What Changes

- **Simulated Web/OS Environment (`jed_m2/environment.py`)**:
  - `WebOSEnvironment`: Stateful digital sandbox tracking interactive UI element trees (buttons, text inputs, dropdowns, links, navigation targets) with bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in [0, 1]$ and normalized center coordinates.
  - Supports realistic step transitions for discrete primitives: `CLICK`, `TYPE`, `SCROLL`, `NAVIGATE`, `API_CALL`, and `FINISH`.
  - Computes deterministic ground-truth task completion, step-level shaped rewards, terminal success rewards, and goal evaluation.
  - Supports both predefined benchmark scenarios (e.g., e-commerce, travel booking, OS administration) and dynamic goal-driven tasks.
- **Interactive Agent Runtime (`jed_m2/agent.py`)**:
  - `JEDM2Agent`: Execution engine wrapping `JEDM2Model` for closed-loop decision making.
  - Employs true $O(1)$ constant-memory autoregressive recurrent rollout using `model.step(...)`, retaining layer-wise Mamba SSM states, conv buffers, and MLA KV-cache across multi-turn interactions.
  - Decoupled System 1 / System 2 execution loop: Autoregressively generates latent thought tokens (System 2) until `ActionGate` exceeds the commitment threshold, then predicts discrete primitive, continuous $(x, y) \in [0, 1]$ screen coordinates, and argument payloads (System 1).
  - Handles observation formatting/parsing, action dispatching, turn history tracking, and rollout resetting.
- **Interactive Agent CLI Script (`interactive_agent.py`)**:
  - Command-line interface supporting manual task goal specification or automated execution of predefined benchmark scenarios.
  - Real-time terminal output displaying System 2 reasoning rationales, action primitives, predicted continuous $(x, y)$ coordinates, text arguments, environment transition feedback, and pure ASCII visual canvas representation of the UI state and pointer clicks.
- **Public API Exposure (`jed_m2/__init__.py`)**:
  - Exports environment classes (`WebOSEnvironment`, `UIElement`, `EnvironmentState`, `StepResult`) and agent runtime classes (`JEDM2Agent`, `AgentStepOutput`, `TurnRecord`).
- **Phase 6A Verification Suite (`verify_phase6a.py`)**:
  - Benchmarks $O(1)$ constant memory invariance over long-horizon rollouts (up to 64 consecutive recurrent steps).
  - Profiles per-step rollout latency (ms/step) on CUDA in `bfloat16`.
  - Evaluates end-to-end task execution and completion accuracy on simulated multi-turn tasks.
  - Validates zero NaNs/Infs and strict compliance with the 6.0 GB VRAM ceiling.

## Capabilities

### New Capabilities
- `interactive-webos-environment`: Deterministic simulated Web/OS sandbox environment modeling interactive UI element trees, action transitions (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH), spatial coordinate hit-testing, deterministic step/terminal rewards, and goal evaluation.
- `interactive-agent-runtime`: Interactive LAM execution engine and CLI tool managing closed-loop agent rollouts with $O(1)$ constant-memory recurrent state persistence (Mamba SSM + MLA KV-cache), dual-system reasoning and action decoding, and ASCII visual canvas rendering.

### Modified Capabilities
<!-- None. Existing capabilities (unified-lam-backbone, dual-system-action-heads, etc.) remain intact and are utilized directly. -->

## Impact

- **New Files**:
  - `jed_m2/environment.py`: Simulated Web/OS sandbox environment.
  - `jed_m2/agent.py`: Interactive agent runtime and stateful rollout manager.
  - `interactive_agent.py`: Interactive CLI entrypoint with ASCII UI rendering.
  - `verify_phase6a.py`: Comprehensive Phase 6A verification and benchmark script.
- **Modified Files**:
  - `jed_m2/__init__.py`: Public export of runtime and environment classes.
  - `PROJECT_CONTEXT.md`: Update Phase 6A status to verified.
- **Dependencies & Hardware**:
  - 100% pure PyTorch with zero external compiled C++/CUDA packages or Hugging Face dependencies.
  - Strict compliance with NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) and 6.0 GB active training/inference ceiling.
