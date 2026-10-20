# Tasks

## 1. Web/OS Environment Simulation (`jed_m2/environment.py`)

- [x] 1.1 Implement `UIElement`, `EnvironmentState`, and `StepResult` data structures with normalized bounding boxes, element types, and interaction attributes, and verify data structure instantiation and serialization.
- [x] 1.2 Implement `WebOSEnvironment` state machine supporting action primitive transitions (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH), spatial hit-testing in $[0, 1] \times [0, 1]$, and observation generation, and verify transition state mutations.
- [x] 1.3 Implement benchmark scenario definitions (e-commerce product purchase, travel hotel booking, OS cloud admin) and deterministic reward/goal evaluation, and verify step and terminal reward calculations.

## 2. Interactive Agent Runtime (`jed_m2/agent.py`)

- [x] 2.1 Implement `AgentStepOutput`, `TurnRecord`, and `JEDM2Agent` runtime skeleton with observation encoding and $O(1)$ constant-memory recurrent state persistence via `model.step(...)`, and verify layer-wise state caching across multiple turns.
- [x] 2.2 Implement decoupled dual-system execution loop (System 2 latent thought token generation until `ActionGate` fires, followed by System 1 action primitive, coordinate, and argument decoding), and verify dual-system decision dispatch.
- [x] 2.3 Implement episode reset functionality, rollout turn history tracking, and safety guardrails (max thought tokens limit), and verify state isolation across consecutive episodes.

## 3. Interactive CLI Interface (`interactive_agent.py`)

- [x] 3.1 Implement pure ASCII 2D visual canvas renderer mapping UI bounding boxes and continuous pointer coordinates $(x, y) \in [0, 1]$ into a text grid, and verify clean ASCII rendering without Windows CP1252 encoding errors.
- [x] 3.2 Implement interactive CLI entrypoint supporting automated scenario execution and step-by-step interactive inspection, and verify CLI execution on simulated tasks.

## 4. Module Exposure & Phase 6A Verification Suite

- [x] 4.1 Expose environment and agent runtime classes in `jed_m2/__init__.py`, and verify all symbols import cleanly.
- [x] 4.2 Implement `verify_phase6a.py` test suite covering environment transitions, spatial hit testing, agent recurrent rollout, and dual-system decoding, and verify all unit checks pass.
- [x] 4.3 Benchmark $O(1)$ constant-memory invariance over 64 consecutive recurrent steps on CUDA in `bfloat16`, per-step latency profiling (ms/step), and strict compliance with the 6.0 GB VRAM ceiling, and verify all benchmarks succeed with exit code 0.
- [x] 4.4 Update `PROJECT_CONTEXT.md` marking Phase 6A completed and recording verification results and VRAM benchmarks.
