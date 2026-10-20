# Spec Delta

## Purpose

Provides an interactive agent runtime engine wrapping JEDM2Model for live closed-loop decision making, constant-memory recurrent rollouts, dual-system rationale and action decoding, and ASCII visual terminal inspection.

## ADDED Requirements

### Requirement: Constant-Memory Recurrent Agent Step
The agent runtime SHALL execute multi-turn interactive rollouts using $O(1)$ constant-memory autoregressive recurrent updates via `model.step(...)`, persisting layer-wise Mamba SSM states, conv buffers, and MLA KV-cache across turns without reprocessing past token history.

#### Scenario: Recurrent rollout memory invariance
- **WHEN** the agent executes an extended sequence of up to 64 consecutive interactive steps
- **THEN** GPU VRAM allocation remains bounded and constant within a 5% tolerance across steps 16 through 64

#### Scenario: Agent state reset
- **WHEN** the agent's reset method is invoked between episodes
- **THEN** recurrent state buffers and turn history are reinitialized to clean initial states

### Requirement: Decoupled Dual-System Decision Loop
The agent runtime SHALL execute a decoupled dual-system decision loop that generates latent thought tokens (System 2) until the ActionGate logit exceeds the commitment threshold, then emits the discrete action primitive, continuous $(x, y) \in [0, 1]$ screen coordinates, and argument payloads (System 1).

#### Scenario: System 2 reasoning commits to System 1 action
- **WHEN** the agent processes a new environment turn observation
- **THEN** it generates reasoning thought tokens until the action gate fires or maximum thought length is reached, returning both the rationale and structured action

#### Scenario: Thought generation guardrail
- **WHEN** the action gate does not exceed threshold within a configurable maximum reasoning budget
- **THEN** the runtime forcefully triggers action commitment to prevent execution deadlock

### Requirement: Observation Parsing and Action Dispatching
The agent runtime SHALL parse environment DOM text and observation vectors into token IDs and tensor embeddings, and format model predictions into structured actions suitable for environment stepping.

#### Scenario: Structured turn observation and dispatch
- **WHEN** an observation containing DOM elements, goal text, and dense features is supplied to the agent
- **THEN** the agent converts the observation into model inputs, advances internal state, and emits a structured action payload

### Requirement: Interactive Terminal CLI and ASCII Canvas
The CLI interface SHALL execute manual or benchmark task scenarios, printing step-by-step System 2 rationales, predicted action primitives, and a pure ASCII 2D visual canvas showing element boundaries and pointer click locations.

#### Scenario: Interactive terminal run with ASCII rendering
- **WHEN** a user or test runner executes the interactive agent CLI on a benchmark scenario
- **THEN** the console outputs ASCII element bounding boxes, click coordinate marks, rationale text, and environment feedback without throwing Unicode encoding errors

### Requirement: Verification Suite Compliance
The verification suite SHALL benchmark 64-step recurrent rollouts, latency profiling in milliseconds per step on CUDA in bfloat16, task completion evaluation, zero NaNs/Infs, and strict adherence to the 6.0 GB VRAM limit.

#### Scenario: Verification run passes all assertions
- **WHEN** verify_phase6a.py is executed on host GPU
- **THEN** all memory invariance, latency profiling, task completion, and VRAM ceiling (< 6144 MB) checks pass with exit code 0
