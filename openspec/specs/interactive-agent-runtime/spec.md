# interactive-agent-runtime Specification

## Purpose
Provides an interactive agent runtime engine wrapping JEDM2Model for live closed-loop decision making, constant-memory recurrent rollouts, dual-system rationale and action decoding, and ASCII visual terminal inspection.

## Requirements

### Requirement: Constant-Memory Recurrent Agent Step
The agent runtime SHALL execute multi-turn interactive rollouts using $O(1)$ constant-memory autoregressive recurrent updates via `model.step(...)`, persisting layer-wise Mamba SSM states, conv buffers, and MLA KV-cache across turns without reprocessing past token history.

#### Scenario: Recurrent rollout memory invariance
- **WHEN** the agent executes an extended sequence of up to 64 consecutive interactive steps
- **THEN** GPU VRAM allocation remains bounded and constant within a 5% tolerance across steps 16 through 64

#### Scenario: Agent state reset
- **WHEN** the agent's reset method is invoked between episodes
- **THEN** recurrent state buffers and turn history are reinitialized to clean initial states

### Requirement: Decoupled Dual-System Decision Loop
The agent runtime SHALL execute a decoupled dual-system decision loop that generates latent thought tokens (System 2) until the ActionGate logit exceeds the commitment threshold, then emits the discrete action primitive (optionally refined via JEPA latent rollout planning), continuous $(x, y) \in [0, 1]$ screen coordinates, and argument payloads (System 1).

#### Scenario: System 2 reasoning commits to System 1 action
- **WHEN** the agent processes a new environment turn observation
- **THEN** it generates reasoning thought tokens until the action gate fires or maximum thought length is reached, returning both the rationale and structured action

#### Scenario: Thought generation guardrail
- **WHEN** the action gate does not exceed threshold within a configurable maximum reasoning budget
- **THEN** the runtime forcefully triggers action commitment to prevent execution deadlock

#### Scenario: Latent planning integration within decision loop
- **WHEN** active JEPA planning is enabled and thought generation concludes
- **THEN** the decision loop runs latent action rollouts to verify or re-rank candidate action primitives before dispatching the chosen action to the environment

### Requirement: Observation Parsing and Action Dispatching
The agent runtime SHALL parse environment DOM text, live accessibility trees, and observation vectors into token IDs and tensor embeddings, synchronize live screenshots with the visual patch encoder, and dispatch structured actions to both simulated and live browser environments.

#### Scenario: Structured turn observation and dispatch
- **WHEN** an observation containing DOM elements, goal text, and dense features is supplied to the agent
- **THEN** the agent converts the observation into model inputs, advances internal state, and emits a structured action payload

#### Scenario: Live browser observation ingestion and dispatch
- **WHEN** the agent steps against a live browser environment yielding live accessibility trees and real viewport screenshots
- **THEN** the agent extracts DOM token sequences, computes visual patch tokens via the patch encoder, conditions coordinate predictions on visual features, and dispatches the action to the live browser

### Requirement: Interactive Terminal CLI and ASCII Canvas
The CLI interface SHALL support both simulated sandbox environments and live Playwright browser sessions via a `--browser {simulated,playwright}` flag, printing step-by-step System 2 rationales, predicted action primitives, and an ASCII visual canvas showing element boundaries and pointer locations.

#### Scenario: Interactive terminal run with ASCII rendering
- **WHEN** a user or test runner executes the interactive agent CLI on a benchmark scenario
- **THEN** the console outputs ASCII element bounding boxes, click coordinate marks, rationale text, and environment feedback without throwing Unicode encoding errors

#### Scenario: Live browser session execution via CLI
- **WHEN** a user executes `interactive_agent.py` with `--browser playwright` and an optional `--url`
- **THEN** the CLI launches a live browser automation session, navigates to the target page, displays extracted elements, and runs multi-turn agent interaction steps

### Requirement: Multimodal Visual Grounding in Recurrent Agent Step
The agent runtime SHALL pass live screenshot visual patch tokens to `model.step(...)` during recurrent rollout, enabling visual cross-attention coordinate grounding on real web pages while preserving $O(1)$ constant-memory recurrence.

#### Scenario: Live screenshot patch grounding during recurrent step
- **WHEN** a live screenshot tensor is supplied during `agent.act(...)` or `agent.step_environment(...)`
- **THEN** the agent projects the image through `VisionPatchEncoder` into patch tokens, feeds them to `model.step(..., visual_tokens=visual_tokens)`, and bounds coordinate outputs to $[0, 1] \times [0, 1]$

### Requirement: Verification Suite Compliance
The verification suite SHALL benchmark 64-step recurrent rollouts, latency profiling in milliseconds per step on CUDA in bfloat16, task completion evaluation, zero NaNs/Infs, and strict adherence to the 6.0 GB VRAM limit.

#### Scenario: Verification run passes all assertions
- **WHEN** verify_phase6a.py is executed on host GPU
- **THEN** all memory invariance, latency profiling, task completion, and VRAM ceiling (< 6144 MB) checks pass with exit code 0

### Requirement: Active JEPA Model-Predictive Latent Rollout Planning
The agent runtime SHALL provide an active model-predictive planning mechanism utilizing the JEPA latent world model to simulate and evaluate future state representations before committing to actions.

#### Scenario: Candidate action latent rollout evaluation
- **WHEN** the agent reaches action commitment for an observation with goal embedding $z_{\text{goal}}$ and current latent state $z_t$
- **THEN** the agent predicts next latent states $\hat{z}_{t+1} = \text{Predictor}(z_t, a_i)$ for candidate action primitives $a_i \in \{\text{CLICK}, \text{TYPE}, \text{SCROLL}, \text{NAVIGATE}\}$ and computes the cosine distance $1 - \cos(\hat{z}_{t+1}, z_{\text{goal}})$ for each candidate

#### Scenario: Predictive action selection and re-ranking
- **WHEN** active JEPA planning is enabled during agent execution
- **THEN** candidate actions are scored by combining policy logits with negative predictive goal distance, selecting the action that most directly progresses toward the latent goal state

#### Scenario: Graceful fallback when JEPA model is unavailable
- **WHEN** `model.jepa` is not configured or disabled
- **THEN** the agent falls back to direct policy argmax decoding without error or latency degradation
