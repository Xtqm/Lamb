# Spec Delta

## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: Multimodal Visual Grounding in Recurrent Agent Step
The agent runtime SHALL pass live screenshot visual patch tokens to `model.step(...)` during recurrent rollout, enabling visual cross-attention coordinate grounding on real web pages while preserving $O(1)$ constant-memory recurrence.

#### Scenario: Live screenshot patch grounding during recurrent step
- **WHEN** a live screenshot tensor is supplied during `agent.act(...)` or `agent.step_environment(...)`
- **THEN** the agent projects the image through `VisionPatchEncoder` into patch tokens, feeds them to `model.step(..., visual_tokens=visual_tokens)`, and bounds coordinate outputs to $[0, 1] \times [0, 1]$
