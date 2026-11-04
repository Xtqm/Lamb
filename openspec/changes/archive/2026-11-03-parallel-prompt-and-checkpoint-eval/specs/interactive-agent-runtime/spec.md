# Spec Delta: interactive-agent-runtime

## MODIFIED Requirements

### Requirement: Observation Parsing and Action Dispatching
The agent runtime SHALL parse environment DOM text, live accessibility trees, and observation vectors into token IDs and tensor embeddings, ingest observation token sequences using parallel associative scan prefill (`model.forward(...)`) rather than sequential token-by-token loops, synchronize live screenshots with the visual patch encoder, and dispatch structured actions to both simulated and live browser environments.

#### Scenario: Structured turn observation and dispatch
- **WHEN** an observation containing DOM elements, goal text, and dense features is supplied to the agent
- **THEN** the agent converts the observation into model inputs, advances internal state, and emits a structured action payload

#### Scenario: Live browser observation ingestion and dispatch
- **WHEN** the agent steps against a live browser environment yielding live accessibility trees and real viewport screenshots
- **THEN** the agent extracts DOM token sequences, computes visual patch tokens via the patch encoder, conditions coordinate predictions on visual features, and dispatches the action to the live browser

#### Scenario: Parallel observation prompt prefill latency
- **WHEN** the agent ingests an observation prompt sequence containing up to 2,048 DOM tokens
- **THEN** the prompt is processed in a single parallel tensor prefill pass via parallel scan and attention, bounding prefill latency to under 100 ms on CUDA and correctly seeding layer-wise recurrent states for subsequent autoregressive rollout
