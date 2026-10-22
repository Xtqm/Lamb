# Spec Delta

## Purpose

Provides an asynchronous background sidecar daemon, stateful multi-turn session management, human-in-the-loop safety supervision, real-time telemetry streaming, and interactive CLI HUD controls for local desktop automation.

## ADDED Requirements

### Requirement: Asynchronous Daemon REST and WebSocket Endpoints
The sidecar daemon SHALL expose asynchronous HTTP and WebSocket endpoints including POST /task to submit natural language automation goals, GET /status to query execution telemetry, POST /abort for emergency stops, and GET /stream for real-time thought and action event streaming.

#### Scenario: Client task submission and streaming
- **WHEN** a client posts a task goal to the /task endpoint
- **THEN** the daemon accepts the goal, initiates policy execution, and streams thought rationales and action events across the /stream WebSocket.

#### Scenario: Emergency abort execution
- **WHEN** a client sends a POST request to /abort during an active automation task
- **THEN** the daemon cancels ongoing execution immediately, releases mouse and keyboard controls, and sets execution status to aborted.

### Requirement: Low Idle Footprint and GPU Resource Management
The sidecar daemon SHALL maintain an idle process footprint below 100 MB of system RAM and 0% idle GPU utilization when no tasks are active, while respecting a 6.0 GB total VRAM ceiling on the NVIDIA GeForce RTX 5060 Laptop GPU during active inference.

#### Scenario: Idle resource verification
- **WHEN** the sidecar daemon is launched and idling without an assigned task
- **THEN** host system memory consumption remains under 100 MB and GPU active utilization remains at 0%.

### Requirement: Stateful Multi-Turn Session Management
The sidecar runtime SHALL maintain recurrent Mamba SSM hidden states, 1D convolution rolling buffers, and MLA KV-cache across multiple steps of a task without recomputing historical tokens or causing sequence length explosion.

#### Scenario: Multi-turn constant memory rollout
- **WHEN** multiple successive interaction turns are executed within an active session
- **THEN** recurrent state buffers are updated in $O(1)$ memory without increasing inference memory allocation per step.

### Requirement: Human-in-the-Loop Safety Interception
The safety supervisor SHALL intercept candidate agent actions prior to dispatch and halt execution when detecting destructive file system operations, authentication or password input fields, or financial payment transactions, requiring explicit human approval.

#### Scenario: Sensitive operation interception
- **WHEN** an action proposes a destructive command (such as rm or format) or targets financial/credential input controls
- **THEN** the supervisor pauses execution, notifies the user via toast notification or CLI prompt, and blocks dispatch until confirmation is granted.

### Requirement: Model Scale Flexibility
The sidecar daemon and CLI SHALL support loading either JED-M2-Nano (51M) or JED-M2-Medium (273M) architectures via configuration flags or CLI parameters without architectural code modifications.

#### Scenario: Loading specified model size
- **WHEN** the daemon is initiated with `--model_size nano` or `--model_size medium`
- **THEN** the corresponding model architecture configuration and weights are loaded into memory and initialized for inference.

### Requirement: Interactive CLI HUD and Real-Time Monitoring
The sidecar CLI SHALL provide unified `daemon`, `exec`, and `monitor` commands, rendering an interactive ASCII terminal HUD displaying step-by-step System 2 thoughts, committed action primitives, and target coordinate overlays.

#### Scenario: Live monitoring of active daemon
- **WHEN** a user executes `python sidecar_agent.py monitor`
- **THEN** the CLI attaches to the local daemon WebSocket stream and renders step-by-step System 2 rationales, action commitments, and target coordinates in real time.
