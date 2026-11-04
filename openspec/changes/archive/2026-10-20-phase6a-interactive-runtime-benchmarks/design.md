# Design

## Context

Prior phases completed the foundational building blocks of the JED-M2 Large Action Model:
- **Phase 1**: Low-Rank Multi-Head Latent Attention (`mla.py`) and Sparse Top-2 MoE (`moe.py`).
- **Phase 2**: Pure PyTorch causal Mamba SSM (`mamba.py`) and JEPA Latent World Model (`jepa.py`).
- **Phase 3**: Dual-System Action Heads (`heads.py`) and Unified 3-Block Backbone (`model.py`) supporting sequence forward and recurrent `step(...)`.
- **Phase 4**: Synthetic Multi-Turn Trajectory Dataset (`dataset.py`) and Multi-Task Loss Harness (`losses.py`).
- **Phase 5**: Production Training Harness (`trainer.py`) and RL Scaffolding (`rl.py`).

In Phase 6A, we construct the interactive execution layer:
1. `jed_m2/environment.py`: Simulated Web/OS sandbox environment.
2. `jed_m2/agent.py`: Interactive runtime execution engine.
3. `interactive_agent.py`: Interactive CLI entrypoint with pure ASCII UI canvas rendering.
4. `verify_phase6a.py`: Comprehensive verification and benchmarking script.

All components must strictly adhere to the project constraints:
- 100% pure PyTorch (no external C++/CUDA compilation packages like `mamba-ssm` or `causal-conv1d`).
- `torch.bfloat16` precision on NVIDIA GeForce RTX 5060 Laptop GPU (sm_120 Blackwell architecture).
- Strict VRAM ceiling of 6.0 GB (6144 MB).
- Pure ASCII console outputs for Windows 11 CP1252 compatibility.

## Goals / Non-Goals

**Goals:**
- Provide a robust, deterministic state machine (`WebOSEnvironment`) modeling web and OS desktop workflows with interactive UI element trees, bounding boxes in $[0, 1]$, action transitions (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH), spatial coordinate hit-testing, step rewards, and goal evaluation.
- Provide an interactive agent runtime (`JEDM2Agent`) wrapping `JEDM2Model` with $O(1)$ constant-memory autoregressive recurrent rollout via `step(...)`, retaining layer-wise Mamba SSM states, conv buffers, and MLA KV-cache across multiple turns.
- Implement decoupled dual-system execution: autoregressively generate latent rationale tokens (System 2) until `ActionGate` exceeds the action threshold, then decode discrete primitive, continuous $(x, y) \in [0, 1]$ screen coordinates, and argument payloads (System 1).
- Provide an interactive terminal CLI (`interactive_agent.py`) with manual goal input, scenario selection, and a pure ASCII visual canvas representation of UI elements and pointer targets.
- Provide a standalone verification benchmark (`verify_phase6a.py`) testing 64-step constant-memory invariance, latency profiling (ms/step), task completion evaluation, and strict VRAM ceiling compliance.

**Non-Goals:**
- Heavy external browser automation engines (Playwright, Puppeteer, Selenium) or OS accessibility APIs - this phase provides a self-contained simulated digital sandbox.
- Online RL training loops (DPO/GRPO policy iteration) - this is deferred to Phase 6B (`train_rl.py`).
- Multimodal visual screenshot patch encoders - deferred to Phase 6D (`jed_m2/vision.py`).

## Decisions

### 1. Simulated Web/OS Environment State Machine (`jed_m2/environment.py`)

- **State Representation**:
  - `EnvironmentState`: Captures current URL/page name, viewport scroll offset, list of active `UIElement` instances, focused element ID, history of executed actions, step count, and cumulative reward.
  - Interactive element flags: `tag`, `role`, `text`, `value`, `is_clickable`, `is_input`, `is_scrollable`, `bbox = (x_min, y_min, x_max, y_max)`.
- **Action Transition Semantics**:
  - `CLICK`: Evaluates spatial coordinates $(x, y) \in [0, 1]$. Performs spatial hit-testing across all active elements. If coordinates land inside an element's bounding box, activates the element (toggles button, focuses input, selects dropdown). If coordinates miss, records an off-target click.
  - `TYPE`: Applies string argument payload to the focused or hit-tested text input element.
  - `SCROLL`: Adjusts normalized scroll offset and reveals scroll-dependent elements.
  - `NAVIGATE`: Changes active page/URL and loads the target page's element tree.
  - `API_CALL`: Executes simulated programmatic API action and returns API response in observation text.
  - `FINISH`: Signals episode completion and triggers final goal verification.
- **Deterministic Evaluation & Rewards**:
  - Evaluates whether the terminal state meets the scenario goal conditions (e.g. required element clicked, correct text entered, successful navigation).
  - Step reward: $+0.1$ for productive step toward goal, $-0.05$ step penalty, $+1.0$ for successful goal completion, $-0.2$ for invalid action.

### 2. Interactive Agent Runtime Architecture (`jed_m2/agent.py`)

- **Stateful Recurrent Rollout**:
  - Maintains `recurrent_states: List[Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]`: Per-layer tuple of `(ssm_state, conv_state, kv_cache)` across all 3 `JEDM2Block` layers.
  - Retains global sequence position `pos: int` tracking cumulative tokens rolled out.
  - Step function invokes `model.step(token, states=recurrent_states, pos=pos)` one token at a time.
- **Decoupled Dual-System Decision Loop**:
  1. *Observation Ingestion*: Serializes environment state and goal into prompt tokens using `SimpleLAMTokenizer`. Steps model over observation tokens to update internal Mamba and KV states.
  2. *System 2 Latent Reasoning*: Appends `<thought>` token. Autoregressively generates thought tokens one by one by sampling/argmax from `lm_logits`. At each thought token, queries `action_output.gate_logits`.
  3. *Action Commitment*: When `sigmoid(gate_prob) >= threshold` OR `thought_len >= max_thought_tokens`, terminates System 2 reasoning and commits to action emission.
  4. *System 1 Action Decoding*: Reads `action_output.action_type_logits` (argmax discrete primitive), `action_output.coordinates` (continuous $(x, y)$), and `action_output.argument_logits` (decoded string payload).
  5. Emits `AgentStepOutput` and records turn into agent history.

### 3. $O(1)$ Constant-Memory Invariance

- **Why constant memory**: Standard Transformers append historical tokens to a context window, causing $O(T^2)$ or $O(T)$ memory growth and eventual out-of-memory errors on long horizons.
- **JED-M2 Mechanism**:
  - Mamba SSM states (`ssm_state`: $[1, 1024, 16]$, `conv_state`: $[1, 1024, 4]$) remain strictly fixed in shape regardless of horizon $T$.
  - MLA KV cache stores low-rank compressed keys and values ($c_{KV}=128$). Over 64 steps, the KV cache addition is mere kilobytes: $3 \text{ layers} \times 64 \text{ tokens} \times 128 \times 2 \text{ bytes} \approx 49 \text{ KB}$.
  - Thus, active allocated GPU VRAM remains flat across steps 16 to 64 within < 5% variance.

### 4. Interactive CLI & ASCII Visual Canvas (`interactive_agent.py`)

- **Pure ASCII UI Renderer**:
  - To prevent Windows terminal CP1252 encoding exceptions, all canvas rendering uses strictly 7-bit ASCII characters (`+`, `-`, `|`, `*`, `[`, `]`, ` `).
  - Renders a 60x18 text grid representing normalized $[0, 1] \times [0, 1]$ screen space.
  - Bounding boxes drawn as rectangles with element labels; pointer click coordinates marked with `*` or `[X]`.
- **Interactive Modes**:
  - Mode 1: Automated benchmark scenario execution (e-commerce, travel, os admin).
  - Mode 2: Interactive step-by-step mode pausing after each turn for user inspection or enter key.

### 5. Verification Benchmark Suite (`verify_phase6a.py`)

- **Structure**:
  1. `test_environment_simulation()`: Tests state transitions, spatial hit testing, reward signals, and termination.
  2. `test_agent_runtime_rollout()`: Tests agent initialization, observation encoding, dual-system thought-to-action transition, and state reset.
  3. `test_constant_memory_invariance()`: Benchmarks 64 consecutive recurrent steps on CUDA in `bfloat16`, asserting VRAM memory variation between step 16 and step 64 is < 5%.
  4. `test_latency_profiling()`: Measures CUDA kernel event timing (ms/step) over 32 steps.
  5. `test_end_to_end_scenarios()`: Executes full multi-turn benchmark episodes and measures completion accuracy.
  6. `test_vram_ceiling_and_health()`: Asserts peak VRAM < 6144 MB and verifies zero NaNs/Infs in all model outputs.

## Risks / Trade-offs

- **[Risk]** Autoregressive thought generation without fine-tuned weights could loop indefinitely if action gate never fires.
  → **Mitigation**: Implement a strict `max_thought_tokens` ceiling (default 16 tokens) in `JEDM2Agent`. If the threshold is not reached, force action commitment.
- **[Risk]** Windows CP1252 terminal crashes on non-ASCII box-drawing characters.
  → **Mitigation**: Enforce 100% pure ASCII characters (`+`, `-`, `|`, `#`, `*`) across all environment, agent, CLI, and verification printouts.
- **[Risk]** Latency on CUDA without compiled kernels.
  → **Mitigation**: Pure PyTorch operations in `bfloat16` are natively accelerated by Blackwell Tensor Cores; recurrent `step(...)` operates on batch size 1 with minimal memory traffic.
