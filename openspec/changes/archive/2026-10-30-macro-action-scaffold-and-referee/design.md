# Design

## Context

See `proposal.md` for background and motivation. JED-M2 Nano demonstrated high offline spatial precision (93.06% candidate snapping hit rate), but scored 0.0% interactive task completion in live WebOS episodes. The policy executes correct subgoals (typing search queries, clicking search and filter buttons) but fails to emit `ActionType.FINISH`, causing every episode to truncate at 8 steps with zero terminal completion reward. Furthermore, raw execution without input clearing and render settling exposes the compact policy to timing desynchronization and compounding covariate shift.

## Goals / Non-Goals

**Goals:**
- Provide a deterministic `ExecutionScaffold` that abstracts asynchronous timing, renders settling, and executes input-clearing macro-actions before typing.
- Provide a `CompletionReferee` that inspects environment state machines and programmatic oracles to identify milestone and goal completion, signaling termination and preventing infinite-loop truncation.
- Integrate the scaffold and referee into `rl_pipeline.py` so that interactive GRPO rollouts achieve non-zero completion rewards and normal advantage contrast.
- Calibrate `compute_shaped_reward` to the SOTA decoupled reward formulation ($w_{\text{type}} R_{\text{type}} + w_{\text{coord}} R_{\text{coord}} + w_{\text{arg}} R_{\text{arg}} + w_{\text{complete}} R_{\text{complete}}$).

**Non-Goals:**
- Training a neural referee model from scratch (a lightweight, deterministic programmatic referee achieves 100% rule-based precision on WebOS and benchmark oracles without eating parameters).
- Altering the underlying JED-M2 model architecture or vocabulary.
- Launching the full 25-iteration GRPO training run in this change (full training is authorized only after Phase 2 verification passes).

## Decisions

### 1. Dedicated `jed_m2/scaffold.py` Module
- **Decision**: Centralize macro-action mechanics (`ExecutionScaffold`) and termination referee (`CompletionReferee`) in a dedicated module `jed_m2/scaffold.py`, imported by `WebOSEnvironment`, `RealTaskDOMEnvironment`, and `rl_pipeline.py`.
- **Rationale**: Isolates execution wrapper logic from core model inference and raw environment physics. Keeps browser and WebOS interfaces consistent.
- **Alternatives Considered**: Inlining referee logic into `rl_pipeline.py` (rejected: would duplicate code between interactive evaluation in `real_eval.py` and RL sampling).

### 2. Rule-Based Programmatic Referee vs. Neural Referee
- **Decision**: Implement a programmatic referee inspecting environment state metadata (e.g. `stage == 'cart_confirmed'`, `stage == 'booked'`, or oracle criteria).
- **Rationale**: The environments (`WebOSEnvironment`, `PlaywrightEnvironment`, `RealTaskDOMEnvironment`) already maintain deterministic state machines and declarative oracles. A programmatic referee is zero-overhead, 100% verifiable, and adds no inference latency or VRAM footprint.
- **Alternatives Considered**: 430k parameter neural referee like UINav (unnecessary for simulated/oracle environments, though can be added later if pure black-box visual evaluation is needed).

### 3. Decoupled Hierarchical Reward Formulation
- **Decision**: Update `compute_shaped_reward` in `jed_m2/rl.py` to decouple element grounding, action type validity, and terminal success:
  $$R = R_{\text{action\_type}} + R_{\text{element\_hit}} + R_{\text{argument}} + R_{\text{referee\_complete}} - R_{\text{step\_penalty}}$$
- **Rationale**: In the previous formulation, the $-0.05$ step penalty dominated rollouts because completion reward was unreachable without `FINISH`. Decoupling rewards ensures that each correct action receives positive reinforcement regardless of whether the entire trajectory finishes.

## Risks / Trade-offs

- **[Risk]** Referee terminating prematurely before user-intended multi-step completion.
  - **Mitigation**: Require all stage-specific terminal indicators (e.g. `cart_confirmed`, confirmed reservation) to evaluate True before referee signals completion.
- **[Risk]** Over-reliance on referee masking poor policy `FINISH` emission.
  - **Mitigation**: Award an additional policy bonus (+0.5) when the model natively emits `ActionType.FINISH` in the terminal state versus relying solely on referee forced termination.
