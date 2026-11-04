# Proposal

## Why

Offline evaluation of JED-M2 Nano demonstrated strong step accuracy (81.24%) and multi-element candidate snapping hit rate (93.06%), yet interactive evaluation across WebOS and browser episodes yielded 0.0% end-to-end task success. The root cause established by both empirical telemetry and recent SOTA literature (e.g., UINav, CC-Net, MobileRL) is twofold: (1) the absence of a deterministic macro-action execution scaffold (rendering stability polling, input clearing before typing) which exposes small models to compounding asynchronous timing errors, and (2) the missing "Feasibility and Completion Referee", causing agents to execute correct initial subgoals but never emit `ActionType.FINISH`, truncating at maximum steps with zero terminal reward. Implementing this scaffold and referee layer (Phase 2) is the prerequisite before launching online reinforcement learning (GRPO).

## What Changes

- **Deterministic Macro-Action Execution Scaffold**: Implement execution wrappers that handle low-level browser and environment mechanics deterministically:
  - Automatic input clearing (`Ctrl+A` / clear value) before typing text into input fields.
  - DOM mutation and render stability polling to ensure DOM elements settle before dispatching the next model action.
  - Safe click retries on transient detached elements.
- **Task Feasibility & Completion Referee**:
  - Implement a programmatic referee in `WebOSEnvironment`, `RealTaskDOMEnvironment`, and `rl_pipeline.py` that monitors task milestone satisfaction.
  - When all goal conditions are satisfied, the referee signals task completion and injects or validates the `ActionType.FINISH` transition, eliminating the infinite 8-step truncation loop.
  - Terminate infeasible or repeating action loops early to preserve exploration budgets.
- **Hierarchical Decoupled RL Reward Calibration**:
  - Calibrate the GRPO reward function to follow the decoupled formulation:
    $$R = w_{\text{type}} R_{\text{type}} + w_{\text{coord}} R_{\text{candidate-hit}} + w_{\text{arg}} R_{\text{content}} + w_{\text{complete}} R_{\text{referee-complete}}$$
  - Reward distractor-aware candidate snapping hits directly, aligning online RL incentives with the 93.06% candidate snapping capability.

## Capabilities

### Modified Capabilities
- `interactive-webos-environment`: Add macro-action transition handling (automatic input clearing, state settling) and milestone-aware goal verification that works with the completion referee.
- `rl-rollout-pipeline`: Incorporate the deterministic execution scaffold, render stability polling, referee-assisted task termination, and hierarchical decoupled reward formulation into rollout sampling and GRPO batch collation.

## Impact

- `jed_m2/environment.py`: Updates to `WebOSEnvironment.step` to support input clearing and referee-accessible goal state queries.
- `jed_m2/real_environment.py`: Integration of completion referee evaluation and macro-action settling into `RealTaskDOMEnvironment`.
- `jed_m2/rl_pipeline.py`: Addition of `ExecutionScaffold` and `CompletionReferee` into `TrajectoryRolloutSampler.sample_trajectory`.
- `jed_m2/rl.py`: Decoupled multi-factor reward formulation in `compute_shaped_reward`.
- `train_rl.py`: Verification that online GRPO rollouts achieve non-zero completion rewards and normal gradient norms prior to large-scale runs.
