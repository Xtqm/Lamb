# Design: Ground GRPO on Real Web Environments

## Context

The JED-M2 Nano model (53M parameters) has reached Step 15,800 with strong spatial grounding (0.3203 MAE) across 1,009 real-world Mind2Web websites. However, the existing RL harness in `jed_m2/rl_pipeline.py` and `train_rl.py` is decoupled from this pretraining distribution:
- Observations are hardcoded to slice the prompt at `[:24]` tokens, rendering the model blind to DOM elements.
- Initial episodes omit `Goal: {goal}` prompt prefixing.
- The default rollout environment relies on toy procedural state machines (`WebOSEnvironment`).
- Online programmatic task oracles only exist for three static HTML fixtures without dense spatial feedback.

See `proposal.md` for motivation and background.

## Goals / Non-Goals

**Goals:**
- **Stage 1 (Offline Preference Calibration)**: Align Step 15,800 Nano policy weights via Direct Policy Optimization (DPO) on the 1,009-task Mind2Web corpus using authentic expert trajectories and negative spatial/action perturbations.
- **Stage 2 (Pipeline Architectural Alignment)**: Upgrade `TrajectoryRolloutSampler` to prefix `Goal: {goal}` and tokenize candidate interactive elements (up to 512 tokens) parsed via `FastDOMParser`.
- **Stage 3 (Browser Oracle & Reward Shaping)**: Equip `PlaywrightEnvironment` with dense spatial IoU reward shaping against target element bounding boxes alongside milestone and terminal checks.
- **Stage 4 (Online GRPO Execution)**: Execute stable online Group Relative Policy Optimization rollouts ($G=4$) with reference policy KL divergence constraints within the 6.0 GB VRAM envelope.

**Non-Goals:**
- Retraining from scratch: Step 15,800 is the frozen foundation.
- Open-ended web navigation without deterministic task oracles: All rollouts require unambiguous programmatic verification.
- External compiled C++/CUDA packages: All implementations remain 100% native PyTorch.

## Decisions

### Decision 1: Stage 1 Offline DPO on Real Mind2Web Corpus Before Online GRPO
- **Rationale**: Online RL exploration from an uncalibrated policy on complex web tasks suffers from extreme reward sparsity (almost all rollouts score 0.0), causing gradient collapse. Running DPO directly on paired offline Mind2Web data sharpens action and spatial decision boundaries on authentic web layouts before online sampling.
- **Alternative Considered**: Starting directly with online GRPO. Rejected because reward sparsity and distribution shift from procedural environments cause policy degradation.

### Decision 2: Integrate `FastDOMParser` and Restore Observation Context Budget
- **Rationale**: `FastDOMParser` in `jed_m2/real_dataset.py` prunes complex DOM trees into candidate interactive elements (`button`, `input`, `a`, `select`, ARIA roles) with normalized bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$. Expanding the rollout observation budget from 24 to 512 tokens gives the model full perceptual visibility of candidate targets matching pretraining format.
- **Alternative Considered**: Passing raw full HTML into the context window. Rejected because raw DOMs exceed 10,000+ tokens and blow through memory limits.

### Decision 3: Enforce `Goal: {goal}` Token Prefixing at Episode Initialization
- **Rationale**: `RealTrajectoryCollator` trained the model with `Goal: {goal} <obs> ...`. In `TrajectoryRolloutSampler`, rollouts currently lack goal conditioning, forcing the model to guess actions without task context.
- **Alternative Considered**: Appending goals to every turn's observation. Rejected because it violates the established pretraining grammar and consumes token budget redundantly.

### Decision 4: Dense Spatial IoU Reward Shaping in `WebTaskOracle`
- **Rationale**: Discrete 0/1 terminal rewards make multi-step web tasks difficult to explore under temperature sampling. Adding continuous spatial IoU reward $R_{\text{spatial}} = \text{IoU}((cx, cy), \text{bbox}_{\text{target}})$ gives immediate positive reinforcement when the policy clicks the correct target element, even on intermediate steps.
- **Alternative Considered**: Sparse terminal reward only ($R_{\text{terminal}} = 1.0$). Rejected because credit assignment fails across 4-8 turn exploratory trajectories.

## Risks / Trade-offs

- **[Risk] Headless Playwright Chromium launch delays or sandbox permission blocks.**
  - *Mitigation*: Ensure browser launches with `--no-sandbox`, `--disable-dev-shm-usage`, and appropriate timeout handling; provide automated fallback to local fixture serving.
- **[Risk] VRAM exceeding 6.0 GB on RTX 5060 Laptop GPU during GRPO group rollouts ($G=4$).**
  - *Mitigation*: Enforce bfloat16 precision, activation checkpointing, sequence length limit $\le 1024$, and sequential rollout evaluation before batched policy gradient computation.
- **[Risk] Spatial coordinate mode collapse during offline DPO.**
  - *Mitigation*: Maintain reference policy KL constraint and bounded perturbation displacements ($0.15 \le |\Delta| \le 0.35$).
