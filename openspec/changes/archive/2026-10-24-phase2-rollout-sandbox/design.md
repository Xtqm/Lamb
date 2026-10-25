# Design: Phase 2 - Grounded Rollout Pipeline and Headless Browser Sandbox

## Context

Following Phase 1 DPO calibration, the policy parameters are aligned with authentic web trajectories. However, `TrajectoryRolloutSampler` in `jed_m2/rl_pipeline.py` currently limits observation tokens to 24 and drops goal conditioning, while `PlaywrightEnvironment` in `jed_m2/browser.py` relies on sparse terminal oracle checks. Phase 2 unifies observation parsing and reward shaping across rollouts.

## Goals / Non-Goals

**Goals:**
- Prepend `Goal: {goal}` prompt prefixing at rollout episode start.
- Expand observation budget to 512 tokens and format interactive elements using `FastDOMParser`.
- Add dense spatial IoU reward shaping into `WebTaskOracle` ($R_{\text{spatial}} = \text{IoU}((cx, cy), \text{bbox}_{\text{target}})$).
- Verify interactive browser rollouts in headless Chromium.

**Non-Goals:**
- Online GRPO training (deferred to Phase 3).
- Adding unverified live external web domains.

## Decisions

### Decision 1: Prefix `Goal: {goal}` at $t=0$ in `sample_trajectory`
- **Rationale**: Aligns the rollout token grammar with `RealTrajectoryCollator`.
- **Alternative Considered**: Re-encoding goal at every turn. Rejected as redundant and wasteful of context tokens.

### Decision 2: Remove `[:24]` Slice and Ingest up to 512 Observation Tokens
- **Rationale**: Real webpages require 128-512 tokens to express interactive candidate elements (`button`, `input`, `a`, `select`). Slicing to 24 blinded the agent.

### Decision 3: Continuous Spatial IoU Reward in `WebTaskOracle`
- **Rationale**: Provides dense intermediate reward feedback whenever a click action lands on or near a target element, preventing advantage starvation during multi-step tasks.

## Risks / Trade-offs

- **[Risk] Long sequence length exceeding memory during multi-turn rollouts.**
  - *Mitigation*: Cap turns per episode and enforce `max_seq_len = 1024`.
