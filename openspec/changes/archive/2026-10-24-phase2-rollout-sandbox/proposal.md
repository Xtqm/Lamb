# Proposal: Phase 2 - Grounded Rollout Pipeline and Headless Browser Sandbox

## Why

With Phase 1 (offline DPO calibration) complete, the model has sharp decision boundaries on Mind2Web trajectories. However, online GRPO exploration still suffers from two critical bottlenecks: the rollout sampler truncates observations to 24 tokens (rendering the agent blind to page elements) and omits goal prompt conditioning, while the live browser environment lacks dense spatial IoU reward shaping against candidate element bounding boxes. Phase 2 eliminates these bottlenecks, giving the agent full perceptual visibility and immediate spatial learning signals during interactive rollouts.

## What Changes

- **Goal-Conditioned Rollouts**: Inject `Goal: {goal}` prompt prefix at episode initialization in `TrajectoryRolloutSampler` matching pretraining token grammar.
- **Uncapped Observation Context**: Expand rollout observation budget up to 512 tokens (eliminating the artificial 24-token cutoff) to capture all interactive elements.
- **FastDOMParser Integration**: Stream and format live DOM interactive elements (buttons, inputs, links, select, ARIA roles) with normalized bounding boxes into observation text.
- **Dense Spatial IoU Reward Shaping**: Implement spatial IoU rewards in `WebTaskOracle` ($R_{\text{spatial}} = \text{IoU}((cx, cy), \text{bbox}_{\text{target}})$) to provide continuous feedback on intermediate steps.

## Capabilities

### Modified Capabilities

- `rl-rollout-pipeline`: Enforce explicit goal prefixing, observation budget expansion up to 512 tokens, and `FastDOMParser` candidate element extraction during live trajectory sampling.
- `playwright-browser-environment`: Implement dense spatial IoU reward shaping against target element bounding boxes and structured DOM candidate extraction for live Playwright Chromium sessions.

## Impact

- **Affected Code**: `jed_m2/rl_pipeline.py`, `jed_m2/browser.py`.
- **Dependencies**: Native PyTorch and Playwright 1.64.0 Chromium runtime.
