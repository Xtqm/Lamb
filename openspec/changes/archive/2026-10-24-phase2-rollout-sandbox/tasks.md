# Tasks: Phase 2 - Grounded Rollout Pipeline and Headless Browser Sandbox

## 1. Rollout Sampler Observation & Goal Ingestion

- [x] 1.1 Update `TrajectoryRolloutSampler` in `jed_m2/rl_pipeline.py` to prepend explicit `Goal: {goal}` prompt prefix at episode initialization, matching pretraining token grammar.
- [x] 1.2 Remove the hardcoded `[:24]` token truncation in `TrajectoryRolloutSampler`, expanding observation token budget up to 512 tokens to give full perceptual visibility of interactive elements.

## 2. Browser Environment & Dense Spatial Oracle Enhancement

- [x] 2.1 Upgrade `PlaywrightEnvironment` in `jed_m2/browser.py` to extract candidate interactive elements with bounding boxes from live DOM and format observation text compatible with `FastDOMParser`.
- [x] 2.2 Implement spatial IoU dense reward shaping in `WebTaskOracle` ($R_{\text{spatial}} = \text{IoU}((cx, cy), \text{bbox}_{\text{target}})$) alongside existing milestone and terminal reward logic.
- [x] 2.3 Verify live browser rollout execution by sampling a test rollout episode in headless Chromium with `TrajectoryRolloutSampler` and confirming valid observation tokens and oracle rewards.
