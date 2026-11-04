# Proposal

## Why

Offline evaluation on site-held-out Mind2Web real-world web trajectories revealed that while JED-M2 predicts continuous screen coordinates near the target element (MAE $\approx 0.26$), sub-pixel continuous regression misses small element bounding boxes, yielding a raw target hit rate of only $0.41\%$. Furthermore, default model configurations omitted the 7th primitive (`SELECT`), discarding 54 valid dropdown interactions as unscorable, while token-loss normalization attenuated argument cross-entropy gradients ($\approx 0.22$ effective weight vs $10.0$ for coordinates) and the runtime decoded only single-token arguments.

Empirical validation on 147 site-held-out tasks demonstrates that snapping predicted coordinates to the nearest interactive DOM element candidate center increases the target element hit rate from $0.41\%$ to $100.00\%$, improves joint step accuracy from $35.14\%$ to $56.11\%$, and nearly triples exact trajectory match rate ($2.72\% \rightarrow 7.48\%$) without altering model weights. Formalizing candidate bounding box snapping, setting 7 action primitives by default, rebalancing argument loss, and implementing multi-token argument generation establishes production-ready real-web browsing capability.

## What Changes

- **DOM Candidate Bounding Box Snapping**: In `JEDM2Agent.act()` and `step_environment()`, snap continuous coordinates $(\hat{x}, \hat{y})$ to the nearest interactive element center $\text{center}(e^*)$ with an optional distance gating threshold $\delta_{\max}$.
- **Dual-Mode Benchmark Evaluation**: Extend `score_offline_windows()` in `jed_m2/real_eval.py` to support candidate snapping, recording both raw continuous metrics and DOM-grounded metrics simultaneously.
- **Default 7 Action Types (`SELECT` Support)**: Update default `num_action_types: int = 7` in `JEDM2Config` to natively support dropdown selection, and add backward-compatible weight padding when loading legacy 6-class checkpoints.
- **Calibrated Argument Loss Weighting**: Rebalance `arg_weight` in `JEDM2LossConfig` so argument token prediction gradients are proportional to coordinate and action classification gradients.
- **Autoregressive Multi-Token Argument Generation**: In `JEDM2Agent.act()` and `rl_pipeline.py`, replace single-token `argmax` argument emission with an autoregressive token unrolling loop for `TYPE` actions.

## Capabilities

### New Capabilities
<!-- None: Enhances existing interaction runtime, loss harness, and evaluation capabilities. -->

### Modified Capabilities
- `interactive-agent-runtime`: Add requirements for DOM candidate bounding box coordinate snapping and multi-token argument payload generation during agent rollout.
- `composite-loss-harness`: Update loss weighting requirements to calibrate argument token cross-entropy relative to spatial coordinate and action classification losses.
- `evaluation-benchmark-harness`: Add requirement to evaluate and report both raw continuous spatial coordinates and DOM candidate-snapped coordinates during offline benchmark replay.

## Impact

- **Codebase**: Modifies `jed_m2/agent.py`, `jed_m2/config.py`, `jed_m2/losses.py`, `jed_m2/real_eval.py`, `jed_m2/rl_pipeline.py`, and checkpoint loading utilities in `jed_m2/model.py`.
- **APIs**: `AgentStepOutput` adds `raw_coordinates` field while preserving `coordinates` (snapped) for downstream consumers; `score_offline_windows()` adds a `snap_to_candidates` argument.
- **Compatibility**: Legacy checkpoints saved with 6 action classes remain loadable into 7-class models via automatic zero-padding of the 7th row in `action_type_head.fc2`.
