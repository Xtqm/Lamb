# Tasks

## 1. Candidate Bounding Box Snapping in Agent Runtime

- [x] 1.1 Implement candidate element bounding box snapping in `jed_m2/agent.py` (`act()` and `step_environment()`), computing nearest candidate center with configurable distance gating threshold `max_snap_distance = 0.20`, and verify with unit test in `tests/test_candidate_snapping.py`
- [x] 1.2 Update `AgentStepOutput` container in `jed_m2/agent.py` to record `raw_coordinates` while preserving `coordinates` (snapped) for downstream environments and telemetry, verifying serialization via dictionary export test

## 2. Default 7 Action Primitives and Legacy Checkpoint Support

- [x] 2.1 Set default `num_action_types: int = 7` in `JEDM2Config` (`jed_m2/config.py`), ensuring newly instantiated models natively support the `SELECT` primitive
- [x] 2.2 Implement tensor zero-padding compatibility in model checkpoint loading helpers (`jed_m2/model.py`) to transparently pad legacy 6-class `fc2.weight` and `fc2.bias` tensors to 7 classes, verifying clean loading of legacy checkpoints without shape errors

## 3. Calibrated Argument Loss and Multi-Token Argument Generation

- [x] 3.1 Calibrate `arg_weight` in `JEDM2LossConfig` (`jed_m2/losses.py`) so argument token prediction gradients are proportional to action type cross-entropy and coordinate Huber losses, verifying finite loss calculation and gradient flow
- [x] 3.2 Implement an autoregressive token unrolling loop for `TYPE` actions in `JEDM2Agent.act()` (`jed_m2/agent.py`) and `rl_pipeline.py`, generating multi-token argument strings until `</arg>` delimiter or max 16 tokens, verifying multi-word query emission

## 4. Dual-Mode Benchmark Evaluation and System Integration

- [x] 4.1 Update `score_offline_windows()` in `jed_m2/real_eval.py` to support candidate snapping evaluation, computing and returning both raw continuous and DOM-grounded candidate-snapped metrics in parallel
- [x] 4.2 Execute full test suite (`pytest tests/`) to ensure all existing and new unit tests pass with 100% green status and zero regressions
