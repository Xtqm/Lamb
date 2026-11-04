# Tasks: Native FINISH Action Calibration & Referee Decoupling

## 1. Terminal Reward Calibration & Loss Weighting

- [x] 1.1 Update `compute_shaped_reward()` in `jed_m2/rl.py` to accept `terminal_action: Optional[ActionType]` and implement the asymmetric terminal reward schedule (+2.5 for valid `FINISH`, -1.5 for premature `FINISH`, -0.5 for timeouts) and verify via unit tests in `tests/test_native_finish.py`
- [x] 1.2 Update `JEDM2LossConfig` and `JEDM2Loss` in `jed_m2/losses.py` to add `finish_action_weight: float = 2.0` on terminal `ActionType.FINISH` cross-entropy targets and verify via unit tests in `tests/test_argument_and_loss.py`

## 2. In-The-Loop Referee Decoupling & Autonomous Agent Termination

- [x] 2.1 Decouple `CompletionReferee.evaluate_termination()` from the step loop in `TrajectoryRolloutSampler.collect_episode()` and `sample_trajectory()` in `jed_m2/rl_pipeline.py`, terminating solely on agent `ActionType.FINISH` or step exhaustion, and calling referee post-hoc
- [x] 2.2 Update `JEDM2Agent.step()` and interactive execution loop in `jed_m2/agent.py` to record `ActionType.FINISH` as terminal and yield execution without referee dependency
- [x] 2.3 Update `CompletionReferee` in `jed_m2/scaffold.py` documentation and interface to formalize its post-hoc evaluation role and verify with `tests/test_scaffold.py`

## 3. Benchmark Harness Metrics & Post-Hoc Reporting

- [x] 3.1 Update `BenchmarkMetrics` and evaluation routines in `jed_m2/benchmark.py` to compute and report `autonomous_success_rate`, `referee_assisted_rate`, and `native_finish_f1` (precision, recall, F1)
- [x] 3.2 Add integration tests in `tests/test_native_finish.py` verifying that rollouts ending with valid `FINISH` receive full terminal rewards while rollouts without `FINISH` receive no completion bonus

## 4. End-to-End Verification & Test Suite

- [x] 4.1 Run the complete test suite (`pytest tests/`) and verify all tests pass with zero regressions
- [x] 4.2 Run closed-loop rollouts across simulated and real web tasks to measure and confirm non-zero native `FINISH` F1 score
