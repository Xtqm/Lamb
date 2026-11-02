# Proposal: Native FINISH Action Calibration & Completion Referee Decoupling

## Why

In JED-M2's current interactive rollout and evaluation pipeline, task completion relies on an external programmatic referee (`CompletionReferee`) that inspects environment internal state machines and terminates episodes with a 100% terminal reward when intermediate milestones are reached. Because the referee intercepts the execution loop, the agent never learns when a task is complete, yielding an empirical native `FINISH` action F1 score of **0.000**. 

For JED-M2 to operate as a truly autonomous Large Action Model and qualify for peer-reviewed publication, episode termination must be an autonomous agent decision. The external referee must be decoupled from the live interactive decision loop and converted into a post-hoc evaluation oracle, while reinforcement learning reward functions and training datasets are calibrated to supervise and reward autonomous `ActionType.FINISH` emissions.

## What Changes

- **Referee Decoupling**: Decouple `CompletionReferee.evaluate_termination()` from live decision loops in `TrajectoryRolloutSampler` and `JEDM2Agent`, restricting episode termination strictly to the agent emitting `ActionType.FINISH` or reaching the maximum step budget.
- **Post-Hoc Oracle Evaluation**: Reposition `CompletionReferee` exclusively as an external evaluation and reward oracle called after an episode concludes to score whether task criteria were met.
- **Calibrated Terminal Reward Function**: Update `compute_shaped_reward()` in `jed_m2/rl.py` to award a prominent terminal bonus (+2.5) when the agent emits `FINISH` with all task goals satisfied, and levy a negative penalty (-1.5) when the agent emits `FINISH` prematurely on an incomplete state.
- **Explicit Terminal Step Supervision**: Ensure all trajectory generators (`SyntheticTrajectoryGenerator`, real trajectory processing) explicitly emit supervised `ActionType.FINISH` terminal steps so imitation pre-training and RL policy gradients directly optimize the terminal action logits.
- **Autonomous Termination Benchmark Metrics**: Update `evaluation-benchmark-harness` to track and report Native `FINISH` Precision, Recall, and F1 score, alongside Autonomous Task Success Rate (success requiring model-generated `FINISH`) versus Referee-assisted Success Rate.

## Capabilities

### New Capabilities
<!-- None -->

### Modified Capabilities
- `rl-rollout-pipeline`: Decouple in-the-loop completion referee from `TrajectoryRolloutSampler`, enforce autonomous agent `FINISH` termination, and calibrate terminal reward shaping with premature finish penalties.
- `interactive-agent-runtime`: Update `JEDM2Agent` execution loop to handle `ActionType.FINISH` as a self-directed terminal action without depending on scaffold/referee early termination.
- `evaluation-benchmark-harness`: Add formal tracking and reporting for Native `FINISH` F1 score, autonomous success rate, and referee-decoupled evaluation protocols.

## Impact

- **Affected Code**: `jed_m2/scaffold.py`, `jed_m2/rl_pipeline.py`, `jed_m2/rl.py`, `jed_m2/agent.py`, `jed_m2/benchmark.py`, `jed_m2/dataset.py`.
- **APIs**: `TrajectoryRolloutSampler.collect_episode()` and `sample_trajectory()` stop terminating prematurely via referee; `evaluate_termination()` becomes an oracle evaluation helper.
- **Dependencies**: No external dependencies added (pure PyTorch and Python standard library).
- **Verification**: Complete test suite pass (`pytest tests/`), unit tests for premature finish penalty, and verification of non-zero native `FINISH` F1 during rollout sampling.
