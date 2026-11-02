# Design: Native FINISH Action Calibration & Referee Decoupling

## Context

In JED-M2 commit `04f11ae`, `CompletionReferee` was introduced to prevent agent rollouts from looping indefinitely when browser state machines transitioned through milestones. However, by having `evaluate_termination()` trigger in-the-loop termination whenever environment milestones were met, the agent was shielded from ever needing to predict `ActionType.FINISH`.

As documented in the scientific audit, this produced an empirical policy state where the model's native `FINISH` action F1 is **0.000**, with 100% of successful online completions artificially triggered by the external referee rather than the agent's policy.

See `proposal.md` for motivation and scope boundaries.

## Goals / Non-Goals

**Goals:**
- Decouple `CompletionReferee` from the active interactive execution loop in `TrajectoryRolloutSampler` and `JEDM2Agent`, transforming it into an external post-hoc evaluation oracle.
- Restrict episode termination strictly to autonomous agent decisions (`action_type == ActionType.FINISH`) or execution budget limits (`current_step >= max_steps` or loop guardrails).
- Implement an asymmetric terminal reward schedule in `compute_shaped_reward()` that strongly rewards valid autonomous termination (+2.5) while penalizing premature `FINISH` emissions (-1.5) and non-terminating timeouts (-0.5).
- Ensure class-balanced action loss supervision during pretraining/fine-tuning so the model learns positive representation for `ActionType.FINISH`.
- Provide formal benchmark metrics in `benchmark.py` reporting `autonomous_success_rate`, `referee_assisted_rate`, and `native_finish_f1` (Precision, Recall, F1).

**Non-Goals:**
- Removing the referee entirely: `CompletionReferee` remains a valuable ground-truth programmatic milestone validator, but operates strictly post-hoc.
- Changing coordinate regression heads or tokenizer vocabulary budgets (handled in Phase 1).

## Decisions

### Decision 1: Post-Hoc Oracle Decoupling
- **Choice:** Remove `evaluate_termination()` from the turn loop in `TrajectoryRolloutSampler.collect_episode()` and `sample_trajectory()`. The episode continues stepping until either:
  1. The agent predicts `ActionType.FINISH`
  2. The agent exceeds `max_turns`
  3. The agent repeats the identical action 3 consecutive times (deadlock safety)
- **Rationale:** Forces the policy to take full responsibility for detecting when all sub-goals have been satisfied.
- **Alternatives Considered:**
  - *Keep referee active with a probabilistic dropout*: Rejected because stochastic overrides still confuse policy gradient credit assignment.

### Decision 2: Asymmetric Terminal Reward Schedule
- **Choice:** In `jed_m2/rl.py`, update `compute_shaped_reward()` to accept `terminal_action: Optional[ActionType]`:
  - `FINISH` with `goal_satisfied == True`: $+2.5$ terminal bonus + $+1.0$ milestone bonus.
  - `FINISH` with `goal_satisfied == False`: $-1.5$ premature termination penalty.
  - Timeout (`step >= max_steps` without `FINISH`): $0.0$ completion bonus + $-0.5$ truncation penalty.
- **Rationale:** Prevents "lazy policy collapse" where the model emits `FINISH` on step 0 to minimize step penalties, while compelling the model to close the trajectory when done.
- **Alternatives Considered:**
  - *Zero penalty for premature finish*: Causes policy collapse where agent terminates immediately to avoid step costs.
  - *Sparse binary 0/1 reward*: Slow convergence in sparse web tasks; dense sub-action matching (IoU + type) must remain active.

### Decision 3: Terminal Class Weighting in Supervised Loss
- **Choice:** In `JEDM2LossConfig`, provide `finish_action_weight: float = 2.0` applied to cross-entropy loss when target action is `ActionType.FINISH`.
- **Rationale:** `FINISH` appears only once per multi-turn trajectory (representing ~5% of action tokens). A $2.0\times$ weight prevents the action classification head from ignoring the terminal token in favor of high-frequency `CLICK` and `TYPE`.

### Decision 4: Transparent Metric Reporting in Benchmark Harness
- **Choice:** In `jed_m2/benchmark.py`, calculate and report:
  - `Autonomous Success Rate (ASR %)`: Trajectories ending with agent-predicted `FINISH` where `goal_satisfied == True`.
  - `Referee-Assisted Rate (RAR %)`: Trajectories reaching goal state regardless of terminal primitive.
  - `Native FINISH F1`: Micro and macro F1 score for `ActionType.FINISH`.
- **Rationale:** Provides full transparency for academic publication, proving the LAM is genuinely autonomous.

## Risks / Trade-offs

- **[Risk] Initial Success Rate Drop During Exploration:**
  - *Mitigation:* Decoupling the referee means an agent that solves all milestones but fails to emit `FINISH` will time out with zero completion reward. DPO preference pairing (`generate_preference_pairs`) explicitly pairs trajectories that emitted valid `FINISH` against those that timed out, accelerating policy adaptation.
- **[Risk] Early Premature FINISH Exploits:**
  - *Mitigation:* The $-1.5$ premature termination penalty ensures that taking any valid intermediate action yields higher return than an early premature exit.

## Migration Plan

1. Update `compute_shaped_reward()` in `jed_m2/rl.py` to support `terminal_action` and premature penalty.
2. Update `TrajectoryRolloutSampler` in `jed_m2/rl_pipeline.py` to terminate on autonomous `FINISH` and call referee post-hoc.
3. Update `JEDM2Agent` in `jed_m2/agent.py` to handle `FINISH` as terminal.
4. Update `JEDM2Loss` in `jed_m2/losses.py` with `finish_action_weight`.
5. Update `jed_m2/benchmark.py` to compute and report `native_finish_f1` and `autonomous_success_rate`.
6. Add unit tests in `tests/test_native_finish.py` and run full regression suite.
