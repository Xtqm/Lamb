"""
Unit tests for Native FINISH Decision Calibration & Decoupled Termination.
Verifies:
- Terminal reward schedule: +2.5 for valid FINISH, -1.5 for premature FINISH, -0.5 for timeouts.
- In-the-loop referee decoupling: episodes do not terminate until agent emits FINISH or steps out.
- Post-hoc scoring: referee validates milestones without overriding action trajectory.
"""

import pytest
from jed_m2.heads import ActionType
from jed_m2.rl import compute_shaped_reward


def test_shaped_reward_valid_finish():
    """Agent emits FINISH when goal is satisfied -> awards completion bonus (+2.5)."""
    rew = compute_shaped_reward(
        goal_satisfied=True,
        terminal_action=ActionType.FINISH,
        milestones_count=2,
        turn_count=3,
        iou_sum=0.8,
        action_type_matches=3,
        argument_matches=1,
    )
    # Expected: +2.5 (task) + 1.0 (milestone) + 0.4 (iou) + 0.6 (type) + 0.3 (arg) - 0.06 (step) = 4.74
    assert rew == pytest.approx(4.74, abs=0.01)
    assert rew > 4.0


def test_shaped_reward_premature_finish():
    """Agent emits FINISH when goal is NOT satisfied -> levies harsh penalty (-1.5)."""
    rew = compute_shaped_reward(
        goal_satisfied=False,
        terminal_action=ActionType.FINISH,
        milestones_count=1,
        turn_count=1,
        iou_sum=0.0,
        action_type_matches=1,
        argument_matches=0,
    )
    # Expected: -1.5 (task penalty) + 0.5 (milestone) + 0.0 + 0.2 (type) + 0.0 - 0.02 (step) = -0.82
    assert rew < 0.0
    assert rew == pytest.approx(-0.82, abs=0.01)


def test_shaped_reward_timeout_without_finish():
    """Agent times out or terminates with non-FINISH action -> levies timeout penalty (-0.5)."""
    # Even if milestones were satisfied, failing to emit FINISH incurs timeout penalty
    rew = compute_shaped_reward(
        goal_satisfied=True,
        terminal_action=ActionType.CLICK,
        milestones_count=3,
        turn_count=8,
        iou_sum=1.5,
        action_type_matches=6,
        argument_matches=1,
    )
    # Expected: -0.5 (timeout) + 1.5 (milestones) + 0.75 (iou) + 1.2 (type) + 0.3 (arg) - 0.16 (step) = 3.09
    assert rew == pytest.approx(3.09, abs=0.01)


def test_shaped_reward_legacy_fallback():
    """When terminal_action is None, maintains backward compatibility."""
    rew_success = compute_shaped_reward(
        goal_satisfied=True,
        milestones_count=2,
        turn_count=3,
        iou_sum=0.8,
        action_type_matches=3,
        argument_matches=1,
        completion_bonus=2.0,
    )
    assert rew_success == pytest.approx(4.24, abs=0.01)


def test_rollout_sampler_autonomous_success_vs_timeout():
    """Verifies TrajectoryRolloutSampler requires both goal satisfaction and FINISH for traj.success."""
    from jed_m2.model import create_jed_m2_nano
    from jed_m2.dataset import SimpleLAMTokenizer
    from jed_m2.rl_pipeline import TrajectoryRolloutSampler, RolloutTurn
    from jed_m2.environment import WebOSEnvironment
    import torch

    model = create_jed_m2_nano(vocab_size=256)
    config = model.config
    config.max_seq_len = 128
    tokenizer = SimpleLAMTokenizer(vocab_size=256)
    sampler = TrajectoryRolloutSampler(
        model=model,
        config=config,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    env = WebOSEnvironment(scenario_id="ecommerce")
    env.reset()
    # Transition env to goal satisfied state
    env.goal_satisfied = True

    # Case A: Trajectory sampled on WebOSEnvironment where goal_satisfied == True
    # If the sampled action wasn't FINISH, autonomous success must be False!
    traj_timeout = sampler.sample_trajectory(env=env, max_turns=1)
    final_act = traj_timeout.turns[-1].action_type if traj_timeout.turns else None
    if final_act != ActionType.FINISH:
        assert traj_timeout.success is False

    # Check collate shaped rewards reward calculation
    # Simulate a group of 2 trajectories: one ending with valid FINISH, one ending with CLICK
    from jed_m2.rl_pipeline import RolloutTrajectory
    traj_valid_finish = RolloutTrajectory(
        scenario_id="test",
        goal="Test goal",
        turns=[
            RolloutTurn(
                turn_id=0,
                obs_text="obs",
                thought_tokens=[10],
                thought_text="finish",
                action_type=ActionType.FINISH,
                action_type_id=ActionType.FINISH.value,
                coordinates=(0.0, 0.0),
                argument="",
                step_tokens=[1, 2],
                step_mask=[True, True],
                reward=1.0,
                done=True,
                gate_prob=0.9,
            )
        ],
        full_token_ids=[1, 2, 3],
        trainable_mask=[True, True, True],
        step_rewards=[1.0],
        cumulative_reward=1.0,
        terminal_reward=1.0,
        success=True,
        num_steps=1,
    )

    traj_timed_out = RolloutTrajectory(
        scenario_id="test",
        goal="Test goal",
        turns=[
            RolloutTurn(
                turn_id=0,
                obs_text="obs",
                thought_tokens=[10],
                thought_text="click",
                action_type=ActionType.CLICK,
                action_type_id=ActionType.CLICK.value,
                coordinates=(0.5, 0.5),
                argument="",
                step_tokens=[1, 2],
                step_mask=[True, True],
                reward=0.2,
                done=False,
                gate_prob=0.9,
            )
        ],
        full_token_ids=[1, 2, 3],
        trainable_mask=[True, True, True],
        step_rewards=[0.2],
        cumulative_reward=0.2,
        terminal_reward=0.0,
        success=False,
        num_steps=1,
    )

    batch = sampler.collate_rollouts([traj_valid_finish, traj_timed_out], reward_mode="shaped")
    # Adv for valid finish must be strictly higher than timed out trajectory
    advantages = batch["advantages"].tolist()
    assert advantages[0] > advantages[1], f"Valid FINISH must have higher advantage: {advantages}"


def test_benchmark_autonomous_metrics_reporting():
    """Verifies TaskEvaluator computes autonomous_success_rate, referee_assisted_rate, and native_finish_f1."""
    from jed_m2.benchmark import TaskEvaluator, StepMetric

    evaluator = TaskEvaluator()

    # Step for ep 1: Autonomous FINISH
    sm1 = evaluator.evaluate_step(
        turn_id=0,
        action_type_pred=ActionType.FINISH.value,
        action_type_gt=ActionType.FINISH.value,
        coord_pred=(0.0, 0.0),
        coord_gt=(0.0, 0.0),
        reward=1.0,
        subgoal_achieved=True,
    )
    ep1 = evaluator.evaluate_episode(
        episode_id="ep1",
        benchmark_name="test",
        task_goal="Search and checkout item",
        step_metrics=[sm1],
        task_success=True,
        optimal_length=1,
        autonomous_success=True,
        referee_assisted=True,
        final_action="FINISH",
    )

    # Step for ep 2: Referee assisted but failed autonomous termination (CLICK instead of FINISH)
    sm2 = evaluator.evaluate_step(
        turn_id=0,
        action_type_pred=ActionType.CLICK.value,
        action_type_gt=ActionType.FINISH.value,
        coord_pred=(0.5, 0.5),
        coord_gt=(0.0, 0.0),
        reward=0.0,
        subgoal_achieved=False,
    )
    ep2 = evaluator.evaluate_episode(
        episode_id="ep2",
        benchmark_name="test",
        task_goal="Book travel hotel",
        step_metrics=[sm2],
        task_success=False,
        optimal_length=1,
        autonomous_success=False,
        referee_assisted=True,
        final_action="CLICK",
    )

    summary = evaluator.aggregate([ep1, ep2], benchmark_name="test")

    # ASR should be 50.0% (1/2), while RAR should be 100.0% (2/2)
    assert summary.autonomous_success_rate == 50.0
    assert summary.referee_assisted_rate == 100.0
    assert summary.task_success_rate == 50.0

    # FINISH F1: TP=1, FP=0, FN=1 -> Prec = 1.0, Rec = 0.5, F1 = 2/3 = 0.667
    assert summary.native_finish_f1 == pytest.approx(0.667, abs=0.01)
    assert summary.native_finish_precision == pytest.approx(1.0, abs=0.01)
    assert summary.native_finish_recall == pytest.approx(0.5, abs=0.01)

    # Verify ASCII table format
    table_str = evaluator.format_ascii_table(summary)
    assert "ASR (%)" in table_str
    assert "RAR (%)" in table_str
    assert "Fin F1" in table_str
    assert "50.0%" in table_str
    assert "100.0%" in table_str

