"""
Unit tests for JED-M2 Outcome-Supervised Long-Horizon GRPO and Curriculum Pacing.
"""

import pytest
import torch
from jed_m2.config import JEDM2Config
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model
from jed_m2.rl import CurriculumScheduler, GRPOTrainer, compute_shaped_reward
from jed_m2.rl_pipeline import TrajectoryRolloutSampler
from jed_m2.scenario_generator import DynamicBookingEnvironment, DynamicScenarioGenerator


def test_outcome_supervised_reward_formulation():
    """Verify that outcome-supervised rewards follow exact +1.0 success, -0.5 loop, and milestone rules."""
    # 1. Successful trajectory with 2 milestones and 4 turns
    r_success = compute_shaped_reward(
        goal_satisfied=True,
        milestones_count=2,
        turn_count=4,
        terminal_action=ActionType.FINISH,
        outcome_supervised=True,
    )
    # Expected: 1.0 (success) + 0.15*2 (milestones) - 0.01*4 (step penalty) = 1.26
    assert r_success == pytest.approx(1.26, abs=1e-3)

    # 2. Failed trajectory with loop
    r_loop = compute_shaped_reward(
        goal_satisfied=False,
        milestones_count=1,
        turn_count=6,
        repeated_action_count=2,
        terminal_action=ActionType.CLICK,
        outcome_supervised=True,
    )
    # Expected: -0.5 (loop) + 0.15*1 (milestone) - 0.01*6 (step penalty) = -0.41
    assert r_loop == pytest.approx(-0.41, abs=1e-3)

    # 3. Premature FINISH on incomplete state
    r_premature = compute_shaped_reward(
        goal_satisfied=False,
        milestones_count=0,
        turn_count=2,
        terminal_action=ActionType.FINISH,
        outcome_supervised=True,
    )
    # Expected: -0.5 (premature finish) - 0.01*2 = -0.52
    assert r_premature == pytest.approx(-0.52, abs=1e-3)


def test_curriculum_scheduler_rolling_threshold_and_safety_cap():
    """Verify dynamic threshold (>=80% over 50 episodes) and hard safety fallback cap (300 episodes)."""
    scheduler = CurriculumScheduler(
        initial_level=1,
        success_threshold=0.80,
        window_size=50,
        min_episodes_per_level=10,
        max_episodes_per_level=300,
    )

    assert scheduler.current_level == 1

    # Simulate 20 successful outcomes (100% success rate >= 80%)
    advanced = scheduler.record_outcomes([True] * 20)
    assert advanced
    assert scheduler.current_level == 2

    # In level 2, simulate low success rate (20% success) for 299 episodes -> should NOT advance yet
    for _ in range(29):
        scheduler.record_outcomes([False, False, False, False, True] * 2)  # 290 episodes
    scheduler.record_outcomes([False] * 9)  # Total 299 episodes
    assert scheduler.current_level == 2

    # The 300th episode triggers the hard safety fallback cap!
    advanced_fallback = scheduler.record_outcomes([False])
    assert advanced_fallback
    assert scheduler.current_level == 3


def test_grouped_outcome_rollouts_and_grpo_loss():
    """Verify group rollout collection (G=4) with outcome rewards and finite GRPO loss."""
    config = JEDM2Config(hidden_dim=64, num_layers=2, max_seq_len=128)
    model = JEDM2Model(config).to(torch.device("cpu"), dtype=torch.float32)
    sampler = TrajectoryRolloutSampler(model=model, config=config, device="cpu", dtype=torch.float32)
    generator = DynamicScenarioGenerator(config=config, device="cpu", dtype=torch.float32)

    # Sample group of 4 rollouts with outcome supervision and obstacle exposure
    group = sampler.sample_group_rollouts(
        scenario_id="booking",
        group_size=4,
        temperature=0.7,
        max_turns=3,
        generator=generator,
        seed=101,
        outcome_supervised=True,
        obstacle_prob=0.5,
    )

    assert len(group) == 4
    for traj in group:
        assert traj.num_steps > 0
        assert len(traj.turns) > 0

    # Collate with reward_mode="outcome"
    batch = sampler.collate_rollouts(group, reward_mode="outcome")
    assert batch["rewards"].shape == (4,)
    assert batch["advantages"].shape == (4,)
    assert not torch.isnan(batch["advantages"]).any()

    # Verify GRPOTrainer compute_loss
    trainer = GRPOTrainer(model=model, config=None)
    loss, metrics = trainer.compute_loss(
        input_ids=batch["input_ids"],
        targets=batch["targets"],
        mask=batch["mask"],
        old_token_logps=batch["old_token_logps"],
        rewards=batch["rewards"],
        old_coord_logps=batch["old_coord_logps"],
        target_coords=batch["target_coords"],
        advantages=batch["advantages"],
    )

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert isinstance(loss.item(), float)
    assert "loss_grpo" in metrics
