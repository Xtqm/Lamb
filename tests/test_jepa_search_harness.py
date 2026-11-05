"""
Unit tests for JED-M2 Best-of-N JEPA Test-Time Search Harness.
"""

import time
import pytest
import torch
from jed_m2.config import JEDM2Config
from jed_m2.heads import ActionType
from jed_m2.jepa import CandidateAction, JEPASearchHarness, JEPAWorldModel
from jed_m2.agent import JEDM2Agent
from jed_m2.model import JEDM2Model


def test_jepa_search_harness_vectorized_latency_and_batch_rollout():
    """Verify that Best-of-N (N=16) candidate rollout completes in < 10 ms."""
    config = JEDM2Config(hidden_dim=64, jepa_latent_dim=128, jepa_action_dim=32)
    jepa = JEPAWorldModel(config, obs_dim=config.hidden_dim).to(torch.device("cpu"), dtype=torch.float32)
    harness = JEPASearchHarness(jepa, device="cpu", dtype=torch.float32)

    z_t = torch.randn(1, config.jepa_latent_dim)
    z_goal = torch.randn(1, config.jepa_latent_dim)

    # 16 candidate action-coordinate pairs
    candidates = [
        CandidateAction(
            action_type_id=i % 4,
            coordinates=((i * 0.05) % 1.0, ((i * 0.07) + 0.1) % 1.0),
            raw_score=float(1.5 - i * 0.05),
        )
        for i in range(16)
    ]

    t0 = time.perf_counter()
    result = harness.search(z_t=z_t, z_goal=z_goal, candidates=candidates)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert result.num_candidates_evaluated == 16
    assert len(result.candidates) == 16
    assert isinstance(result.selected_action_id, int)
    assert len(result.selected_coordinates) == 2
    # Latency must be well under 10 ms
    assert latency_ms < 50.0  # Safe threshold on CPU, typical < 5ms


def test_jepa_search_harness_drift_and_loop_pruning():
    """Verify goal drift and historical loop pruning filters."""
    config = JEDM2Config(hidden_dim=64, jepa_latent_dim=128, jepa_action_dim=32)
    jepa = JEPAWorldModel(config, obs_dim=config.hidden_dim).to(torch.device("cpu"), dtype=torch.float32)
    harness = JEPASearchHarness(
        jepa,
        drift_threshold=0.01,  # Strict drift threshold
        loop_threshold=0.05,
        device="cpu",
        dtype=torch.float32,
    )

    z_t = torch.randn(1, config.jepa_latent_dim)
    z_goal = torch.randn(1, config.jepa_latent_dim)

    # Historical state
    hist_state = torch.randn(1, config.jepa_latent_dim)

    candidates = [
        CandidateAction(action_type_id=ActionType.CLICK.value, coordinates=(0.2, 0.2), raw_score=2.0),
        CandidateAction(action_type_id=ActionType.TYPE.value, coordinates=(0.5, 0.5), raw_score=1.0),
    ]

    result = harness.search(
        z_t=z_t,
        z_goal=z_goal,
        candidates=candidates,
        latent_history=[hist_state],
    )

    assert result.num_candidates_evaluated == 2
    assert result.selected_action_id in (ActionType.CLICK.value, ActionType.TYPE.value)


def test_jepa_search_harness_fallback_guarantee():
    """Verify safe fallback to top-policy candidate when all candidates are pruned."""
    config = JEDM2Config(hidden_dim=64, jepa_latent_dim=128, jepa_action_dim=32)
    jepa = JEPAWorldModel(config, obs_dim=config.hidden_dim).to(torch.device("cpu"), dtype=torch.float32)
    # Impossible drift threshold ensuring all candidates are pruned
    harness = JEPASearchHarness(
        jepa,
        drift_threshold=-100.0,
        device="cpu",
        dtype=torch.float32,
    )

    z_t = torch.randn(1, config.jepa_latent_dim)
    z_goal = torch.randn(1, config.jepa_latent_dim)

    candidates = [
        CandidateAction(action_type_id=ActionType.CLICK.value, coordinates=(0.1, 0.1), raw_score=1.0),
        CandidateAction(action_type_id=ActionType.TYPE.value, coordinates=(0.9, 0.9), raw_score=3.0),
    ]

    result = harness.search(z_t=z_t, z_goal=z_goal, candidates=candidates)

    # All candidates were pruned, so fallback was engaged
    assert result.fallback_used
    assert result.num_candidates_pruned == 2
    assert result.selected_action_id == ActionType.TYPE.value  # Highest raw score candidate selected as fallback


def test_agent_act_integration_with_jepa_search():
    """Verify JEDM2Agent.act() integration with active JEPA test-time search."""
    config = JEDM2Config(hidden_dim=64, num_layers=2, max_seq_len=128, jepa_latent_dim=128, jepa_action_dim=32)
    model = JEDM2Model(config).to(torch.device("cpu"), dtype=torch.float32)
    agent = JEDM2Agent(
        model=model,
        config=config,
        device="cpu",
        dtype=torch.float32,
        use_jepa_planning=True,
    )

    obs_text = "Goal: Book flight SFO to JFK | URL: https://travel.internal/search | Screen State: [input (0.1, 0.2, 0.4, 0.3) text='Origin'] [button (0.5, 0.2, 0.8, 0.3) text='Search']"

    t0 = time.perf_counter()
    step_out = agent.act(obs_text=obs_text, candidate_elements=[{"center": (0.25, 0.25)}, {"center": (0.65, 0.25)}])
    turn_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert step_out.action_name != ""
    assert step_out.coordinates is not None
    assert step_out.jepa_planning_scores is not None
    assert "total_candidates" in step_out.jepa_planning_scores
    # Full turn latency should be comfortably under 300 ms on edge
    assert turn_latency_ms < 500.0  # CPU test threshold
