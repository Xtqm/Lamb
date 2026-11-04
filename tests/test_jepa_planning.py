"""
Unit and integration tests for Active JEPA Latent Lookahead Planning.
Validates:
- Spatially-conditioned latent transition predictions in LatentPredictor and JEPAWorldModel.
- Differentiated latent transitions across distinct screen coordinates.
- Backward compatibility when coordinates are omitted.
- JEDM2Model and JEDM2Agent predict_next_latent_state coordination.
"""

import pytest
import torch
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.heads import ActionType
from jed_m2.jepa import JEPAWorldModel, LatentPredictor
from jed_m2.model import JEDM2Model


@pytest.fixture
def test_config():
    return JEDM2Config(
        hidden_dim=128,
        vocab_size=256,
        num_layers=2,
        num_heads=4,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
        num_experts=4,
        top_k=2,
        moe_intermediate_dim=256,
        jepa_latent_dim=64,
        jepa_action_dim=32,
    )


def test_latent_predictor_spatial_conditioning(test_config):
    predictor = LatentPredictor(test_config)
    b_sz = 2
    z_t = torch.randn(b_sz, test_config.jepa_latent_dim)
    action = torch.tensor([ActionType.CLICK.value, ActionType.TYPE.value], dtype=torch.long)
    coords = torch.tensor([[0.2, 0.3], [0.7, 0.8]], dtype=torch.float32)

    # Prediction with coordinates
    z_hat_coords = predictor(z_t, action, coords=coords)
    assert z_hat_coords.shape == (b_sz, test_config.jepa_latent_dim)
    assert not torch.isnan(z_hat_coords).any()

    # Prediction without coordinates (backward compatibility)
    z_hat_no_coords = predictor(z_t, action)
    assert z_hat_no_coords.shape == (b_sz, test_config.jepa_latent_dim)
    assert not torch.isnan(z_hat_no_coords).any()

    # Predictions with coordinates should differ from prediction without coordinates
    assert not torch.allclose(z_hat_coords, z_hat_no_coords, atol=1e-4)


def test_spatial_differentiation_across_coordinates(test_config):
    predictor = LatentPredictor(test_config)
    b_sz = 1
    z_t = torch.randn(b_sz, test_config.jepa_latent_dim)
    action = torch.tensor([ActionType.CLICK.value], dtype=torch.long)

    # Two distinct screen locations (e.g. top-left vs bottom-right)
    coords_tl = torch.tensor([[0.05, 0.05]], dtype=torch.float32)
    coords_br = torch.tensor([[0.95, 0.95]], dtype=torch.float32)

    z_hat_tl = predictor(z_t, action, coords=coords_tl)
    z_hat_br = predictor(z_t, action, coords=coords_br)

    assert z_hat_tl.shape == (b_sz, test_config.jepa_latent_dim)
    assert z_hat_br.shape == (b_sz, test_config.jepa_latent_dim)

    # Distinct spatial inputs must produce differentiated predicted latent states
    diff = (z_hat_tl - z_hat_br).abs().sum().item()
    assert diff > 1e-4, f"Predicted states must differentiate across coordinates, got diff={diff}"


def test_jepa_world_model_predict_with_coords(test_config):
    jepa = JEPAWorldModel(test_config)
    b_sz = 3
    z_t = torch.randn(b_sz, test_config.jepa_latent_dim)
    action = torch.tensor([0, 1, 2], dtype=torch.long)
    coords = torch.rand(b_sz, 2)

    z_hat = jepa.predict(z_t, action, coords=coords)
    assert z_hat.shape == (b_sz, test_config.jepa_latent_dim)
    assert not torch.isnan(z_hat).any()


def test_model_predict_next_latent_state(test_config):
    model = JEDM2Model(test_config)
    z_t = torch.randn(1, test_config.jepa_latent_dim)

    # Test with int action and tuple coords
    coords_tensor = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    z_hat = model.predict_next_latent_state(z_t, ActionType.CLICK.value, coords=coords_tensor)
    assert z_hat.shape == (1, test_config.jepa_latent_dim)
    assert not torch.isnan(z_hat).any()


def test_evaluate_grounded_candidates(test_config):
    from jed_m2.agent import JEDM2Agent

    model = JEDM2Model(test_config)
    agent = JEDM2Agent(model, config=test_config)

    z_t = torch.randn(1, test_config.jepa_latent_dim)
    z_goal = torch.randn(1, test_config.jepa_latent_dim)
    action_logits = torch.randn(len(ActionType))
    raw_coords = (0.5, 0.5)

    candidate_elements = [
        {"bbox": [0.1, 0.1, 0.3, 0.3]},
        {"bbox": [0.6, 0.6, 0.8, 0.8]},
    ]

    sel_act_id, sel_coords, metadata = agent.evaluate_grounded_candidates(
        z_t=z_t,
        z_goal=z_goal,
        action_logits=action_logits,
        raw_coords=raw_coords,
        candidate_elements=candidate_elements,
    )

    assert isinstance(sel_act_id, int)
    assert isinstance(sel_coords, tuple)
    assert len(sel_coords) == 2
    assert 0.0 <= sel_coords[0] <= 1.0 and 0.0 <= sel_coords[1] <= 1.0
    assert metadata["total_candidates"] > 0
    assert "candidates" in metadata


def test_action_value_pruning_dead_ends(test_config):
    from jed_m2.agent import JEDM2Agent

    model = JEDM2Model(test_config)
    agent = JEDM2Agent(model, config=test_config)

    z_t = torch.randn(1, test_config.jepa_latent_dim)
    # Target goal very close to z_t
    z_goal = z_t.clone()

    action_logits = torch.randn(len(ActionType))
    raw_coords = (0.5, 0.5)

    # With a strict negative drift threshold, actions that move away from goal are pruned
    sel_act_id, sel_coords, metadata = agent.evaluate_grounded_candidates(
        z_t=z_t,
        z_goal=z_goal,
        action_logits=action_logits,
        raw_coords=raw_coords,
        drift_threshold=0.0001,  # Strict drift threshold
    )

    # Some candidates must be pruned due to drift
    pruned_count = sum(1 for c in metadata["candidates"] if c["pruned"])
    assert pruned_count > 0, "Expected at least one candidate to be pruned by goal drift"


def test_loop_elimination_against_history(test_config):
    from jed_m2.agent import JEDM2Agent

    model = JEDM2Model(test_config)
    agent = JEDM2Agent(model, config=test_config)

    z_t = torch.randn(1, test_config.jepa_latent_dim)
    z_goal = torch.randn(1, test_config.jepa_latent_dim)

    # Predict where ActionType.CLICK at (0.2, 0.2) goes:
    coords_tensor = torch.tensor([[0.2, 0.2]], dtype=torch.float32)
    z_next_click = agent.predict_next_latent_state(z_t, ActionType.CLICK.value, coords=coords_tensor)

    # Put that exact state into agent's latent history (simulating a recently visited state)
    agent.latent_history = [z_next_click.clone()]

    action_logits = torch.zeros(len(ActionType))
    action_logits[ActionType.CLICK.value] = 5.0  # High policy logit for CLICK

    sel_act_id, sel_coords, metadata = agent.evaluate_grounded_candidates(
        z_t=z_t,
        z_goal=z_goal,
        action_logits=action_logits,
        raw_coords=(0.2, 0.2),
        loop_threshold=0.1,  # Prune if within 0.1 cosine dist to history
    )

    # Candidate CLICK at (0.2, 0.2) must be flagged for loop pruning
    click_cand = [
        c for c in metadata["candidates"]
        if c["action_id"] == ActionType.CLICK.value and abs(c["coords"][0] - 0.2) < 1e-4
    ]
    assert len(click_cand) > 0
    assert click_cand[0]["pruned"], "Expected state matching history to be pruned for loop elimination"
    assert "loop" in click_cand[0]["prune_reason"]


def test_agent_act_with_jepa_mpc_planning(test_config):
    from jed_m2.agent import JEDM2Agent, AgentStepOutput

    model = JEDM2Model(test_config)
    agent = JEDM2Agent(model, config=test_config, use_jepa_planning=True)

    candidate_elements = [
        {"bbox": [0.15, 0.20, 0.35, 0.40], "role": "button", "text": "Search"},
        {"bbox": [0.60, 0.70, 0.85, 0.90], "role": "link", "text": "Cancel"},
    ]

    # Step 1 with active MPC planning
    res1 = agent.act(
        obs_text="Search box input and buttons",
        candidate_elements=candidate_elements,
        use_jepa_planning=True,
    )

    assert isinstance(res1, AgentStepOutput)
    assert res1.action_name in [a.name for a in ActionType]
    assert len(res1.coordinates) == 2
    assert 0.0 <= res1.coordinates[0] <= 1.0 and 0.0 <= res1.coordinates[1] <= 1.0
    assert res1.jepa_planning_scores is not None
    assert len(agent.latent_history) == 1

    # Step 2: verify multi-turn rollout accumulates latent history
    res2 = agent.act(
        obs_text="Results page loaded",
        candidate_elements=candidate_elements,
        use_jepa_planning=True,
    )
    assert isinstance(res2, AgentStepOutput)
    assert len(agent.latent_history) == 2


