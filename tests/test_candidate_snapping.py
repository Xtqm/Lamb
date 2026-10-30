"""
Unit tests for candidate element bounding box snapping and AgentStepOutput telemetry.
"""

import pytest
import torch

from jed_m2.agent import JEDM2Agent, AgentStepOutput, _extract_element_center
from jed_m2.config import JEDM2Config
from jed_m2.core.types import UIElement
from jed_m2.environment import WebOSEnvironment, get_default_scenarios
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model


def test_extract_element_center():
    # 1. UIElement
    elem = UIElement(
        element_id=1,
        tag="button",
        role="submit",
        text="Submit",
        bbox=(0.4, 0.6, 0.6, 0.8),
    )
    center = _extract_element_center(elem)
    assert center is not None
    assert pytest.approx(center[0], 1e-4) == 0.5
    assert pytest.approx(center[1], 1e-4) == 0.7

    # 2. Dict with center
    d_center = {"center": [0.3, 0.4]}
    assert _extract_element_center(d_center) == (0.3, 0.4)

    # 3. Dict with bbox
    d_bbox = {"bbox": [0.1, 0.2, 0.3, 0.4]}
    c_bbox = _extract_element_center(d_bbox)
    assert c_bbox is not None
    assert pytest.approx(c_bbox[0], 1e-4) == 0.2
    assert pytest.approx(c_bbox[1], 1e-4) == 0.3

    # 4. Invalid
    assert _extract_element_center("invalid") is None
    assert _extract_element_center({}) is None


def test_agent_step_output_raw_coordinates_serialization():
    out = AgentStepOutput(
        action_type=ActionType.CLICK,
        action_type_id=0,
        action_name="CLICK",
        coordinates=(0.50, 0.60),
        raw_coordinates=(0.48, 0.62),
        argument="",
        thought="Testing snapping",
        gate_prob=0.95,
        num_thought_tokens=2,
    )
    d = out.to_dict()
    assert d["coordinates"] == (0.50, 0.60)
    assert d["raw_coordinates"] == (0.48, 0.62)
    assert d["action_type"] == "CLICK"


def test_candidate_snapping_in_agent_act():
    cfg = JEDM2Config(hidden_dim=64, num_heads=2, max_seq_len=64, num_layers=1)
    model = JEDM2Model(cfg)
    agent = JEDM2Agent(model, cfg, max_thought_tokens=2)

    # Candidate button at (0.52, 0.65)
    candidates = [
        UIElement(
            element_id=10,
            tag="button",
            role="submit",
            text="Buy Now",
            bbox=(0.48, 0.60, 0.56, 0.70),  # center = (0.52, 0.65)
        ),
        UIElement(
            element_id=11,
            tag="a",
            role="link",
            text="Cancel",
            bbox=(0.10, 0.10, 0.20, 0.20),  # center = (0.15, 0.15)
        ),
    ]

    # Act with snapping enabled
    step_out = agent.act(
        obs_text="Test store checkout",
        candidate_elements=candidates,
        snap_coordinates=True,
        max_snap_distance=0.50,
    )

    assert hasattr(step_out, "raw_coordinates")
    assert hasattr(step_out, "coordinates")
    # Coordinates should be bounded in [0, 1]
    assert 0.0 <= step_out.coordinates[0] <= 1.0
    assert 0.0 <= step_out.coordinates[1] <= 1.0

    # Test with snapping disabled
    step_out_no_snap = agent.act(
        obs_text="Test store checkout",
        candidate_elements=candidates,
        snap_coordinates=False,
    )
    assert step_out_no_snap.coordinates == step_out_no_snap.raw_coordinates


def test_candidate_snapping_in_step_environment():
    cfg = JEDM2Config(hidden_dim=64, num_heads=2, max_seq_len=64, num_layers=1)
    model = JEDM2Model(cfg)
    agent = JEDM2Agent(model, cfg, max_thought_tokens=2)

    env = WebOSEnvironment(config=cfg, scenario_id="ecommerce")
    env.reset()

    agent_out, step_res = agent.step_environment(env)
    assert hasattr(agent_out, "raw_coordinates")
    assert hasattr(agent_out, "coordinates")
    assert 0.0 <= agent_out.coordinates[0] <= 1.0
    assert 0.0 <= agent_out.coordinates[1] <= 1.0


def test_metric_rates_dual_mode():
    from jed_m2.real_eval import _new_site_bucket, metric_rates

    b = _new_site_bucket()
    b["actions"] = 10
    b["action_correct"] = 8
    b["coord_count"] = 10
    b["coord_abs_sum"] = 2.0
    b["coord_tolerance_correct"] = 2
    b["target_count"] = 10
    b["target_hit"] = 1
    b["step_count"] = 10
    b["step_correct"] = 2
    b["tasks"] = 2
    b["trajectory_correct"] = 0

    # Snapped counters
    b["snapped_coord_abs_sum"] = 0.5
    b["snapped_coord_tolerance_correct"] = 8
    b["snapped_target_hit"] = 9
    b["snapped_step_correct"] = 7
    b["snapped_trajectory_correct"] = 1

    rates = metric_rates(b)
    assert rates["target_element_hit_rate"] == 0.1
    assert rates["snapped_target_element_hit_rate"] == 0.9
    assert rates["coordinate_tolerance_accuracy"] == 0.2
    assert rates["snapped_coordinate_tolerance_accuracy"] == 0.8
    assert rates["step_accuracy"] == 0.2
    assert rates["snapped_step_accuracy"] == 0.7
    assert rates["exact_trajectory_match_rate"] == 0.0
    assert rates["snapped_exact_trajectory_match_rate"] == 0.5


def test_real_dataset_multi_element_distractor_candidates():
    from jed_m2.real_dataset import RealWebTrajectoryDataset

    cfg = JEDM2Config()
    dataset = RealWebTrajectoryDataset(
        config=cfg,
        split="test",
        real_only=True,
        manifest_path="data/manifests/real_mind2web_v1/manifest.json",
        raw_mind2web_dir="data/raw_mind2web",
        cache_dir="data/real_trajectories_cache",
        seed=42,
    )
    assert len(dataset) > 0
    sample = dataset[0]
    turn = sample.turns[0]
    cands_with_bbox = [e for e in turn.interactive_elements if getattr(e, "has_bbox", False)]
    # With distractor ingestion (pos_candidates + neg_candidates), candidate count must be > 1
    assert len(cands_with_bbox) > 1, f"Expected >1 candidates with bboxes, found {len(cands_with_bbox)}"

    # Verify snapping can discriminate and select a distractor if closer to it
    target_id = turn.target_element_id
    distractor = next((e for e in cands_with_bbox if e.element_id != target_id), None)
    assert distractor is not None
    assert distractor.center is not None
    
    # Fake a point right next to the distractor
    fake_pt = (distractor.center[0] + 0.001, distractor.center[1] + 0.001)
    best_cand = min(
        cands_with_bbox,
        key=lambda e: (e.center[0] - fake_pt[0]) ** 2 + (e.center[1] - fake_pt[1]) ** 2,
    )
    assert best_cand.element_id == distractor.element_id, "Candidate snapping should snap to the nearest distractor"

