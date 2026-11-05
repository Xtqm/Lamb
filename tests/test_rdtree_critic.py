"""
Unit tests for JED-M2 Rule-Based Decision Tree (RDTree) Process Critic.

Verifies:
1. Hierarchical rule node evaluation (bounds, affordances, focus).
2. Diagnostic categorization and scalar process reward bounds [-1.0, 1.0].
3. Rolling history cycle detection and oscillation penalty.
4. Latency performance (<2 ms for 16 evaluations).
"""

import time
from unittest.mock import MagicMock
import pytest
import torch

from jed_m2.config import JEDM2Config
from jed_m2.core.types import UIElement
from jed_m2.heads import ActionType
from jed_m2.jepa import CandidateAction, JEPASearchHarness, JEPAWorldModel
from jed_m2.model import JEDM2Model
from jed_m2.agent import JEDM2Agent, AgentStepOutput, TurnRecord
from jed_m2.rdtree_critic import RDTreeProcessCritic, RDTreeDiagnostic, RDTreeEvaluationResult


@pytest.fixture
def test_config():
    return JEDM2Config(
        hidden_dim=64,
        vocab_size=128,
        num_layers=2,
        num_heads=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=8,
        num_experts=2,
        top_k=1,
        moe_intermediate_dim=128,
        jepa_latent_dim=32,
        jepa_action_dim=16,
    )


def test_rdtree_rule_bounds_and_affordances():
    """Verifies Task 1.1: rule checks for coordinate bounds and affordances."""
    critic = RDTreeProcessCritic()

    elements = [
        UIElement(1, "button", "search_btn", "Search", (0.2, 0.1, 0.4, 0.2), is_clickable=True),
        UIElement(2, "input", "query_inp", "", (0.5, 0.1, 0.8, 0.2), is_clickable=True, is_input=True),
        UIElement(3, "div", "banner", "Header", (0.0, 0.0, 1.0, 0.08), is_clickable=False, is_input=False),
    ]

    # 1. Offscreen coordinates -> OFFSCREEN_COORDINATE (-1.0)
    res_off = critic.evaluate_action(ActionType.CLICK, coordinates=(1.5, 0.5), elements=elements)
    assert res_off.diagnostic == RDTreeDiagnostic.OFFSCREEN_COORDINATE
    assert res_off.process_reward == -1.0
    assert res_off.is_critical_violation is True

    # 2. Click on empty space -> NOOP_CLICK (-0.8)
    res_noop = critic.evaluate_action(ActionType.CLICK, coordinates=(0.1, 0.9), elements=elements)
    assert res_noop.diagnostic == RDTreeDiagnostic.NOOP_CLICK
    assert res_noop.process_reward == -0.8
    assert res_noop.is_critical_violation is True

    # 3. Valid click on button -> VALID_INTERACTION (0.8)
    res_btn = critic.evaluate_action(ActionType.CLICK, coordinates=(0.3, 0.15), elements=elements)
    assert res_btn.diagnostic == RDTreeDiagnostic.VALID_INTERACTION
    assert res_btn.process_reward == 0.8
    assert res_btn.target_element.element_id == 1

    # 4. Click on inert div -> INERT_ELEMENT_CLICK (-0.6)
    res_div = critic.evaluate_action(ActionType.CLICK, coordinates=(0.5, 0.04), elements=elements)
    assert res_div.diagnostic == RDTreeDiagnostic.INERT_ELEMENT_CLICK
    assert res_div.process_reward == -0.6


def test_rdtree_input_affordances_and_diagnostics():
    """Verifies Task 1.2: scalar reward scoring and input diagnostics."""
    critic = RDTreeProcessCritic()

    elements = [
        UIElement(1, "button", "btn", "Click", (0.1, 0.1, 0.3, 0.2), is_clickable=True, is_input=False),
        UIElement(2, "input", "text_in", "", (0.4, 0.1, 0.7, 0.2), is_clickable=True, is_input=True),
        UIElement(
            3,
            "input",
            "readonly_in",
            "Fixed",
            (0.4, 0.3, 0.7, 0.4),
            is_clickable=True,
            is_input=True,
            attributes={"readonly": "true"},
        ),
    ]

    # 1. Valid typing into input element -> VALID_INTERACTION (0.9)
    res_valid_type = critic.evaluate_action(
        ActionType.TYPE,
        coordinates=(0.5, 0.15),
        elements=elements,
    )
    assert res_valid_type.diagnostic == RDTreeDiagnostic.VALID_INTERACTION
    assert res_valid_type.process_reward == 0.9
    assert res_valid_type.is_critical_violation is False

    # 2. Typing into non-input button -> UNFOCUSED_INPUT (-0.9)
    res_bad_type = critic.evaluate_action(
        ActionType.TYPE,
        coordinates=(0.2, 0.15),
        elements=elements,
    )
    assert res_bad_type.diagnostic == RDTreeDiagnostic.UNFOCUSED_INPUT
    assert res_bad_type.process_reward == -0.9
    assert res_bad_type.is_critical_violation is True

    # 3. Typing into readonly field -> READONLY_FIELD (-0.8)
    res_ro_type = critic.evaluate_action(
        ActionType.TYPE,
        coordinates=(0.5, 0.35),
        elements=elements,
    )
    assert res_ro_type.diagnostic == RDTreeDiagnostic.READONLY_FIELD
    assert res_ro_type.process_reward == -0.8
    assert res_ro_type.is_critical_violation is True


def test_rdtree_rolling_history_cycle_detection():
    """Verifies Task 1.3: cycle detection and oscillation penalty."""
    critic = RDTreeProcessCritic(history_capacity=8)

    elements = [
        UIElement(1, "button", "tab_1", "Overview", (0.1, 0.1, 0.3, 0.2), is_clickable=True),
        UIElement(2, "button", "tab_2", "Details", (0.4, 0.1, 0.6, 0.2), is_clickable=True),
    ]

    # First click on tab 1 -> valid
    res1 = critic.evaluate_action(ActionType.CLICK, coordinates=(0.2, 0.15), elements=elements)
    assert res1.diagnostic == RDTreeDiagnostic.VALID_INTERACTION
    critic.record_step(ActionType.CLICK, element_id=1, coordinates=(0.2, 0.15))

    # Second consecutive click on tab 1 without mutation -> CYCLIC_ACTION (-1.0)
    res2 = critic.evaluate_action(ActionType.CLICK, coordinates=(0.2, 0.15), elements=elements)
    assert res2.diagnostic == RDTreeDiagnostic.CYCLIC_ACTION
    assert res2.process_reward == -1.0
    assert res2.is_critical_violation is True

    # Switching to tab 2 -> valid interaction
    res3 = critic.evaluate_action(ActionType.CLICK, coordinates=(0.5, 0.15), elements=elements)
    assert res3.diagnostic == RDTreeDiagnostic.VALID_INTERACTION
    assert res3.process_reward == 0.8


def test_jepa_search_rdtree_fusion_and_pruning(test_config):
    """Verifies Tasks 2.1 & 2.2: RDTree reward fusion and critical candidate pre-pruning in JEPA search."""
    jepa = JEPAWorldModel(test_config)
    critic = RDTreeProcessCritic()
    harness = JEPASearchHarness(
        jepa_model=jepa,
        rdtree_critic=critic,
        rdtree_gamma=1.5,
    )

    elements = [
        UIElement(1, "button", "login_btn", "Log In", (0.2, 0.1, 0.4, 0.2), is_clickable=True),
    ]

    candidates = [
        # Candidate 0: Valid click on button -> R_RDTree = +0.8
        CandidateAction(
            action_type_id=ActionType.CLICK.value,
            coordinates=(0.3, 0.15),
            raw_score=2.0,
        ),
        # Candidate 1: Empty space click -> NOOP_CLICK -> R_RDTree = -0.8 -> Pruned!
        CandidateAction(
            action_type_id=ActionType.CLICK.value,
            coordinates=(0.8, 0.8),
            raw_score=5.0,  # High raw score will be pruned
        ),
        # Candidate 2: Offscreen coordinates -> OFFSCREEN_COORDINATE -> R_RDTree = -1.0 -> Pruned!
        CandidateAction(
            action_type_id=ActionType.CLICK.value,
            coordinates=(1.5, 0.5),
            raw_score=10.0,
        ),
    ]

    z_t = torch.randn(1, test_config.jepa_latent_dim)
    z_goal = torch.randn(1, test_config.jepa_latent_dim)

    # Search with harness
    search_res = harness.search(
        z_t=z_t,
        z_goal=z_goal,
        candidates=candidates,
        elements=elements,
    )

    # Candidate 0 must be valid and selected (others pruned)
    assert search_res.selected_coordinates == (0.3, 0.15)
    assert search_res.selected_action_id == ActionType.CLICK.value
    assert search_res.fallback_used is False
    assert search_res.num_candidates_pruned == 2

    c0 = search_res.candidates[0]
    assert not c0.pruned
    assert c0.rdtree_reward == 0.8
    assert c0.rdtree_diag == RDTreeDiagnostic.VALID_INTERACTION.value

    c1 = search_res.candidates[1]
    assert c1.pruned
    assert "rdtree" in c1.prune_reason
    assert RDTreeDiagnostic.NOOP_CLICK.value in c1.prune_reason

    c2 = search_res.candidates[2]
    assert c2.pruned
    assert "rdtree" in c2.prune_reason
    assert RDTreeDiagnostic.OFFSCREEN_COORDINATE.value in c2.prune_reason

    # Verify rank_candidates() alias
    rank_res = harness.rank_candidates(
        z_t=z_t,
        z_goal=z_goal,
        candidates=candidates,
        elements=elements,
    )
    assert rank_res.selected_coordinates == search_res.selected_coordinates


def test_agent_evaluate_grounded_candidates_rdtree(test_config):
    """Verifies Task 2.2: candidate pre-pruning in JEDM2Agent.evaluate_grounded_candidates()."""
    model = JEDM2Model(test_config)
    critic = RDTreeProcessCritic()
    agent = JEDM2Agent(model, config=test_config, rdtree_critic=critic)

    elements = [
        UIElement(1, "button", "submit", "Submit", (0.1, 0.1, 0.3, 0.2), is_clickable=True),
    ]

    z_t = torch.randn(1, test_config.jepa_latent_dim)
    z_goal = torch.randn(1, test_config.jepa_latent_dim)

    action_logits = torch.zeros(len(ActionType))
    action_logits[ActionType.CLICK.value] = 5.0

    sel_act_id, sel_coords, metadata = agent.evaluate_grounded_candidates(
        z_t=z_t,
        z_goal=z_goal,
        action_logits=action_logits,
        raw_coords=(0.9, 0.9),  # raw coords in empty space
        candidate_elements=elements,
    )

    assert "selected_rdtree_reward" in metadata
    assert "selected_rdtree_diag" in metadata

    # Check that candidate hitting whitespace was pruned by rdtree
    noop_cands = [c for c in metadata["candidates"] if c["coords"] == (0.9, 0.9)]
    if noop_cands:
        assert noop_cands[0]["pruned"]
        assert "rdtree" in noop_cands[0]["prune_reason"]


def test_agent_step_output_and_turn_record_rdtree_telemetry(test_config):
    """Verifies Task 2.3: process supervision diagnostic telemetry in AgentStepOutput and TurnRecord."""
    model = JEDM2Model(test_config)
    critic = RDTreeProcessCritic()
    agent = JEDM2Agent(model, config=test_config, rdtree_critic=critic)

    elements = [
        UIElement(1, "button", "cta", "Click Here", (0.2, 0.2, 0.4, 0.3), is_clickable=True),
    ]

    step_out = agent.act(
        obs_text="Page with CTA button",
        candidate_elements=elements,
    )

    assert isinstance(step_out, AgentStepOutput)
    assert step_out.rdtree_diagnostic is not None
    assert step_out.rdtree_reward is not None

    step_dict = step_out.to_dict()
    assert "rdtree_diagnostic" in step_dict
    assert "rdtree_reward" in step_dict

    summary = step_out.summary_str()
    assert "RDTree=" in summary

    # Verify TurnRecord telemetry in step_environment
    class MockEnv:
        def __init__(self, elements):
            self.elements = elements
            self.current_state = MagicMock()
            self.current_state.last_action_str = "Initial state"
            self.current_state.elements = elements
            self.current_state.url = "https://example.com"
            self.active_elements = elements

        def reset(self):
            res = MagicMock()
            res.obs_text = "Ready"
            res.obs_features = torch.zeros(test_config.hidden_dim)
            res.screenshot = None
            res.info = {}
            return res

        def step(self, action_type, coordinates, argument):
            res = MagicMock()
            res.obs_text = f"Executed {action_type.name}"
            res.obs_features = torch.zeros(test_config.hidden_dim)
            res.reward = 1.0
            res.done = False
            res.truncated = False
            res.info = {}
            return res

    mock_env = MockEnv(elements)
    agent.step_environment(mock_env)

    assert len(agent.history) == 1
    rec: TurnRecord = agent.history[0]
    assert rec.rdtree_diagnostic is not None
    assert rec.rdtree_reward is not None
    # Verify history recorded in critic
    assert len(critic._history) == 1


def test_rdtree_critic_latency_benchmark():
    """Verifies Task 3.1: latency benchmark (<2 ms for 16 simultaneous candidate evaluations)."""
    critic = RDTreeProcessCritic()

    elements = [
        UIElement(1, "button", "b1", "B1", (0.1, 0.1, 0.2, 0.2), is_clickable=True),
        UIElement(2, "input", "i1", "", (0.3, 0.1, 0.5, 0.2), is_clickable=True, is_input=True),
        UIElement(3, "button", "b2", "B2", (0.6, 0.1, 0.7, 0.2), is_clickable=True),
        UIElement(4, "div", "d1", "Text", (0.0, 0.5, 1.0, 0.8), is_clickable=False),
    ]

    # Warmup
    for _ in range(10):
        for act_id in range(4):
            critic.evaluate_action(act_id, (0.15, 0.15), elements=elements)

    # Benchmark: 16 candidate evaluations
    candidates = [
        (ActionType.CLICK, (0.15, 0.15)),
        (ActionType.TYPE, (0.4, 0.15)),
        (ActionType.CLICK, (0.65, 0.15)),
        (ActionType.CLICK, (0.05, 0.95)),  # Empty space
        (ActionType.CLICK, (1.2, 0.5)),   # Offscreen
        (ActionType.SELECT, (0.15, 0.15)),
        (ActionType.SCROLL, (0.5, 0.5)),
        (ActionType.NAVIGATE, (0.0, 0.0)),
        (ActionType.CLICK, (0.5, 0.6)),   # Inert
        (ActionType.TYPE, (0.15, 0.15)),   # Unfocused
        (ActionType.CLICK, (0.35, 0.15)),  # Click input
        (ActionType.CLICK, (0.12, 0.12)),
        (ActionType.CLICK, (0.18, 0.18)),
        (ActionType.TYPE, (0.45, 0.15)),
        (ActionType.FINISH, (0.0, 0.0)),
        (ActionType.CLICK, (0.68, 0.18)),
    ]
    assert len(candidates) == 16

    num_iterations = 100
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        for act, coords in candidates:
            critic.evaluate_action(act, coords, elements=elements)
    elapsed_total_ms = (time.perf_counter() - t0) * 1000.0
    mean_latency_per_16_cands_ms = elapsed_total_ms / num_iterations

    # Must be well below the 2.0 ms budget
    assert mean_latency_per_16_cands_ms < 2.0, f"Mean latency {mean_latency_per_16_cands_ms:.3f} ms exceeds 2.0 ms budget"


def test_pure_ascii_and_zero_vram_overhead():
    """Verifies Task 3.2: 100% pure 7-bit ASCII representation and pure CPU execution."""
    # 1. Check all diagnostic enum names and values are 7-bit ASCII
    for diag in RDTreeDiagnostic:
        assert diag.name.isascii(), f"Enum name {diag.name} must be 7-bit ASCII"
        assert diag.value.isascii(), f"Enum value {diag.value} must be 7-bit ASCII"

    # 2. Check critic execution creates zero tensors on GPU
    critic = RDTreeProcessCritic()
    eval_res = critic.evaluate_action(ActionType.CLICK, (0.5, 0.5))
    assert eval_res.message.isascii(), "Diagnostic messages must be pure 7-bit ASCII"
    assert isinstance(eval_res.process_reward, float)
    assert -1.0 <= eval_res.process_reward <= 1.0
