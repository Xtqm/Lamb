"""
Unit tests for JED-M2 DynamicCRMEnvironment (Enterprise CRM & Sales Pipeline Domain).
"""

import pytest
from jed_m2.heads import ActionType
from jed_m2.scenario_generator import (
    CRMDeal,
    DynamicCRMEnvironment,
    verify_crm_outcome,
)


def test_crm_environment_init_and_reset():
    """Verify initialization across distinct seeds produces valid state and goals."""
    for seed in [11, 222, 999]:
        env = DynamicCRMEnvironment(seed=seed)
        step_res = env.reset()

        assert step_res.obs_text is not None
        assert "Goal:" in step_res.obs_text
        assert env.target_company != ""
        assert env.target_value > 0
        assert env.crm_view == "pipeline"
        assert len(env.deals) >= 3
        assert len(env.current_state.elements) > 0
        assert not env.goal_satisfied


def test_crm_environment_create_and_advance_deal():
    """Verify end-to-end deal creation and pipeline stage advancement to Closed-Won."""
    env = DynamicCRMEnvironment(seed=456)
    env.reset()

    # 1. Click "+ New Deal" button on Kanban pipeline
    new_deal_btn = next(el for el in env.current_state.elements if el.role == "new_deal_btn")
    env.step(ActionType.CLICK, coordinates=new_deal_btn.center)

    assert env.crm_view == "modal"
    assert "deal_modal_opened" in env.milestones_achieved

    # 2. Fill deal modal: company name, deal value, and save deal
    company_input = next(el for el in env.current_state.elements if el.role == "deal_company_input")
    value_input = next(el for el in env.current_state.elements if el.role == "deal_value_input")
    save_btn = next(el for el in env.current_state.elements if el.role == "save_deal_btn")

    env.step(ActionType.TYPE, coordinates=company_input.center, argument=env.target_company)
    env.step(ActionType.TYPE, coordinates=value_input.center, argument=str(int(env.target_value)))
    env.step(ActionType.CLICK, coordinates=save_btn.center)

    assert env.crm_view == "pipeline"
    assert "deal_created" in env.milestones_achieved

    # Find the created deal in the deals registry
    created_deal = next((d for d in env.deals.values() if d.company == env.target_company), None)
    assert created_deal is not None
    assert created_deal.value == env.target_value

    # 3. Advance deal through stages to Closed-Won
    # Advance button for target deal
    advance_btn = next(
        el for el in env.current_state.elements if el.role == f"advance_deal_{created_deal.deal_id}"
    )
    # Advance until Closed-Won (typically Lead -> Qualified -> Demo -> Proposal -> Closed-Won)
    for _ in range(4):
        if created_deal.stage == "Closed-Won":
            break
        env.step(ActionType.CLICK, coordinates=advance_btn.center)
        # Refresh advance_btn if re-rendered
        matching = [
            el for el in env.current_state.elements if el.role == f"advance_deal_{created_deal.deal_id}"
        ]
        if matching:
            advance_btn = matching[0]

    assert created_deal.stage == "Closed-Won"
    assert env.goal_satisfied
    assert "deal_closed_won" in env.milestones_achieved

    # 4. Programmatic outcome verification
    verification = verify_crm_outcome(env)
    assert verification.goal_satisfied
    assert "deal_closed_won" in verification.milestones_achieved
    assert len(verification.detected_errors) == 0
