"""
Unit tests for JED-M2 Procedural Obstacles, Self-Healing Interception, and Programmatic Outcome Verifiers.
"""

import pytest
from jed_m2.heads import ActionType
from jed_m2.scenario_generator import (
    DynamicBookingEnvironment,
    DynamicCRMEnvironment,
    DynamicCloudAdminEnvironment,
    DynamicEcommerceEnvironment,
    DynamicFormPortalEnvironment,
    ObstacleInjector,
    ObstacleState,
    OutcomeVerificationResult,
    verify_booking_outcome,
    verify_cloud_outcome,
    verify_crm_outcome,
    verify_ecommerce_outcome,
    verify_environment_outcome,
    verify_form_outcome,
)


def test_obstacle_injector_cookie_consent_interception():
    """Verify that an active cookie consent obstacle blocks underlying clicks until dismissed."""
    env = DynamicEcommerceEnvironment(seed=42)
    env.reset()

    # Inject cookie consent overlay
    injector = ObstacleInjector()
    obs_state = injector.inject_obstacle(env, obstacle_type="cookie_consent")

    assert env.obstacle_state is not None
    assert env.obstacle_state.active
    assert env.obstacle_state.obstacle_type == "cookie_consent"

    # Observation text must contain clear obstacle cue
    step_res = env._format_step_result(reward=0.0, done=False, truncated=False)
    assert "[OBSTACLE: cookie_consent]" in step_res.obs_text

    # Elements must include cookie consent buttons
    cookie_btn = next((el for el in env.current_state.elements if el.role in ("cookie_accept_btn", "cookie_dismiss_btn")), None)
    assert cookie_btn is not None

    # Clicking an underlying element should be intercepted/blocked while modal is active
    search_bar = next((el for el in env.current_state.elements if el.role == "search_bar"), None)
    if search_bar:
        prev_search = env.search_query
        env.step(ActionType.TYPE, coordinates=search_bar.center, argument="headphones")
        # Action was intercepted; search query remains unchanged
        assert env.search_query == prev_search

    # Self-healing: click accept button
    env.step(ActionType.CLICK, coordinates=cookie_btn.center)
    assert not env.obstacle_state.active
    assert "obstacle_dismissed" in env.milestones_achieved

    # Now underlying interactions succeed
    if search_bar:
        env.step(ActionType.TYPE, coordinates=search_bar.center, argument="headphones")
        assert env.search_query == "headphones"


def test_obstacle_injector_validation_error_and_loading_delay():
    """Verify inline validation error and loading delay obstacles."""
    env = DynamicFormPortalEnvironment(seed=99)
    env.reset()
    injector = ObstacleInjector()

    # 1. Validation error injection
    injector.inject_obstacle(env, obstacle_type="validation_error")
    assert env.obstacle_state.active
    assert any("error" in el.role or "alert" in el.role for el in env.current_state.elements)

    # Dismiss/fix validation error
    dismiss_el = next((el for el in env.current_state.elements if el.role == "dismiss_validation_btn"), None)
    if dismiss_el:
        env.step(ActionType.CLICK, coordinates=dismiss_el.center)
        assert not env.obstacle_state.active

    # 2. Loading delay injection
    injector.inject_obstacle(env, obstacle_type="loading_delay")
    assert env.obstacle_state.active
    assert any("loading" in el.role or "spinner" in el.role for el in env.current_state.elements)

    # Step clears delay
    env.step(ActionType.CLICK, coordinates=(0.5, 0.5))
    assert not env.obstacle_state.active


def test_programmatic_verifiers_across_all_five_domains():
    """Verify that programmatic outcome verifiers accurately evaluate satisfaction across 5 domains."""
    # 1. E-Commerce
    ecom = DynamicEcommerceEnvironment(seed=50)
    ecom.reset()
    assert not verify_ecommerce_outcome(ecom).goal_satisfied
    # Manually satisfy goal
    ecom.cart.append(ecom.target_item)
    ecom.stage = "complete"
    ecom.goal_satisfied = True
    res_ecom = verify_ecommerce_outcome(ecom)
    assert res_ecom.goal_satisfied
    assert "order_placed" in res_ecom.milestones_achieved

    # 2. Form Portal
    form = DynamicFormPortalEnvironment(seed=50)
    form.reset()
    assert not verify_form_outcome(form).goal_satisfied
    form.goal_satisfied = True
    assert verify_form_outcome(form).goal_satisfied

    # 3. Booking
    booking = DynamicBookingEnvironment(seed=50)
    booking.reset()
    assert not verify_booking_outcome(booking).goal_satisfied

    # 4. CRM
    crm = DynamicCRMEnvironment(seed=50)
    crm.reset()
    assert not verify_crm_outcome(crm).goal_satisfied

    # 5. Cloud Admin
    cloud = DynamicCloudAdminEnvironment(seed=50)
    cloud.reset()
    assert not verify_cloud_outcome(cloud).goal_satisfied
    target_inst = next((i for i in cloud.instances if i["id"] == cloud.target_instance_id), None)
    if target_inst:
        target_inst["state"] = "Stopped"
    cloud.goal_satisfied = True
    assert verify_cloud_outcome(cloud).goal_satisfied

    # Unified verifier dispatch
    assert verify_environment_outcome(cloud).goal_satisfied
