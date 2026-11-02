"""
Unit tests for JED-M2 Deterministic Macro-Action Scaffold and Completion Referee.
"""

import pytest
from jed_m2.environment import WebOSEnvironment
from jed_m2.heads import ActionType
from jed_m2.scaffold import CompletionReferee, ExecutionScaffold


def test_scaffold_input_clearing_macro():
    env = WebOSEnvironment(scenario_id="ecommerce")
    env.reset()

    scaffold = ExecutionScaffold(clear_input_before_type=True)

    # Find the search input element
    search_input = [el for el in env.current_state.elements if el.is_input][0]
    search_input.value = "dirty previous text"

    # Execute TYPE action with scaffold
    step_res = scaffold.execute_step(
        env=env,
        action_type=ActionType.TYPE,
        coordinates=search_input.center,
        argument="headphones",
    )

    assert search_input.value == "headphones"
    assert "dirty" not in search_input.value


def test_completion_referee_ecommerce_stage():
    env = WebOSEnvironment(scenario_id="ecommerce")
    env.reset()

    # Initial state should not be satisfied
    assert not CompletionReferee.is_goal_satisfied(env)

    # Transition to results stage
    submit_btn = [el for el in env.current_state.elements if el.role == "submit_btn"][0]
    env.step(ActionType.CLICK, coordinates=submit_btn.center)
    assert not CompletionReferee.is_goal_satisfied(env)

    # Transition to product details
    item_card = [el for el in env.current_state.elements if el.role in ("first_product_link", "item_card")][0]
    env.step(ActionType.CLICK, coordinates=item_card.center)
    assert not CompletionReferee.is_goal_satisfied(env)

    # Transition to cart confirmed
    checkout_btn = [el for el in env.current_state.elements if el.role in ("checkout_btn", "proceed_checkout")][0]
    env.step(ActionType.CLICK, coordinates=checkout_btn.center)

    # Referee should detect goal completion without waiting for explicit FINISH!
    assert CompletionReferee.is_goal_satisfied(env)

    # Evaluate termination
    done, satisfied, reason = CompletionReferee.evaluate_termination(
        env=env,
        current_step=3,
        max_steps=8,
        action_type=ActionType.CLICK,
    )
    assert done is True
    assert satisfied is True
    assert reason == "referee_goal_achieved"


def test_completion_referee_loop_cutoff():
    env = WebOSEnvironment(scenario_id="travel")
    env.reset()

    done, satisfied, reason = CompletionReferee.evaluate_termination(
        env=env,
        current_step=4,
        max_steps=8,
        action_type=ActionType.CLICK,
        repeated_action_count=3,
    )
    assert done is True
    assert satisfied is False
    assert reason == "loop_truncated"


def test_decoupled_shaped_reward():
    from jed_m2.rl import compute_shaped_reward

    # Test baseline success
    rew_success = compute_shaped_reward(
        goal_satisfied=True,
        milestones_count=2,
        turn_count=3,
        iou_sum=0.8,
        action_type_matches=3,
        argument_matches=1,
    )
    # Expected: 2.0 (complete) + 1.0 (milestones) + 0.4 (iou) + 0.6 (type) + 0.3 (arg) - 0.06 (step) = 4.24
    assert rew_success > 3.5

    # Test in-progress positive exploration without goal completion
    rew_progress = compute_shaped_reward(
        goal_satisfied=False,
        milestones_count=1,
        turn_count=2,
        iou_sum=0.4,
        action_type_matches=2,
        argument_matches=0,
    )
    assert rew_progress > 0.0, "Decoupled reward must award positive return for correct sub-actions even before completion"


def test_scaffold_resolve_argument_macro():
    env = WebOSEnvironment(scenario_id="ecommerce")
    env.reset()
    scaffold = ExecutionScaffold()

    # Case 1: Plain natural text string passed directly
    res1 = scaffold._resolve_argument_macro(env, "Sony WH-1000XM5")
    assert res1 == "Sony WH-1000XM5"

    # Case 2: Delimited <arg> text </arg>
    res2 = scaffold._resolve_argument_macro(env, "<arg> Bose QuietComfort </arg>")
    assert res2 == "Bose QuietComfort"

    # Case 3: Empty string falls back to goal regex
    res3 = scaffold._resolve_argument_macro(env, "")
    assert res3 == "noise-cancelling headphones"

    # Case 4: Legacy tok_ placeholder falls back to goal regex
    res4 = scaffold._resolve_argument_macro(env, "tok_3412 tok_981")
    assert res4 == "noise-cancelling headphones"


