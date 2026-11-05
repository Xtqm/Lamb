"""
Unit tests for JED-M2 DynamicBookingEnvironment (Travel Booking Gym Domain).
"""

import pytest
from jed_m2.heads import ActionType
from jed_m2.scenario_generator import (
    DynamicBookingEnvironment,
    FlightBookingRecord,
    verify_booking_outcome,
)


def test_booking_environment_init_and_reset():
    """Verify initialization across distinct seeds produces valid state and goals."""
    for seed in [42, 101, 777]:
        env = DynamicBookingEnvironment(seed=seed)
        step_res = env.reset()

        assert step_res.obs_text is not None
        assert "Goal:" in step_res.obs_text
        assert env.target_origin != env.target_destination
        assert env.target_passenger_name != ""
        assert env.booking_stage == "search"
        assert len(env.current_state.elements) > 0
        assert not env.goal_satisfied
        assert not env.is_terminated


def test_booking_environment_end_to_end_flow():
    """Verify step-by-step navigation through travel booking stages to goal satisfaction."""
    env = DynamicBookingEnvironment(seed=123)
    env.reset()

    # 1. Search stage: Fill origin, destination, and click search
    origin_el = next(el for el in env.current_state.elements if el.role == "origin_input")
    dest_el = next(el for el in env.current_state.elements if el.role == "dest_input")
    search_btn = next(el for el in env.current_state.elements if el.role == "search_flights_btn")

    env.step(ActionType.TYPE, coordinates=origin_el.center, argument=env.target_origin)
    env.step(ActionType.TYPE, coordinates=dest_el.center, argument=env.target_destination)
    res_search = env.step(ActionType.CLICK, coordinates=search_btn.center)

    assert env.booking_stage == "flights"
    assert "search_submitted" in env.milestones_achieved

    # 2. Flights stage: Select target flight
    target_flight_btn = next(
        (el for el in env.current_state.elements if el.role == f"select_flight_{env.target_flight_number}"),
        None,
    )
    assert target_flight_btn is not None
    env.step(ActionType.CLICK, coordinates=target_flight_btn.center)

    assert env.booking_stage == "details"
    assert "flight_selected" in env.milestones_achieved

    # 3. Details stage: Fill passenger name, select seat, proceed to review
    name_el = next(el for el in env.current_state.elements if el.role == "passenger_name_input")
    email_el = next(el for el in env.current_state.elements if el.role == "passenger_email_input")
    seat_btn = next(el for el in env.current_state.elements if el.role.startswith("select_seat_"))
    review_btn = next(el for el in env.current_state.elements if el.role == "proceed_to_review_btn")

    env.step(ActionType.TYPE, coordinates=name_el.center, argument=env.target_passenger_name)
    env.step(ActionType.TYPE, coordinates=email_el.center, argument=env.target_passenger_email)
    env.step(ActionType.CLICK, coordinates=seat_btn.center)
    env.step(ActionType.CLICK, coordinates=review_btn.center)

    assert env.booking_stage == "review"
    assert "passenger_details_filled" in env.milestones_achieved
    assert "seat_selected" in env.milestones_achieved

    # 4. Review stage: Confirm booking
    confirm_btn = next(el for el in env.current_state.elements if el.role == "confirm_booking_btn")
    res_confirm = env.step(ActionType.CLICK, coordinates=confirm_btn.center)

    assert env.booking_stage == "complete"
    assert env.goal_satisfied
    assert env.confirmed_booking is not None
    assert env.confirmed_booking.confirmed
    assert env.confirmed_booking.origin == env.target_origin
    assert env.confirmed_booking.destination == env.target_destination
    assert env.confirmed_booking.passenger_name == env.target_passenger_name

    # 5. Programmatic outcome verification
    verification = verify_booking_outcome(env)
    assert verification.goal_satisfied
    assert "booking_confirmed" in verification.milestones_achieved
    assert len(verification.detected_errors) == 0
