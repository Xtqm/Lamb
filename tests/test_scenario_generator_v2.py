"""
Unit test for DynamicScenarioGenerator multi-domain batch synthesis across 50 distinct random seeds.
"""

import pytest
from jed_m2.scenario_generator import DynamicScenarioGenerator


def test_dynamic_scenario_generator_batch_generation_50_seeds():
    generator = DynamicScenarioGenerator()
    domains = ["ecommerce", "form_portal", "booking", "crm", "cloud_admin", "web_browser"]

    assert set(domains).issubset(set(generator.AVAILABLE_DOMAINS))

    # Test 50 distinct seeds across all 5+ domains
    for i in range(50):
        dom = domains[i % len(domains)]
        seed = 1000 + i * 37
        env = generator.create_environment(domain=dom, seed=seed)

        assert env is not None
        assert env.current_state is not None
        assert len(env.current_state.elements) > 0
        assert env.active_scenario.goal != ""
        assert not env.goal_satisfied
        assert not env.is_terminated
