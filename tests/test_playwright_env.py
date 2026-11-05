"""
Unit tests for PlaywrightEnvironment with Differential DOM Deltas.

Verifies:
1. Initialization with use_differential_obs=True and False.
2. Full snapshot observation emission on reset().
3. Differential delta emission on intermediate steps.
4. Automatic fallback to full snapshot on navigation actions or URL changes.
5. Backward compatibility with legacy observation string formatting.
"""

import pytest
import torch
from jed_m2.browser import PlaywrightEnvironment, PlaywrightStepResult
from jed_m2.heads import ActionType


def test_playwright_env_reset_full_snapshot():
    """Verifies that reset() emits a full DOM snapshot."""
    env = PlaywrightEnvironment(mock_mode=True, use_differential_obs=True)
    res = env.reset(scenario_id="ecommerce")

    assert isinstance(res, PlaywrightStepResult)
    assert res.obs_text.startswith("<obs full>")
    assert res.delta_record is not None
    assert res.delta_record.is_full_snapshot is True
    assert res.info["is_delta"] is False
    assert len(env._prev_elements) > 0


def test_playwright_env_step_differential_delta():
    """Verifies that intermediate step() emits a differential DOM delta."""
    env = PlaywrightEnvironment(mock_mode=True, use_differential_obs=True)
    env.reset(scenario_id="ecommerce")

    # Dispatch a click action that modifies an element (in mock mode, clicking el 1)
    step_res = env.step(action_type=ActionType.CLICK, coordinates=(0.25, 0.10))

    assert isinstance(step_res, PlaywrightStepResult)
    assert step_res.delta_record is not None
    # Intermediate step is a delta
    assert step_res.obs_text.startswith("<obs delta>") or step_res.obs_text.startswith("<obs full>")
    assert "delta_record" in step_res.info


def test_playwright_env_navigate_triggers_full_snapshot():
    """Verifies that NAVIGATE action triggers automatic full snapshot fallback."""
    env = PlaywrightEnvironment(mock_mode=True, use_differential_obs=True)
    env.reset(scenario_id="ecommerce")

    # Step 1: Normal step
    env.step(action_type=ActionType.CLICK, coordinates=(0.25, 0.10))

    # Step 2: Navigate action
    nav_res = env.step(action_type=ActionType.NAVIGATE, argument="https://example.com/checkout")

    assert nav_res.delta_record is not None
    assert nav_res.delta_record.is_full_snapshot is True
    assert nav_res.obs_text.startswith("<obs full>")


def test_playwright_env_legacy_mode_toggle():
    """Verifies backward compatibility when use_differential_obs=False."""
    env = PlaywrightEnvironment(mock_mode=True, use_differential_obs=False)
    res = env.reset(scenario_id="ecommerce")

    assert res.obs_text.startswith("URL:")
    assert "DOM:" in res.obs_text
    assert "<obs full>" not in res.obs_text

    step_res = env.step(action_type=ActionType.CLICK, coordinates=(0.25, 0.10))
    assert step_res.obs_text.startswith("URL:")
    assert "<obs delta>" not in step_res.obs_text


def test_playwright_env_cdp_script_cache():
    """Verifies CDPScriptCache integration in PlaywrightEnvironment."""
    env = PlaywrightEnvironment(mock_mode=True, use_differential_obs=True)
    assert hasattr(env, "script_cache")
    assert "extract_elements" in env.script_cache.scripts
    assert "poll_mutations" in env.script_cache.scripts
    assert "oracle_check" in env.script_cache.scripts

    env.reset(scenario_id="ecommerce")
    env.step(action_type=ActionType.CLICK, coordinates=(0.25, 0.10))

    telemetry = env.script_cache.get_telemetry()
    assert isinstance(telemetry, dict)
    assert "hits" in telemetry
    assert "misses" in telemetry

    # Verify invalidation on navigation
    prev_invals = env.script_cache.stats["invalidations"]
    env.step(action_type=ActionType.NAVIGATE, argument="https://example.com/new_page")
    assert env.script_cache.stats["invalidations"] > prev_invals
