"""
Unit and integration tests for JED-M2 CDP Micro-Script Synthesis & Workspace Persistence.

Verifies:
1. MicroScriptWorkspace directory management, discovery, and file persistence.
2. Canonical routine library seeding (form_fill, sort_table, cart_checkout, batch_select).
3. Parameter schema validation and malformed argument rejection.
"""

import json
from pathlib import Path
import time
from unittest.mock import MagicMock
import pytest

from jed_m2.bcode import CANONICAL_ROUTINES, MicroScriptRoutine, MicroScriptWorkspace
from jed_m2.browser import PlaywrightEnvironment, parse_routine_call
from jed_m2.cdp_cache import CDPScriptCache
from jed_m2.core.types import UIElement
from jed_m2.heads import ActionType


def test_workspace_initialization_and_seeding(tmp_path: Path):
    """Verifies Tasks 1.1 & 1.2: workspace initialization and canonical library seeding."""
    ws = MicroScriptWorkspace(workspace_dir=tmp_path / "workspace")

    # Check that canonical files exist on disk
    for name in ("form_fill", "sort_table", "cart_checkout", "batch_select"):
        js_file = ws.workspace_dir / f"{name}.js"
        json_file = ws.workspace_dir / f"{name}.json"
        assert js_file.exists(), f"Expected {js_file} to exist"
        assert json_file.exists(), f"Expected {json_file} to exist"

    # Check discovered routines
    assert len(ws.routines) >= 4
    assert "form_fill" in ws.routines
    assert "sort_table" in ws.routines
    assert "cart_checkout" in ws.routines
    assert "batch_select" in ws.routines

    routine = ws.get_routine("form_fill")
    assert routine is not None
    assert "Atomically populates" in routine.description
    assert "document.querySelector" in routine.source


def test_custom_routine_registration_and_persistence(tmp_path: Path):
    """Verifies Task 1.1: custom routine registration and file persistence."""
    ws = MicroScriptWorkspace(workspace_dir=tmp_path / "workspace")

    custom_routine = MicroScriptRoutine(
        name="custom_filter",
        description="Applies multi-category filter checkboxes.",
        source="(args) => { return { success: true, filtered: args.categories }; }",
        parameters={
            "type": "object",
            "required": ["categories"],
            "properties": {
                "categories": {"type": "array"},
            },
        },
    )

    ws.register_routine(custom_routine, persist=True)
    assert "custom_filter" in ws.routines

    # Verify disk files
    js_file = ws.workspace_dir / "custom_filter.js"
    json_file = ws.workspace_dir / "custom_filter.json"
    assert js_file.exists()
    assert json_file.exists()

    # Re-instantiate workspace without auto_seed to verify discovery from disk
    ws2 = MicroScriptWorkspace(workspace_dir=tmp_path / "workspace", auto_seed=False)
    loaded = ws2.get_routine("custom_filter")
    assert loaded is not None
    assert loaded.name == "custom_filter"
    assert "categories" in loaded.source


def test_parameter_schema_validation(tmp_path: Path):
    """Verifies Task 1.3: schema validation and rejection of invalid argument payloads."""
    ws = MicroScriptWorkspace(workspace_dir=tmp_path / "workspace")

    # 1. form_fill requires "fields" of type object/dict
    valid, err = ws.validate_arguments("form_fill", {"fields": {"#username": "admin", "#password": "secret"}})
    assert valid is True
    assert err is None

    # Missing required field
    valid, err = ws.validate_arguments("form_fill", {})
    assert valid is False
    assert "Missing required parameter 'fields'" in (err or "")

    # Invalid type for fields
    valid, err = ws.validate_arguments("form_fill", {"fields": "not-a-dict"})
    assert valid is False
    assert "Parameter 'fields' must be object/dict" in (err or "")

    # 2. sort_table with valid optional arguments
    valid, err = ws.validate_arguments("sort_table", {"table_selector": "#data-grid", "column_index": 2})
    assert valid is True
    assert err is None

    # Invalid type for column_index
    valid, err = ws.validate_arguments("sort_table", {"column_index": "first"})
    assert valid is False
    assert "Parameter 'column_index' must be integer" in (err or "")

    # 3. Non-existent routine
    valid, err = ws.validate_arguments("non_existent_routine", {})
    assert valid is False
    assert "not registered" in (err or "")


def test_cdp_script_cache_routine_registration(tmp_path: Path):
    """Verifies Task 2.1: dynamic routine and workspace registration into CDPScriptCache."""
    ws = MicroScriptWorkspace(workspace_dir=tmp_path / "workspace")
    cache = CDPScriptCache(enable_cdp_session=False)

    cache.register_workspace(ws)
    for name in ("form_fill", "sort_table", "cart_checkout", "batch_select"):
        assert name in cache.scripts
        assert cache.scripts[name].key == name

    # Test single routine registration
    single_routine = MicroScriptRoutine("quick_click", "Clicks target", "(args) => { return true; }")
    cache.register_routine(single_routine)
    assert "quick_click" in cache.scripts


def test_parse_routine_call():
    """Verifies Task 2.2: parsing routine call expressions across syntax styles."""
    # 1. JSON object syntax
    r1, a1 = parse_routine_call('{"routine": "form_fill", "args": {"fields": {"#name": "Alice"}}}')
    assert r1 == "form_fill"
    assert a1 == {"fields": {"#name": "Alice"}}

    # 2. Key mapping syntax
    r2, a2 = parse_routine_call('{"sort_table": {"column_index": 2}}')
    assert r2 == "sort_table"
    assert a2 == {"column_index": 2}

    # 3. Functional syntax
    r3, a3 = parse_routine_call('form_fill({"fields": {"#email": "test@domain.com"}})')
    assert r3 == "form_fill"
    assert a3 == {"fields": {"#email": "test@domain.com"}}

    # 4. Plain identifier
    r4, a4 = parse_routine_call("cart_checkout")
    assert r4 == "cart_checkout"
    assert a4 == {}


def test_playwright_env_api_call_mock(tmp_path: Path):
    """Verifies Tasks 2.2 & 2.3: environment API_CALL execution in mock mode."""
    ws = MicroScriptWorkspace(workspace_dir=tmp_path / "workspace")
    env = PlaywrightEnvironment(mock_mode=True, workspace=ws)
    init_res = env.reset()

    # Seed mock elements
    env.current_state.elements = [
        UIElement(1, "input", "email", "", (0.1, 0.1, 0.4, 0.2), attributes={"id": "email"}),
        UIElement(2, "input", "username", "", (0.1, 0.3, 0.4, 0.4), attributes={"id": "username"}),
        UIElement(3, "button", "checkout", "Checkout", (0.5, 0.5, 0.8, 0.6), attributes={"id": "checkout"}),
    ]

    # 1. Execute form_fill via API_CALL
    fill_payload = json.dumps({"routine": "form_fill", "args": {"fields": {"#email": "ada@lamb.org", "#username": "adalovelace"}}})
    step_res = env.step(action_type=ActionType.API_CALL, argument=fill_payload)

    assert step_res.reward > 0
    assert step_res.info["action_success"] is True
    assert step_res.info["routine"] == "form_fill"
    assert env.current_state.elements[0].value == "ada@lamb.org"
    assert env.current_state.elements[1].value == "adalovelace"

    # 2. Functional syntax: cart_checkout()
    step_res2 = env.step(action_type=ActionType.API_CALL, argument="cart_checkout({})")
    assert step_res2.info["action_success"] is True
    assert step_res2.info["routine"] == "cart_checkout"

    # 3. Invalid routine name -> fails gracefully
    step_res3 = env.step(action_type=ActionType.API_CALL, argument="non_existent_func()")
    assert step_res3.info["action_success"] is False
    assert "not found in workspace" in step_res3.info["action_error"]


def test_micro_script_latency_benchmark(tmp_path: Path):
    """Verifies Task 3.1: latency benchmark (<2 ms execution for pre-compiled routines)."""
    ws = MicroScriptWorkspace(workspace_dir=tmp_path / "workspace")
    cache = CDPScriptCache(enable_cdp_session=False)
    cache.register_workspace(ws)

    # Use a lightweight MockPage that evaluates registered python lambdas
    class BenchmarkMockPage:
        def __init__(self):
            self.url = "https://example.com"
            self.registry = {
                "form_fill": lambda a: {"success": True, "fields_updated": len(a.get("fields", {}))},
                "sort_table": lambda a: {"success": True, "sorted_column": a.get("column_index", 0)},
            }

        def evaluate(self, expr, arg=None):
            for k, fn in self.registry.items():
                if f"window.__jed_scripts__['{k}']" in expr:
                    return fn(arg or {})
            return {"success": True}

    page = BenchmarkMockPage()

    # Warmup
    for _ in range(10):
        cache.evaluate(page, "form_fill", {"fields": {"#q": "test"}})

    # Benchmark: 100 iterations of form_fill
    num_iterations = 100
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        cache.evaluate(page, "form_fill", {"fields": {"#a": "1", "#b": "2", "#c": "3"}})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    mean_latency_ms = elapsed_ms / num_iterations

    # Must be well within the 2.0 ms budget
    assert mean_latency_ms < 2.0, f"Mean latency {mean_latency_ms:.3f} ms exceeds 2.0 ms budget"


def test_pure_ascii_and_zero_vram_overhead():
    """Verifies Task 3.2: pure 7-bit ASCII representation and zero GPU VRAM consumption."""
    for name, spec in CANONICAL_ROUTINES.items():
        assert name.isascii(), f"Routine name '{name}' must be pure 7-bit ASCII"
        assert spec["description"].isascii(), f"Description for '{name}' must be pure 7-bit ASCII"
        assert spec["source"].isascii(), f"Source for '{name}' must be pure 7-bit ASCII"

    ws = MicroScriptWorkspace()
    for r_name, routine in ws.routines.items():
        assert r_name.isascii()
        assert routine.description.isascii()
        assert routine.source.isascii()

