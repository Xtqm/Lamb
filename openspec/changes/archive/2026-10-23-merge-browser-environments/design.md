# Design: Unified Playwright Environment Architecture

## Context

See `proposal.md` for background and problem statement.

Currently:
- `jed_m2/browser.py` defines `PlaywrightBrowserEnvironment` (Phase 7) focused on Chromium browser automation, live viewport screenshot capture (`PlaywrightStepResult`), and interactive agent rollouts.
- `jed_m2/playwright_env.py` defines `PlaywrightWebEnvironment` and `WebTaskOracle` (Phase 15) focused on RL group rollout sampling, local HTML fixtures, and programmatic oracle task evaluation.

Both classes implement nearly identical Chromium browser lifecycle management, 1280x800 viewport coordinate mapping, action dispatching (`CLICK`, `TYPE`, `SCROLL`, `NAVIGATE`, `FINISH`), accessibility tree extraction, and screenshot tensor formatting.

## Goals / Non-Goals

**Goals:**
- Consolidate all browser management logic into a single canonical class `PlaywrightEnvironment` in `jed_m2/browser.py`.
- Incorporate `WebTaskOracle`, oracle evaluation, and local HTML fixture resolution directly into the unified environment.
- Preserve 100% backward compatibility for all existing call sites and test suites via module-level aliases.
- Provide a robust `mock_mode` and automatic sandbox fallback to avoid browser pipe timeouts during testing.
- Eliminate over 600 lines of duplicated code.

**Non-Goals:**
- Altering the underlying model architectures (`JEDM2Model`, `JEDM2Agent`).
- Creating a new directory hierarchy (such as `jed_m2/envs/`); broad package layout reorganization is reserved for Phase 4.
- Removing or renaming `PlaywrightBrowserEnvironment` or `PlaywrightWebEnvironment` public symbols.

## Decisions

### 1. Canonical Class Location and Compatibility Layer

- **Decision**: Implement `PlaywrightEnvironment` in `jed_m2/browser.py` as the canonical implementation.
- **Rationale**: `jed_m2/browser.py` is the established home for live browser automation. Moving `WebTaskOracle` and fixture helpers into `jed_m2/browser.py` consolidates browser tooling in one location.
- **Backward Compatibility**:
  - In `jed_m2/browser.py`:
    ```python
    PlaywrightBrowserEnvironment = PlaywrightEnvironment
    ```
  - In `jed_m2/playwright_env.py`:
    Turn this file into a thin re-export module:
    ```python
    from jed_m2.browser import (
        PlaywrightEnvironment,
        PlaywrightStepResult,
        WebTaskOracle,
        get_default_web_oracles,
        get_fixture_path,
    )
    PlaywrightWebEnvironment = PlaywrightEnvironment
    ```
  - In `jed_m2/__init__.py`:
    Export `PlaywrightEnvironment`, `PlaywrightBrowserEnvironment`, `PlaywrightWebEnvironment`, `PlaywrightStepResult`, `WebTaskOracle`, and helpers.

### 2. Unified Constructor Signature

The canonical `PlaywrightEnvironment` will accept parameters from both predecessors with sensible defaults:

```python
class PlaywrightEnvironment:
    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        scenario_id: Optional[str] = None,
        oracle: Optional[WebTaskOracle] = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        timeout_ms: int = 10000,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        mock_mode: bool = False,
        fixture_path: Optional[Union[str, Path]] = None,
        local_benchmark_url: Optional[str] = None,
    ):
        ...
```

- When `scenario_id` is passed, the environment automatically attaches the corresponding default oracle and local fixture if present.
- When an explicit `oracle` is passed, it is used for step-level evaluation.
- When `mock_mode=True`, or if Playwright fails to launch due to sandbox pipe constraints, the environment falls back to deterministic mock DOM snapshots.

### 3. Unified StepResult Schema

`PlaywrightStepResult` subclasses `StepResult` to satisfy both Phase 7 agent step contracts and Phase 15 RL rollout contracts:

```python
@dataclass
class PlaywrightStepResult(StepResult):
    screenshot: Optional[torch.Tensor] = None
    raw_accessibility_tree: Optional[Union[Dict[str, Any], str]] = None
    oracle_details: Optional[Dict[str, Any]] = None
```

In `step()`, the environment populates:
- `self.info["screenshot"] = screenshot` (for Phase 15 `test_real_rl.py` compatibility)
- `self.info["oracle"] = oracle_details`
- `result.screenshot = screenshot` (for Phase 7 `verify_phase7.py` compatibility)

### 4. Sandbox-Resilient Mock Mode

In sandboxed environments (such as Windows sandbox execution where child processes cannot open pipe sockets to Chromium):
- If `mock_mode=True` or `sync_playwright().start()` raises an error during initialization, the environment sets `self.is_mock = True`.
- In mock mode:
  - `reset()` returns synthetic DOM trees matching the requested scenario fixture.
  - `step()` updates mock DOM states, evaluates oracles against mock DOM strings, and emits synthetic blank/rendered screenshots without hanging on OS pipes.

## Risks / Trade-offs

- **[Risk]** Downstream scripts rely on idiosyncratic `info` dictionary keys:
  - *Mitigation*: Ensure `info` dictionary contains keys expected by both Phase 7 (`step`, `url`, `title`, `element_count`) and Phase 15 (`screenshot`, `step`, `oracle_details`).
- **[Risk]** Type checks like `isinstance(env, PlaywrightWebEnvironment)`:
  - *Mitigation*: Aliasing `PlaywrightWebEnvironment = PlaywrightEnvironment` guarantees that `isinstance` checks succeed identically for either class name.
- **[Risk]** Context manager behavior (`with PlaywrightEnvironment(...) as env:`):
  - *Mitigation*: Implement `__enter__` and `__exit__` cleanly to close Chromium and page contexts safely.

## Migration Plan

1. Migrate `WebTaskOracle`, `get_default_web_oracles`, and `get_fixture_path` into `jed_m2/browser.py`.
2. Refactor `PlaywrightBrowserEnvironment` into `PlaywrightEnvironment` in `jed_m2/browser.py`, incorporating oracle verification, mock fallback, and fixture support.
3. Replace the implementation in `jed_m2/playwright_env.py` with thin re-exports of canonical classes and functions.
4. Update `jed_m2/__init__.py` to export canonical symbols and backward-compatible aliases.
5. Verify zero regressions via `verify_phase7.py`, `tests/test_real_rl.py`, and interactive agent checks.
