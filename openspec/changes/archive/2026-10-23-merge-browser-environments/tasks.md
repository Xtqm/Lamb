# Tasks

## 1. Oracle & Fixture Consolidation in `jed_m2/browser.py`

- [x] 1.1 Migrate `WebTaskOracle`, `get_default_web_oracles`, and `get_fixture_path` into `jed_m2/browser.py`; verify declarative URL pattern matching, selector presence verification, and milestone reward evaluation with unit assertions.

## 2. Canonical `PlaywrightEnvironment` Implementation

- [x] 2.1 Implement canonical `PlaywrightEnvironment` in `jed_m2/browser.py`, combining live Chromium browser automation with declarative oracle verification, local HTML fixture loading, and resilient `mock_mode` fallback; verify initialization, reset, and action stepping.
- [x] 2.2 Unify `PlaywrightStepResult` in `jed_m2/browser.py` to support viewport screenshot tensors (`screenshot`), raw accessibility trees (`raw_accessibility_tree`), oracle details (`oracle_details`), and dictionary-based info lookups for backward compatibility; verify tensor shapes and key access.

## 3. Façade Re-exports & Compatibility Layer

- [x] 3.1 Refactor `jed_m2/playwright_env.py` into a thin re-export façade aliasing `PlaywrightWebEnvironment = PlaywrightEnvironment` and re-exporting `WebTaskOracle` from `jed_m2.browser`; verify class identity, `isinstance` checks, and module import resolution.
- [x] 3.2 Update `jed_m2/__init__.py` to re-export `PlaywrightEnvironment` along with legacy aliases `PlaywrightBrowserEnvironment` and `PlaywrightWebEnvironment`; verify public API symbols.

## 4. Integration Verification & Regression Suite

- [x] 4.1 Run `pytest tests/test_real_rl.py` and verify programmatic oracle evaluation, local fixture path resolution, and rollout batch shapes succeed with zero regressions.
- [x] 4.2 Run `python verify_phase7.py` and verify browser launch, DOM accessibility snapshot extraction, screenshot tensor formatting, and interactive agent stepping complete successfully.

## Workflow follow-up

- Review proposal, specs, design, and tasks with the user.
- Execute `/openspec-apply-change` to implement the tasks.
