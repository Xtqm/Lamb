# Proposal: Consolidate Playwright Browser Environments into Unified Interface

## Why

The JED-M2 codebase currently maintains two parallel, divergent implementations of the Playwright browser automation sandbox:
1. `jed_m2/browser.py` (`PlaywrightBrowserEnvironment`, 701 lines): Implemented in Phase 7 for multimodal screenshot capture and live interactive agent stepping.
2. `jed_m2/playwright_env.py` (`PlaywrightWebEnvironment`, 687 lines): Implemented in Phase 15 for RL rollout sampling with `WebTaskOracle` programmatic reward shaping and local HTML fixture loading.

These two implementations duplicate over 600 lines of browser lifecycle management, 1280x800 viewport coordinate mapping, action dispatching (`CLICK`, `TYPE`, `SCROLL`, `NAVIGATE`, `FINISH`), and DOM tree extraction. Maintaining both creates architectural drift, confuses developers about which environment to import, and introduces duplicate maintenance overhead. Consolidating them into a canonical `PlaywrightEnvironment` in `jed_m2/browser.py` with backward-compatible aliases eliminates redundancy while unifying multimodal screenshot grounding and oracle reward evaluation.

## What Changes

- **Canonical Browser Environment**: Consolidate `PlaywrightBrowserEnvironment` and `PlaywrightWebEnvironment` into a single, canonical `PlaywrightEnvironment` in `jed_m2/browser.py`.
- **Integrate WebTaskOracle & Fixtures**: Migrate `WebTaskOracle`, `get_default_web_oracles`, and `get_fixture_path` into `jed_m2/browser.py` (or sub-exported cleanly) so the unified environment natively supports programmatic task verification and milestone reward shaping.
- **Unified Step Result**: Unify `PlaywrightStepResult` to carry both multimodal viewport screenshot tensors (`screenshot: Optional[torch.Tensor]`) and programmatic oracle task metrics (`reward: float`, `oracle_details: Optional[Dict[str, Any]]`).
- **Resilient Mock Mode**: Standardize offline/sandboxed execution modes (`mock_mode=True` or automated fallback when Playwright browser pipes fail) to prevent test timeouts in constrained sandboxes.
- **100% Backward Compatibility**:
  - Alias `PlaywrightBrowserEnvironment = PlaywrightEnvironment` in `jed_m2/browser.py`.
  - Re-export `PlaywrightWebEnvironment = PlaywrightEnvironment` and `WebTaskOracle` from `jed_m2/playwright_env.py`.
  - Re-export all historical symbols (`PlaywrightBrowserEnvironment`, `PlaywrightWebEnvironment`, `PlaywrightEnvironment`, `PlaywrightStepResult`, `WebTaskOracle`) in `jed_m2/__init__.py`.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `playwright-browser-environment`: Consolidate browser environment implementations into a single canonical environment supporting both multimodal visual screenshot capture and declarative oracle reward evaluation.

## Impact

- **Affected Code**: `jed_m2/browser.py`, `jed_m2/playwright_env.py`, `jed_m2/rl_pipeline.py`, `jed_m2/agent.py`, `interactive_agent.py`, `train_rl.py`.
- **APIs**: Unifies environment creation under `PlaywrightEnvironment`, while maintaining `PlaywrightBrowserEnvironment` and `PlaywrightWebEnvironment` aliases.
- **Dependencies**: No new external dependencies (pure PyTorch + optional `playwright` / `PIL`).
