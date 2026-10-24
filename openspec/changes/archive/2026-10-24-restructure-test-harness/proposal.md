# Proposal: Restructure Verification Scripts & Test Harness

## Why
Currently, the repository root is heavily cluttered with 17 standalone verification scripts (`verify_phase1.py` through `verify_phase14.py`) alongside stray test scripts (`test_jepa_regression.py`). This creates visual noise, complicates project navigation, and fragments automated test discovery across root files and the `tests/` directory.

Consolidating verification scripts into a dedicated `verifications/` directory and establishing a unified test discovery harness in `tests/` will modernize the repository layout, improve maintainability, and ensure standard `pytest` workflows work out-of-the-box while preserving 100% backward compatibility for existing verification commands.

## What Changes
- **Organize Verification Scripts**: Move all 17 `verify_phase*.py` scripts into a dedicated `verifications/` directory.
- **Centralized Verification Runner**: Create `verifications/run_all.py` to allow executing single or multiple verification phases from a unified CLI entrypoint with summary reporting.
- **Backward-Compatible Root Shims**: Maintain lightweight forwarding launcher shims at the root (e.g. `verify_phase1.py` -> `verifications/verify_phase1.py`) so existing commands and documentation continue to work seamlessly.
- **Consolidate Automated Tests in `tests/`**: Move `test_jepa_regression.py` into `tests/test_jepa_regression.py`.
- **Pytest Configuration**: Add a standardized `pytest.ini` configuration specifying `testpaths = ["tests"]` and clean log formatting.

## Capabilities
*None (Tooling and repository infrastructure refactoring; `skip_specs: true` declared in `.openspec.yaml`).*

## Impact
- **Root Directory**: Cleaned up 18 test/verification files into organized subdirectories.
- **Verification Commands**: All `python verify_phase*.py` and `python verifications/verify_phase*.py` commands function identically.
- **Automated Testing**: Running `pytest` discovers all unit tests in `tests/` without scanning non-test modules.
- **Breaking Changes**: None. Forwarding shims maintain exact CLI invocation parity.
