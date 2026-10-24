# Tasks

## 1. Directory Structure & Test Suite Consolidation

- [x] 1.1 Create `verifications/` directory and move all 17 `verify_phase*.py` files into `verifications/`; verify file presence in `verifications/`.
- [x] 1.2 Move `test_jepa_regression.py` to `tests/test_jepa_regression.py` and create `pytest.ini` configuring `testpaths = ["tests"]`; verify test execution via `pytest`.
- [x] 1.3 Implement centralized verification runner `verifications/run_all.py` supporting `--phase` filtering and summary reporting; verify CLI via `python verifications/run_all.py --help`.

## 2. Backward-Compatible Root Forwarding Shims

- [x] 2.1 Generate lightweight `runpy.run_path` root forwarding shims for all 17 `verify_phase*.py` scripts delegating to `verifications/`; verify delegation logic.
- [x] 2.2 Verify execution of a representative phase script (`verify_phase1.py`) through both the root forwarding shim and direct `verifications/` path.

## 3. End-to-End Suite Verification

- [x] 3.1 Run `pytest` from repository root and verify all 16 test cases across `test_real_rl.py` and `test_jepa_regression.py` pass cleanly.
- [x] 3.2 Run `python verifications/verify_phase2.py` and verify multi-layer model verification passes with zero regressions.
