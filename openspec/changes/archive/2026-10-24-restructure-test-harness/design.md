# Design: Restructure Verification Scripts & Test Harness

## Context
See `proposal.md` for motivation. The root folder currently holds 17 standalone verification scripts (`verify_phase1.py` through `verify_phase14.py`) and a unit test script (`test_jepa_regression.py`). 

## Goals / Non-Goals

**Goals:**
- Move all 17 verification scripts into `verifications/`.
- Maintain 100% backward compatibility for root commands via lightweight `runpy.run_path` forwarding shims.
- Create `verifications/run_all.py` for selective or comprehensive phase verification runs.
- Move unit/regression tests into `tests/` (`tests/test_jepa_regression.py`).
- Add `pytest.ini` to configure automated test discovery in `tests/`.

**Non-Goals:**
- Modifying verification assertion logic or thresholds.
- Removing or renaming any legacy verification entrypoints.

## Decisions

### 1. `runpy.run_path` Forwarding Shims at Root
- **Decision**: Keep root shims for all 17 verification scripts that delegate directly to `verifications/verify_phaseX.py` using Python's standard `runpy.run_path(..., run_name="__main__")`.
- **Rationale**: This allows any existing automation, documentation, or user commands like `python verify_phase7.py` to continue executing seamlessly in-process without any breakage.
- **Alternatives Considered**: Shell aliases or moving files without shims. Rejected because it would break existing references in `README.md` and verification instructions.

### 2. Consolidated `tests/` Test Harness
- **Decision**: Move `test_jepa_regression.py` to `tests/test_jepa_regression.py` and configure `pytest.ini` with `testpaths = ["tests"]`.
- **Rationale**: `pytest` running from the repository root should automatically find and execute all pure-unit tests (`test_real_rl.py` and `test_jepa_regression.py`) quickly without picking up non-pytest scripts.

### 3. Centralized `verifications/run_all.py`
- **Decision**: Implement a CLI tool in `verifications/run_all.py` that lists available verification suites and supports running individual phases (`--phase 7`) or all fast phases with structured status reporting.

## Risks / Trade-offs

- **[Risk]**: Working directory or relative import path differences in `verifications/`.
  - **Mitigation**: Verification scripts import `jed_m2` as a package from `sys.path`. We will ensure root directory is added to `sys.path` in each verification script or runner if not already present.
