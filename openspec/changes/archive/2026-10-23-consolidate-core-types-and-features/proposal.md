# Proposal: Consolidate Core Types and Observation Feature Encoders

## Why

Across 14 rapid developmental phases, DOM element representations and interaction turn containers were repeatedly re-declared in parallel (`UIElement` in `jed_m2/dataset.py`, `UIElement` in `jed_m2/environment.py`, and `InteractiveElement` in `jed_m2/real_dataset.py`; along with `InteractionTurn`, `RealWebInteractionTurn`, and `PreferenceTurn`). Additionally, the deterministic semantic observation hashing logic (`compute_obs_features`) was duplicated between `dataset.py` and `real_dataset.py`, forcing `jed_m2/agent.py` to resort to an inline dynamic import of `RealWebTrajectoryDataset`.

Consolidating these core primitives into dedicated foundational modules (`jed_m2/core/types.py` and `jed_m2/core/features.py`) eliminates code duplication, breaks circular cross-module dependencies, and provides a unified interface for synthetic environments, real HTML DOM parsers, and agent runtimes while maintaining 100% backward compatibility.

## What Changes

- Create `jed_m2/core/types.py` containing the canonical `UIElement` dataclass supporting bounding box geometry (`x_min, y_min, x_max, y_max`), center calculation, DOM serialization (`to_dom_string`), interactive state flags (`is_clickable`, `is_input`, `value`), and optional HTML attributes (`attributes`).
- Alias `InteractiveElement = UIElement` in `jed_m2/core/types.py` and `jed_m2/real_dataset.py` to preserve existing API compatibility.
- Consolidate `InteractionTurn` in `jed_m2/core/types.py` with support for optional screenshot tensors, observation features, and UI element sequences; provide compatibility aliases for `RealWebInteractionTurn` and `PreferenceTurn`.
- Create `jed_m2/core/features.py` containing the standalone `compute_obs_features` function for MD5 word hashing and spatial bounding box feature projection.
- Update `jed_m2/dataset.py` and `jed_m2/real_dataset.py` to delegate `compute_obs_features` to the shared implementation.
- Update `jed_m2/agent.py` to import `compute_obs_features` directly from `jed_m2.core.features` at the top level, eliminating the inline dynamic import.
- Re-export all unified and aliased types in `jed_m2/__init__.py`.
- Verify backward compatibility and regression invariance across all existing verification tests (`verify_phase4.py`, `verify_phase6a.py`, `verify_phase12.py`, `verify_phase14.py`, `test_jepa_regression.py`).

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `synthetic-trajectory-dataset`: Standardize observation element definitions on the canonical `UIElement` and use the centralized `compute_obs_features` encoder.
- `real-web-trajectory-dataset`: Standardize pruned interactive candidates on the canonical `UIElement` (aliased as `InteractiveElement`) and utilize the centralized `compute_obs_features` encoder.

## Impact

- **Affected Files**:
  - `jed_m2/core/__init__.py` (new)
  - `jed_m2/core/types.py` (new)
  - `jed_m2/core/features.py` (new)
  - `jed_m2/dataset.py`
  - `jed_m2/environment.py`
  - `jed_m2/real_dataset.py`
  - `jed_m2/real_preference_dataset.py`
  - `jed_m2/agent.py`
  - `jed_m2/__init__.py`
- **APIs & Compatibility**:
  - Fully backward compatible: Existing callers importing `UIElement`, `InteractiveElement`, `InteractionTurn`, `RealWebInteractionTurn`, or `PreferenceTurn` from `jed_m2`, `jed_m2.dataset`, `jed_m2.real_dataset`, or `jed_m2.environment` will continue to function without changes.
- **Dependencies**: No external dependencies added (pure PyTorch and Python standard library).
