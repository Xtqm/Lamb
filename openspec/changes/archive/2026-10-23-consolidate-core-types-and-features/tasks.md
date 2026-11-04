# Tasks

## 1. Foundational Core Module Creation

- [x] 1.1 Create `jed_m2/core/__init__.py` and implement canonical `UIElement` and `InteractionTurn` in `jed_m2/core/types.py` with normalized bounding box geometry, centers, dimensions, interactive state attributes, and `to_dom_string()` serialization; verify instantiation and serialization with unit assertions.
- [x] 1.2 Implement standalone `compute_obs_features` in `jed_m2/core/features.py` for deterministic MD5 word hashing and spatial bounding box feature projection; verify deterministic output against reference tensor values.

## 2. Synthetic & Environment Integration

- [x] 2.1 Refactor `jed_m2/dataset.py` to import `UIElement` and `InteractionTurn` from `jed_m2.core.types` and delegate `SyntheticTrajectoryGenerator.compute_obs_features` to `jed_m2.core.features.compute_obs_features`; verify synthetic batch generation and collation via `python verify_phase4.py`.
- [x] 2.2 Refactor `jed_m2/environment.py` to import canonical `UIElement` from `jed_m2.core.types`, preserving existing environment states and action dispatch; verify simulated sandbox transitions and goal evaluation via `python verify_phase6a.py`.

## 3. Real Web & Preference Dataset Integration

- [x] 3.1 Refactor `jed_m2/real_dataset.py` to import `UIElement` from `jed_m2.core.types`, alias `InteractiveElement = UIElement`, and delegate `RealWebTrajectoryDataset.compute_obs_features` to `jed_m2.core.features.compute_obs_features`; verify HTML parsing, normalized coordinates, and disk caching via `python verify_phase12.py`.
- [x] 3.2 Refactor `jed_m2/real_preference_dataset.py` to import canonical types from `jed_m2.core.types`, aliasing `PreferenceTurn = InteractionTurn` and `InteractiveElement = UIElement`; verify preference pairing and perturbation injection.

## 4. Runtime Cleanup & Full Regression Suite

- [x] 4.1 Refactor `jed_m2/agent.py` to import `compute_obs_features` statically from `jed_m2.core.features`, eliminating the inline dynamic import of `RealWebTrajectoryDataset`; verify interactive agent decision rollout with and without JEPA planning.
- [x] 4.2 Update `jed_m2/__init__.py` to re-export `compute_obs_features` and unified types while maintaining 100% backward compatibility for all legacy symbols; verify system integrity by executing `pytest test_jepa_regression.py` and `python verify_phase14.py`.

## Workflow follow-up

- Review proposal, specs, design, and tasks with the user.
- Execute `/openspec-apply-change` to implement the tasks.
