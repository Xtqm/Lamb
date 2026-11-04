# Design: Consolidate Core Types and Observation Feature Encoders

## Context

During the rapid construction of JED-M2 across 14 phases, foundational abstractions were repeatedly re-implemented in place:
1. `UIElement` in `jed_m2/dataset.py` (synthetic dataset).
2. `UIElement` in `jed_m2/environment.py` (simulated WebOS sandbox).
3. `InteractiveElement` in `jed_m2/real_dataset.py` (real HTML parser for Mind2Web).
4. `InteractionTurn` in `jed_m2/dataset.py`, `RealWebInteractionTurn` in `jed_m2/real_dataset.py`, and `PreferenceTurn` in `jed_m2/real_preference_dataset.py`.
5. `compute_obs_features` (MD5 hashing + coordinate bucketing) duplicated verbatim in `dataset.py` and `real_dataset.py`, leading to an inline runtime import in `jed_m2/agent.py`.

See `proposal.md` for problem motivation and business drivers.

## Goals / Non-Goals

**Goals:**
- Provide a single canonical `UIElement` dataclass in `jed_m2/core/types.py` that accommodates all properties needed by synthetic environments, simulated sandboxes, and real HTML DOM trees.
- Provide a single canonical `InteractionTurn` container in `jed_m2/core/types.py` accommodating multimodal features, optional screenshot tensors, and UI elements.
- Centralize semantic observation hashing into `jed_m2/core/features.py` as `compute_obs_features`.
- Retain complete backward compatibility: All legacy import paths (`from jed_m2.dataset import UIElement`, `from jed_m2.real_dataset import InteractiveElement, RealWebInteractionTurn`, etc.) continue to work via module-level aliases.
- Clean up `jed_m2/agent.py` to import `compute_obs_features` statically at module level.
- Ensure all existing verification suites pass with zero regressions.

**Non-Goals:**
- Modifying neural architecture or parameter weight matrices (`JEDM2Model`, `MambaBlock`, `MLA`, `MoE`, `heads.py`).
- Refactoring training scripts (`train.py`, `train_pretrain.py`) or browser wrappers (`browser.py`, `playwright_env.py`) — these belong to subsequent planned refactoring phases.
- Changing serialized checkpoint file formats or disk cache schemas.

## Decisions

### Decision 1: Unified `UIElement` Dataclass
Instead of maintaining parallel classes or an inheritance hierarchy, implement a single comprehensive `UIElement` dataclass in `jed_m2/core/types.py`:
```python
@dataclass
class UIElement:
    element_id: int
    tag: str
    role: str
    text: str
    bbox: Tuple[float, float, float, float]  # Normalized [x_min, y_min, x_max, y_max] in [0, 1]
    value: str = ""
    is_clickable: bool = True
    is_input: bool = False
    is_scrollable: bool = False
    is_focused: bool = False
    attributes: Dict[str, str] = field(default_factory=dict)
```
- Includes helper properties: `center`, `width`, `height`.
- Includes DOM serialization: `to_dom_string()`.
- Alias: `InteractiveElement = UIElement`.

*Alternative Considered*: Subclassing `UIElement` from an abstract base class. Rejected because dataclass inheritance in Python can complicate default argument ordering and adds unnecessary complexity for plain data containers.

### Decision 2: Unified `InteractionTurn` Dataclass
Consolidate turn representations into `jed_m2/core/types.py`:
```python
@dataclass
class InteractionTurn:
    turn_id: int
    obs_text: str
    elements: List[UIElement]
    thought: str
    action_type: ActionType
    coordinates: Tuple[float, float]
    argument: str
    reward: float
    done: bool
    obs_features: torch.Tensor
    screenshot: Optional[torch.Tensor] = None
```
- Aliases: `RealWebInteractionTurn = InteractionTurn`, `PreferenceTurn = InteractionTurn`.

### Decision 3: Dedicated Feature Module (`jed_m2/core/features.py`)
Move `compute_obs_features` out of `RealWebTrajectoryDataset` and `dataset.py` into a standalone, pure function:
```python
def compute_obs_features(
    obs_text: str,
    elements: List[UIElement],
    hidden_dim: int,
) -> torch.Tensor:
```
- In `RealWebTrajectoryDataset` and `SyntheticTrajectoryGenerator`, maintain static methods delegating to this function for backward compatibility:
  `compute_obs_features = staticmethod(compute_obs_features)`.

### Decision 4: Module Dependency Order & Hygiene
Ensure `jed_m2/core` does not introduce circular imports:
```
  [jed_m2.heads (ActionType)]
              ^
              |
  [jed_m2.core.types] <---+
              ^           |
              |           |
  [jed_m2.core.features] -+
              ^
              +-- [jed_m2.dataset]
              +-- [jed_m2.real_dataset]
              +-- [jed_m2.environment]
              +-- [jed_m2.agent]
```
`jed_m2/core/types.py` only imports `ActionType` from `jed_m2.heads` and standard PyTorch / dataclass types.

## Risks / Trade-offs

- **[Risk] Existing Pickled Objects or Cache Files**
  - *Risk*: If previously saved `.pt` trajectory caches reference `jed_m2.real_dataset.InteractiveElement`, unpickling might fail if the symbol is moved.
  - *Mitigation*: Re-export `InteractiveElement = UIElement` directly inside `jed_m2/real_dataset.py` and `jed_m2/__init__.py`. PyTorch/Python will resolve the symbol transparently.

- **[Risk] Parameter Signature Mismatch in Call Sites**
  - *Risk*: Code instantiating `UIElement` with positional arguments might break if field order changes.
  - *Mitigation*: Strictly preserve the exact field order of the primary 5 fields (`element_id`, `tag`, `role`, `text`, `bbox`) matching both legacy `UIElement` and `InteractiveElement` definitions. All additional fields carry default values.

- **[Risk] Circular Imports in Runtime**
  - *Risk*: `agent.py` importing `features` and `model.py` importing `agent.py`.
  - *Mitigation*: `core/` modules are strictly low-level and import nothing from `agent.py` or `model.py`.

## Migration Plan

1. Create `jed_m2/core/__init__.py`, `jed_m2/core/types.py`, and `jed_m2/core/features.py`.
2. Refactor `jed_m2/dataset.py`: import `UIElement`, `InteractionTurn` from `jed_m2.core.types`, delegate `compute_obs_features`.
3. Refactor `jed_m2/environment.py`: import `UIElement` from `jed_m2.core.types`.
4. Refactor `jed_m2/real_dataset.py`: import `UIElement` and `InteractionTurn` from `jed_m2.core.types`, re-export aliases, delegate `compute_obs_features`.
5. Refactor `jed_m2/real_preference_dataset.py`: import and re-export aliases.
6. Clean up `jed_m2/agent.py`: remove the inline dynamic import on line 393, import `compute_obs_features` at top level.
7. Update `jed_m2/__init__.py` to export unified symbols.
8. Execute regression tests to verify 100% test pass rate.
