"""
JED-M2 Rule-Based Decision Tree (RDTree) Process Critic.

Provides:
1. Hierarchical symbolic decision tree for step-level action affordance evaluation.
2. Bounded scalar process reward scoring R_RDTree in [-1.0, 1.0].
3. Diagnostic classification: VALID_INTERACTION, NOOP_CLICK, UNFOCUSED_INPUT,
   READONLY_FIELD, CYCLIC_ACTION, OFFSCREEN_COORDINATE.
4. Rolling history oscillation detection and critical violation pre-pruning.

STRICT CONVENTION: 100% native PyTorch/Python without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from jed_m2.core.types import UIElement
from jed_m2.heads import ActionType


class RDTreeDiagnostic(str, Enum):
    """Diagnostic categorization of action affordance and process validity."""

    VALID_INTERACTION = "VALID_INTERACTION"
    OFFSCREEN_COORDINATE = "OFFSCREEN_COORDINATE"
    NOOP_CLICK = "NOOP_CLICK"
    INERT_ELEMENT_CLICK = "INERT_ELEMENT_CLICK"
    UNFOCUSED_INPUT = "UNFOCUSED_INPUT"
    READONLY_FIELD = "READONLY_FIELD"
    INVALID_SELECT_TARGET = "INVALID_SELECT_TARGET"
    CYCLIC_ACTION = "CYCLIC_ACTION"
    REDUNDANT_MUTATION = "REDUNDANT_MUTATION"
    NAVIGATE_OK = "NAVIGATE_OK"
    SCROLL_OK = "SCROLL_OK"
    FINISH_OK = "FINISH_OK"


@dataclass
class RDTreeEvaluationResult:
    """Feedback returned from RDTree symbolic decision evaluation."""

    process_reward: float  # Scalar in [-1.0, 1.0]
    diagnostic: RDTreeDiagnostic
    target_element: Optional[UIElement] = None
    target_handle: Optional[str] = None
    is_critical_violation: bool = False
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "process_reward": round(self.process_reward, 3),
            "diagnostic": self.diagnostic.value,
            "target_handle": self.target_handle,
            "is_critical_violation": self.is_critical_violation,
            "message": self.message,
        }


class RDTreeProcessCritic:
    """
    Symbolic Rule-Based Decision Tree Critic providing step-level process supervision.
    Evaluates proposed actions against physical DOM element affordances, spatial
    bounds, and recent interaction history in <2 ms.
    """

    def __init__(
        self,
        history_capacity: int = 8,
        critical_threshold: float = -0.8,
    ):
        self.history_capacity = history_capacity
        self.critical_threshold = critical_threshold
        # Stores recent (action_type, element_id, (cx, cy)) tuples
        self._history: deque = deque(maxlen=history_capacity)
        self._stats: Dict[str, int] = {
            "evaluations": 0,
            "critical_violations": 0,
            "noop_clicks": 0,
            "cycles_detected": 0,
            "valid_promotions": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset(self) -> None:
        """Flushes history on new scenario or page navigation."""
        self._history.clear()

    def record_step(
        self,
        action_type: Union[ActionType, int],
        element_id: Optional[int] = None,
        coordinates: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Records executed step in rolling history ring buffer."""
        act = ActionType(action_type) if isinstance(action_type, int) else action_type
        self._history.append((act, element_id, coordinates))

    def evaluate_action(
        self,
        action_type: Union[ActionType, int],
        coordinates: Tuple[float, float],
        argument: Optional[str] = None,
        elements: Optional[List[UIElement]] = None,
    ) -> RDTreeEvaluationResult:
        """
        Evaluates a candidate action through the hierarchical symbolic decision tree:
        Node 1: Coordinate bounds check.
        Node 2: Element target resolution.
        Node 3: Affordance compatibility (CLICK, TYPE, SELECT).
        Node 4: History cycle check.
        """
        self._stats["evaluations"] += 1
        act = ActionType(action_type) if isinstance(action_type, int) else action_type
        x, y = coordinates

        # Node 1: Coordinate Bounds Check
        if act in (ActionType.CLICK, ActionType.SELECT):
            if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                self._stats["critical_violations"] += 1
                return RDTreeEvaluationResult(
                    process_reward=-1.0,
                    diagnostic=RDTreeDiagnostic.OFFSCREEN_COORDINATE,
                    is_critical_violation=True,
                    message=f"Coordinates ({x:.2f}, {y:.2f}) outside [0, 1]",
                )

        # Node 2: Target Element Resolution
        target_el: Optional[UIElement] = None
        target_handle: Optional[str] = None

        if elements:
            # 2a. Match by symbolic handle if argument or handle references it
            if argument and argument.startswith("$NODE_"):
                for el in elements:
                    if el.node_handle == argument or f"$NODE_{el.element_id}" == argument:
                        target_el = el
                        target_handle = el.node_handle
                        break

            # 2b. Spatial containment match
            if target_el is None and act in (ActionType.CLICK, ActionType.SELECT, ActionType.TYPE):
                candidates = [el for el in elements if el.contains(x, y)]
                if candidates:
                    # Choose smallest area (most specific child element)
                    target_el = min(candidates, key=lambda el: el.area)
                    target_handle = target_el.node_handle or f"$NODE_{target_el.element_id}"

        # Node 3: Affordance Compatibility
        if act == ActionType.CLICK:
            if target_el is None:
                self._stats["noop_clicks"] += 1
                self._stats["critical_violations"] += 1
                return RDTreeEvaluationResult(
                    process_reward=-0.8,
                    diagnostic=RDTreeDiagnostic.NOOP_CLICK,
                    is_critical_violation=True,
                    message=f"Click at ({x:.2f}, {y:.2f}) hits non-interactive whitespace",
                )

            if not target_el.is_clickable:
                # Check if attributes indicate clickable behavior
                has_click_attr = any(
                    k in ("onclick", "role", "data-action") for k in target_el.attributes
                )
                if has_click_attr or target_el.role in ("button", "link", "tab", "menuitem"):
                    return RDTreeEvaluationResult(
                        process_reward=0.2,
                        diagnostic=RDTreeDiagnostic.VALID_INTERACTION,
                        target_element=target_el,
                        target_handle=target_handle,
                        message="Click targeting container with interactive role/attribute",
                    )
                return RDTreeEvaluationResult(
                    process_reward=-0.6,
                    diagnostic=RDTreeDiagnostic.INERT_ELEMENT_CLICK,
                    target_element=target_el,
                    target_handle=target_handle,
                    is_critical_violation=False,
                    message=f"Element <{target_el.tag}> is not marked clickable",
                )

            # Node 4: History Cycle Check for clicks
            if self._history:
                last_act, last_eid, last_coords = self._history[-1]
                if last_act == ActionType.CLICK and last_eid == target_el.element_id:
                    self._stats["cycles_detected"] += 1
                    self._stats["critical_violations"] += 1
                    return RDTreeEvaluationResult(
                        process_reward=-1.0,
                        diagnostic=RDTreeDiagnostic.CYCLIC_ACTION,
                        target_element=target_el,
                        target_handle=target_handle,
                        is_critical_violation=True,
                        message=f"Repeated identical click on element {target_el.element_id} without state change",
                    )

            self._stats["valid_promotions"] += 1
            return RDTreeEvaluationResult(
                process_reward=0.8,
                diagnostic=RDTreeDiagnostic.VALID_INTERACTION,
                target_element=target_el,
                target_handle=target_handle,
                is_critical_violation=False,
                message=f"Valid click on interactive <{target_el.tag}> [{target_el.role}]",
            )

        elif act == ActionType.TYPE:
            if target_el is None or not target_el.is_input:
                self._stats["critical_violations"] += 1
                return RDTreeEvaluationResult(
                    process_reward=-0.9,
                    diagnostic=RDTreeDiagnostic.UNFOCUSED_INPUT,
                    target_element=target_el,
                    target_handle=target_handle,
                    is_critical_violation=True,
                    message="Type action proposed without valid input field target",
                )

            # Check read-only / disabled attributes
            if target_el.attributes.get("readonly") == "true" or target_el.attributes.get("disabled") == "true":
                self._stats["critical_violations"] += 1
                return RDTreeEvaluationResult(
                    process_reward=-0.8,
                    diagnostic=RDTreeDiagnostic.READONLY_FIELD,
                    target_element=target_el,
                    target_handle=target_handle,
                    is_critical_violation=True,
                    message="Type action proposed on read-only or disabled input",
                )

            self._stats["valid_promotions"] += 1
            return RDTreeEvaluationResult(
                process_reward=0.9,
                diagnostic=RDTreeDiagnostic.VALID_INTERACTION,
                target_element=target_el,
                target_handle=target_handle,
                is_critical_violation=False,
                message=f"Valid typing targeting input <{target_el.tag}> [{target_el.role}]",
            )

        elif act == ActionType.SELECT:
            if target_el and (
                target_el.tag == "select" or target_el.role in ("combobox", "option", "dropdown")
            ):
                self._stats["valid_promotions"] += 1
                return RDTreeEvaluationResult(
                    process_reward=0.8,
                    diagnostic=RDTreeDiagnostic.VALID_INTERACTION,
                    target_element=target_el,
                    target_handle=target_handle,
                    is_critical_violation=False,
                    message=f"Valid selection on <{target_el.tag}>",
                )
            return RDTreeEvaluationResult(
                process_reward=-0.7,
                diagnostic=RDTreeDiagnostic.INVALID_SELECT_TARGET,
                target_element=target_el,
                target_handle=target_handle,
                is_critical_violation=False,
                message="Select action proposed on non-selectable element",
            )

        elif act == ActionType.NAVIGATE:
            return RDTreeEvaluationResult(
                process_reward=0.2,
                diagnostic=RDTreeDiagnostic.NAVIGATE_OK,
                message="Navigation action proposal",
            )

        elif act == ActionType.SCROLL:
            return RDTreeEvaluationResult(
                process_reward=0.3,
                diagnostic=RDTreeDiagnostic.SCROLL_OK,
                message="Scroll viewport action proposal",
            )

        elif act == ActionType.FINISH:
            return RDTreeEvaluationResult(
                process_reward=0.5,
                diagnostic=RDTreeDiagnostic.FINISH_OK,
                message="Task completion declaration",
            )

        return RDTreeEvaluationResult(
            process_reward=0.0,
            diagnostic=RDTreeDiagnostic.VALID_INTERACTION,
            message=f"ActionType {act.name} evaluated neutral",
        )
