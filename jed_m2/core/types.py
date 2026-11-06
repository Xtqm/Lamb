"""
JED-M2 Core Foundation Types.
Canonical data structures representing interactable DOM/GUI elements and interaction trajectory turns.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from jed_m2.heads import ActionType


@dataclass
class UIElement:
    """
    Canonical interactable DOM / GUI element with spatial bounding box and attributes.
    Supports synthetic layout trees, simulated sandboxes, and real-world parsed HTML DOMs.
    Normalized coordinates: [x_min, y_min, x_max, y_max] in [0, 1].
    """

    element_id: int
    tag: str  # e.g., 'button', 'input', 'a', 'select', 'div', 'textarea'
    role: str  # e.g., 'search_button', 'text_input', 'nav_link', 'card'
    text: str  # Inner text, placeholder, or aria-label
    bbox: Tuple[float, float, float, float]  # Normalized (x_min, y_min, x_max, y_max) in [0, 1]
    value: str = ""  # Current typed or selected value
    is_clickable: bool = True
    is_input: bool = False
    is_scrollable: bool = False
    is_focused: bool = False
    attributes: Dict[str, str] = field(default_factory=dict)
    has_bbox: bool = True  # False when the source observation did not provide geometry
    node_handle: str = ""  # Symbolic pointer handle (e.g., "$NODE_1")
    text_handle: str = ""  # Symbolic text digest handle (e.g., "$TEXT_7b8a")

    @property
    def center(self) -> Tuple[float, float]:
        """Returns normalized center pointer coordinates (x, y) in [0, 1]."""
        x_min, y_min, x_max, y_max = self.bbox
        return ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

    @property
    def width(self) -> float:
        """Normalized width in [0, 1]."""
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        """Normalized height in [0, 1]."""
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        """Normalized bounding box area."""
        return self.width * self.height

    def contains(self, x: float, y: float) -> bool:
        """Checks if screen coordinate (x, y) falls inside this element's bounding box."""
        x_min, y_min, x_max, y_max = self.bbox
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    def to_dom_string(self, drop_id_class: bool = False) -> str:
        """Serializes element to concise accessibility DOM representation."""
        cx, cy = self.center
        clean_text = self.text.replace("\n", " ").strip()
        if len(clean_text) > 40:
            clean_text = clean_text[:37] + "..."
        attrs = ""
        drop = drop_id_class or bool(self.attributes and self.attributes.get("__drop_id_class__"))
        if not drop:
            attrs += f" id={self.element_id}"
            if self.attributes and "class" in self.attributes:
                attrs += f" class='{self.attributes['class']}'"
        if self.attributes:
            if "data-testid" in self.attributes:
                attrs += f" data-testid='{self.attributes['data-testid']}'"
            if "aria-label" in self.attributes:
                attrs += f" aria-label='{self.attributes['aria-label']}'"
            if "type" in self.attributes:
                attrs += f" type='{self.attributes['type']}'"
        if self.value:
            attrs += f" value='{self.value}'"
        if self.is_focused:
            attrs += " focused='true'"
        bbox_text = f"[{cx:.2f},{cy:.2f}]" if self.has_bbox else "unavailable"
        return (
            f"<{self.tag}{attrs} role='{self.role}' "
            f"bbox='{bbox_text}'>{clean_text}</{self.tag}>"
        )

    def to_symbolic_string(self) -> str:
        """Serializes element using opaque symbolic handles to prevent indirect prompt injection."""
        cx, cy = self.center
        h = self.node_handle or f"$NODE_{self.element_id}"
        t = self.text_handle or "$TEXT_bound"
        attrs = ""
        if self.attributes and "type" in self.attributes:
            attrs += f" type='{self.attributes['type']}'"
        if self.is_focused:
            attrs += " focused='true'"
        bbox_text = f"[{cx:.2f},{cy:.2f}]" if self.has_bbox else "unavailable"
        return (
            f"<{self.tag} handle='{h}' role='{self.role}' "
            f"bbox='{bbox_text}'{attrs} text_handle='{t}'></{self.tag}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts element attributes to dictionary."""
        return {
            "element_id": self.element_id,
            "tag": self.tag,
            "role": self.role,
            "text": self.text,
            "bbox": self.bbox,
            "value": self.value,
            "is_clickable": self.is_clickable,
            "is_input": self.is_input,
            "is_scrollable": self.is_scrollable,
            "is_focused": self.is_focused,
            "center": self.center,
            "has_bbox": self.has_bbox,
        }


# Backward compatibility alias for real HTML dataset callers
InteractiveElement = UIElement


@dataclass
class InteractionTurn:
    """
    Canonical representation of a single turn step in an agent interaction trajectory.
    Supports synthetic multi-turn episodes, real HTML web traces, and preference rollouts.
    """

    turn_id: int
    obs_text: str
    thought: str
    action_type: ActionType
    coordinates: Tuple[float, float]
    argument: str
    dom_elements: List[UIElement] = field(default_factory=list)
    obs_features: Optional[torch.Tensor] = None  # Dense observation embedding [hidden_dim]
    reward: float = 0.0  # Environment reward signal
    screenshot: Optional[torch.Tensor] = None  # Viewport screenshot tensor [3, H, W]
    url: str = ""  # URL for browser episodes
    target_element_id: Optional[int] = None
    is_perturbed: bool = False  # RL preference perturbation marker
    interactive_elements: Optional[List[UIElement]] = None
    rationale_mask: bool = True  # Whether thought text is recorded supervision
    action_type_mask: bool = True  # Whether the source operation maps unambiguously to a class
    coordinate_mask: bool = True  # Whether the source provides valid spatial supervision
    target_element_mask: bool = True  # Whether a target element is uniquely identified
    argument_mask: bool = True  # Whether the source provides an argument target
    source_operation: Optional[str] = None  # Original source operation before class mapping

    def __post_init__(self):
        if self.interactive_elements is not None and not self.dom_elements:
            self.dom_elements = self.interactive_elements
        elif self.dom_elements and self.interactive_elements is None:
            self.interactive_elements = self.dom_elements

    @property
    def elements(self) -> List[UIElement]:
        """Unified alias for accessing UI elements in the turn."""
        return self.dom_elements


# Backward compatibility aliases for real web and preference datasets
RealWebInteractionTurn = InteractionTurn
PreferenceTurn = InteractionTurn
