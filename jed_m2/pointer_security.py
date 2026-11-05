"""
JED-M2 Symbolic Pointer Security and Information-Flow Control (IFC).

Provides:
1. SymbolicPointer and SymbolicPointerTable: Maps untrusted DOM strings to opaque handles
   ($NODE_<id>, $TEXT_<hash>), storing raw text out-of-band in an isolated register.
2. CapabilityMask: Pre-compiled immutable policy bounding permissible action types,
   target domains/URLs, sensitive state mutations, and cross-origin redirection.
3. Policy verification and boundary de-referencing for secure closed-loop agent execution.

STRICT CONVENTION: 100% native PyTorch/Python without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

from jed_m2.heads import ActionType


@dataclass
class SymbolicPointer:
    """Opaque symbolic pointer handle referencing an in-browser DOM element or text."""

    handle: str  # e.g., "$NODE_1", "$TEXT_8f3a"
    pointer_type: str  # 'node', 'text', 'slot'
    element_id: Optional[int] = None
    hash_digest: str = ""
    tag: str = ""
    role: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None
    center: Optional[Tuple[float, float]] = None
    raw_text: str = ""  # Maintained strictly in out-of-band register
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CapabilityMask:
    """
    Immutable action capability policy compiled from trusted user instructions.
    Restricts allowed action primitives, target domains, and blocked mutations.
    """

    allowed_actions: Set[ActionType] = field(
        default_factory=lambda: {
            ActionType.CLICK,
            ActionType.TYPE,
            ActionType.SCROLL,
            ActionType.NAVIGATE,
            ActionType.FINISH,
            ActionType.SELECT,
        }
    )
    allowed_domains: Set[str] = field(default_factory=lambda: {"*"})  # '*' means unrestricted
    blocked_endpoints: Set[str] = field(
        default_factory=lambda: {
            "/admin/delete",
            "/api/reset",
            "/logout",
            "/danger",
            "/transfer",
            "/exfiltrate",
        }
    )
    allow_navigation: bool = True
    allow_downloads: bool = False
    max_steps_per_domain: int = 100

    def is_action_allowed(self, action_type: Union[ActionType, int]) -> bool:
        """Checks if discrete action primitive is permitted under current policy."""
        try:
            act = ActionType(action_type)
            return act in self.allowed_actions
        except ValueError:
            return False

    def is_domain_allowed(self, target_url: str) -> bool:
        """Checks if destination URL or origin is in the allowed domain set."""
        if "*" in self.allowed_domains:
            return True
        if not target_url:
            return True
        try:
            parsed = urlparse(target_url)
            domain = parsed.netloc.lower() or parsed.path.split("/")[0].lower()
            if ":" in domain:
                domain = domain.split(":")[0]
            for allowed in self.allowed_domains:
                allowed_clean = allowed.lower()
                if domain == allowed_clean or domain.endswith("." + allowed_clean):
                    return True
            return False
        except Exception:
            return False

    def is_endpoint_allowed(self, target_url: str) -> bool:
        """Checks if URL path hits a blocked sensitive or destructive endpoint."""
        if not target_url:
            return True
        try:
            parsed = urlparse(target_url)
            path = parsed.path.lower()
            for blocked in self.blocked_endpoints:
                if blocked.lower() in path:
                    return False
            return True
        except Exception:
            return True

    def validate_action(
        self,
        action_type: Union[ActionType, int],
        target_url: Optional[str] = None,
        target_handle: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates an emitted action against policy invariants.
        Returns (is_allowed, violation_reason).
        """
        try:
            act = ActionType(action_type) if isinstance(action_type, int) else action_type
        except ValueError:
            return False, f"Unknown action type {action_type}"

        if act == ActionType.NAVIGATE and not self.allow_navigation:
            return False, "Navigation actions are disabled by CapabilityMask"

        if not self.is_action_allowed(act):
            return False, f"ActionType {act.name} is not permitted by CapabilityMask"

        if target_url:
            if not self.is_domain_allowed(target_url):
                return False, f"Destination domain for '{target_url}' is not in allowed_domains"
            if not self.is_endpoint_allowed(target_url):
                return False, f"Endpoint '{target_url}' matches blocked sensitive endpoint"

        return True, None


class SymbolicPointerTable:
    """
    Out-of-band registry mapping DOM elements, strings, and trusted input slots
    to opaque symbolic identifiers. Prevents untrusted text from entering model vocabulary.
    """

    def __init__(self):
        self._nodes: Dict[str, SymbolicPointer] = {}
        self._texts: Dict[str, str] = {}  # text_handle -> raw_text
        self._slots: Dict[str, str] = {}  # slot_name -> secret/trusted value
        self._spatial_index: Dict[Tuple[int, int], str] = {}  # quantized (x, y) -> node_handle
        self._stats: Dict[str, int] = {
            "registered_nodes": 0,
            "registered_texts": 0,
            "bound_slots": 0,
            "resolved_handles": 0,
            "blocked_violations": 0,
        }

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def register_text(self, raw_text: str) -> str:
        """
        Registers raw untrusted string into out-of-band ledger.
        Returns opaque handle '$TEXT_<hash>'.
        """
        if not raw_text:
            return "$TEXT_empty"
        digest = hashlib.sha256(raw_text.encode("utf-8", errors="ignore")).hexdigest()[:8]
        handle = f"$TEXT_{digest}"
        self._texts[handle] = raw_text
        self._stats["registered_texts"] += 1
        return handle

    def register_element(
        self,
        element_id: int,
        tag: str,
        role: str,
        raw_text: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        attributes: Optional[Dict[str, str]] = None,
    ) -> SymbolicPointer:
        """Allocates or updates a symbolic pointer for an interactive DOM element."""
        handle = f"$NODE_{element_id}"
        text_handle = self.register_text(raw_text)

        center = None
        if bbox and len(bbox) >= 4:
            center = (round((bbox[0] + bbox[2]) / 2.0, 4), round((bbox[1] + bbox[3]) / 2.0, 4))
            qx = int(center[0] * 100)
            qy = int(center[1] * 100)
            self._spatial_index[(qx, qy)] = handle

        pointer = SymbolicPointer(
            handle=handle,
            pointer_type="node",
            element_id=element_id,
            hash_digest=text_handle.replace("$TEXT_", ""),
            tag=tag,
            role=role,
            bbox=bbox,
            center=center,
            raw_text=raw_text,
            metadata=attributes or {},
        )
        self._nodes[handle] = pointer
        self._stats["registered_nodes"] += 1
        return pointer

    def register_slot(self, slot_name: str, value: str) -> str:
        """
        Registers trusted user input data into a secured slot (e.g. 'search_query', 'email').
        Returns handle '$SLOT_<slot_name>'.
        """
        handle = f"$SLOT_{slot_name}"
        self._slots[slot_name] = value
        self._stats["bound_slots"] += 1
        return handle

    def get_slot_value(self, slot_name: str) -> Optional[str]:
        """Retrieves raw trusted value for a registered slot."""
        key = slot_name[6:] if slot_name.startswith("$SLOT_") else slot_name
        return self._slots.get(key, None)

    def resolve_handle(self, handle: str) -> Optional[SymbolicPointer]:
        """Resolves handle to SymbolicPointer."""
        self._stats["resolved_handles"] += 1
        return self._nodes.get(handle, None)

    def resolve_center(self, handle: str) -> Optional[Tuple[float, float]]:
        """Resolves handle to normalized center coordinates."""
        ptr = self.resolve_handle(handle)
        if ptr and ptr.center is not None:
            return ptr.center
        return None

    def resolve_text(self, text_handle: str) -> Optional[str]:
        """Resolves text handle back to raw text (for inspection or debug only)."""
        return self._texts.get(text_handle, None)

    def find_nearest_handle(self, x: float, y: float, max_dist: float = 0.25) -> Optional[str]:
        """Finds nearest registered node handle to a continuous coordinate pair (x, y)."""
        best_handle = None
        best_dist = float("inf")
        for handle, ptr in self._nodes.items():
            if ptr.center is not None:
                cx, cy = ptr.center
                dist = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
                if dist < best_dist and dist <= max_dist:
                    best_dist = dist
                    best_handle = handle
        return best_handle

    def sanitize_observation_text(self, text: str) -> str:
        """
        Replaces potential untrusted injection strings in observation representations
        with symbolic pointer representations.
        """
        return self.register_text(text)

    def clear(self):
        """Flushes transient nodes on document navigation."""
        self._nodes.clear()
        self._texts.clear()
        self._spatial_index.clear()
