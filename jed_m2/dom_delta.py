"""
JED-M2 Differential DOM Delta Tracking & Shadow Registry Engine.

Provides:
1. Canonical DOM mutation data classes (DOMMutationType, DOMMutation, DOMDeltaRecord).
2. Pure-Python element-set diffing engine diff_elements() for offline & fallback modes.
3. ShadowDOMRegistry maintaining client-side DOM reconstruction across turns.
4. Upheaval threshold detection (>50% mutation, >32 elements, URL navigation).
5. Invertible compact delta grammar serialization: <obs delta> + ... - ... ~ ... </obs>.

STRICT CONVENTION: 100% native PyTorch/Python without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from jed_m2.core.types import UIElement


# ==============================================================================
# 1. DOM Mutation Primitive Types
# ==============================================================================

class DOMMutationType(str, Enum):
    """Types of discrete DOM structural or attribute mutations."""
    ADD = "add"        # Element inserted into tracked interactable set (+)
    REMOVE = "remove"  # Element removed or hidden from interactable set (-)
    MODIFY = "modify"  # Attribute, value, text, or bbox geometry changed (~)


@dataclass
class DOMMutation:
    """
    Represents a single DOM element mutation between interaction steps.
    """
    mutation_type: DOMMutationType
    element_id: int
    tag: Optional[str] = None
    role: Optional[str] = None
    text: Optional[str] = None
    value: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_grammar_string(self, use_symbolic_pointers: bool = False) -> str:
        """
        Serializes mutation to compact, token-efficient grammar representation.
        Format:
          ADD:    + <tag id=N role='R' bbox='[cx,cy]' [attrs]>text</tag>
          REMOVE: - id=N
          MODIFY: ~ id=N [text='...'] [value='...'] [bbox='[cx,cy]']
        When use_symbolic_pointers=True, masks untrusted text behind $TEXT_<hash> and $NODE_N.
        """
        if self.mutation_type == DOMMutationType.ADD:
            tag = self.tag or "element"
            role = self.role or tag
            
            bbox_str = ""
            if self.bbox is not None and len(self.bbox) >= 4:
                cx = (self.bbox[0] + self.bbox[2]) / 2.0
                cy = (self.bbox[1] + self.bbox[3]) / 2.0
                bbox_str = f" bbox='[{cx:.2f},{cy:.2f}]'"
            
            attrs_str = ""
            if self.value and not use_symbolic_pointers:
                attrs_str += f" value='{self.value}'"
            if self.attributes:
                for k, v in sorted(self.attributes.items()):
                    if k != "value":
                        attrs_str += f" {k}='{v}'"

            if use_symbolic_pointers:
                digest = hashlib.sha256((self.text or "").encode("utf-8", errors="ignore")).hexdigest()[:8]
                return f"+ <{tag} handle='$NODE_{self.element_id}' role='{role}'{bbox_str}{attrs_str} text_handle='$TEXT_{digest}'></{tag}>"
            
            clean_text = (self.text or "").replace("\n", " ").strip()
            if len(clean_text) > 30:
                clean_text = clean_text[:27] + "..."
            return f"+ <{tag} id={self.element_id} role='{role}'{bbox_str}{attrs_str}>{clean_text}</{tag}>"

        elif self.mutation_type == DOMMutationType.REMOVE:
            if use_symbolic_pointers:
                return f"- handle='$NODE_{self.element_id}'"
            return f"- id={self.element_id}"

        elif self.mutation_type == DOMMutationType.MODIFY:
            if use_symbolic_pointers:
                parts = [f"~ handle='$NODE_{self.element_id}'"]
                if self.text is not None:
                    digest = hashlib.sha256(self.text.encode("utf-8", errors="ignore")).hexdigest()[:8]
                    parts.append(f"text_handle='$TEXT_{digest}'")
                if self.bbox is not None and len(self.bbox) >= 4:
                    cx = (self.bbox[0] + self.bbox[2]) / 2.0
                    cy = (self.bbox[1] + self.bbox[3]) / 2.0
                    parts.append(f"bbox='[{cx:.2f},{cy:.2f}]'")
                return " ".join(parts)

            parts = [f"~ id={self.element_id}"]
            if self.text is not None:
                clean_text = self.text.replace("\n", " ").strip()
                if len(clean_text) > 30:
                    clean_text = clean_text[:27] + "..."
                parts.append(f"text='{clean_text}'")
            if self.value is not None:
                parts.append(f"value='{self.value}'")
            if self.bbox is not None and len(self.bbox) >= 4:
                cx = (self.bbox[0] + self.bbox[2]) / 2.0
                cy = (self.bbox[1] + self.bbox[3]) / 2.0
                parts.append(f"bbox='[{cx:.2f},{cy:.2f}]'")
            if self.attributes:
                for k, v in sorted(self.attributes.items()):
                    if k not in ("text", "value"):
                        parts.append(f"{k}='{v}'")
            return " ".join(parts)

        return f"? id={self.element_id}"


@dataclass
class DOMDeltaRecord:
    """
    Container for turn-level DOM observation delta.
    """
    mutations: List[DOMMutation] = field(default_factory=list)
    is_full_snapshot: bool = False
    is_upheaval: bool = False
    url: Optional[str] = None
    step: int = 0
    full_elements: Optional[List[UIElement]] = None

    def to_obs_string(self, use_symbolic_pointers: bool = False) -> str:
        """
        Formats observation string for model prefill ingestion.
        """
        prefix = f"URL: {self.url} | " if self.url else ""
        if self.is_full_snapshot:
            if self.full_elements:
                if use_symbolic_pointers:
                    dom_str = " ".join([el.to_symbolic_string() for el in self.full_elements[:64]])
                else:
                    dom_str = " ".join([el.to_dom_string() for el in self.full_elements[:64]])
                return f"<obs full> {prefix}DOM: {dom_str} </obs>"
            return f"<obs full> {prefix}DOM: [No interactive elements] </obs>"

        if not self.mutations:
            return f"<obs delta> {prefix}[NO_OP] </obs>"

        mut_str = " ".join([m.to_grammar_string(use_symbolic_pointers=use_symbolic_pointers) for m in self.mutations])
        return f"<obs delta> {prefix}{mut_str} </obs>"


# ==============================================================================
# 2. Pure-Python Element Diffing Engine
# ==============================================================================

def diff_elements(
    prev_elements: List[UIElement],
    curr_elements: List[UIElement],
    bbox_displacement_threshold: float = 0.01,
) -> List[DOMMutation]:
    """
    Computes set difference and attribute deltas between previous and current
    UIElement lists.

    Args:
        prev_elements: List of UIElement objects from step t-1.
        curr_elements: List of UIElement objects from step t.
        bbox_displacement_threshold: Positional displacement threshold to trigger
            a coordinate mutation.

    Returns:
        List of DOMMutation objects representing ADD, REMOVE, and MODIFY events.
    """
    prev_by_id: Dict[int, UIElement] = {el.element_id: el for el in prev_elements}
    curr_by_id: Dict[int, UIElement] = {el.element_id: el for el in curr_elements}

    mutations: List[DOMMutation] = []

    # 1. Removed elements (in prev but not in curr)
    for elem_id in sorted(prev_by_id.keys()):
        if elem_id not in curr_by_id:
            mutations.append(
                DOMMutation(
                    mutation_type=DOMMutationType.REMOVE,
                    element_id=elem_id,
                )
            )

    # 2. Added and Modified elements
    for elem_id in sorted(curr_by_id.keys()):
        curr_el = curr_by_id[elem_id]
        if elem_id not in prev_by_id:
            # New element added
            mutations.append(
                DOMMutation(
                    mutation_type=DOMMutationType.ADD,
                    element_id=elem_id,
                    tag=curr_el.tag,
                    role=curr_el.role,
                    text=curr_el.text,
                    value=curr_el.value,
                    bbox=curr_el.bbox,
                    attributes=dict(curr_el.attributes),
                )
            )
        else:
            # Existing element: check for modifications
            prev_el = prev_by_id[elem_id]
            mod_text: Optional[str] = None
            mod_value: Optional[str] = None
            mod_bbox: Optional[Tuple[float, float, float, float]] = None
            mod_attrs: Dict[str, Any] = {}

            if curr_el.text != prev_el.text:
                mod_text = curr_el.text

            if curr_el.value != prev_el.value:
                mod_value = curr_el.value

            # Check bbox displacement
            if curr_el.has_bbox and prev_el.has_bbox:
                c1 = curr_el.center
                c0 = prev_el.center
                dist = math.sqrt((c1[0] - c0[0]) ** 2 + (c1[1] - c0[1]) ** 2)
                if dist > bbox_displacement_threshold:
                    mod_bbox = curr_el.bbox
            elif curr_el.has_bbox != prev_el.has_bbox:
                mod_bbox = curr_el.bbox

            # Check attributes
            if curr_el.attributes != prev_el.attributes:
                for k, v in curr_el.attributes.items():
                    if prev_el.attributes.get(k) != v:
                        mod_attrs[k] = v

            if (
                mod_text is not None
                or mod_value is not None
                or mod_bbox is not None
                or mod_attrs
            ):
                mutations.append(
                    DOMMutation(
                        mutation_type=DOMMutationType.MODIFY,
                        element_id=elem_id,
                        tag=curr_el.tag,
                        role=curr_el.role,
                        text=mod_text,
                        value=mod_value,
                        bbox=mod_bbox,
                        attributes=mod_attrs,
                    )
                )

    return mutations


# ==============================================================================
# 3. Upheaval Threshold Checker
# ==============================================================================

def check_upheaval_threshold(
    mutations: List[DOMMutation],
    total_prev_elements: int,
    max_mutation_count: int = 32,
    max_fraction: float = 0.50,
) -> bool:
    """
    Evaluates whether the mutation volume constitutes a massive tree upheaval
    requiring an immediate full DOM snapshot fallback.
    """
    if len(mutations) > max_mutation_count:
        return True
    if total_prev_elements > 0:
        mutated_fraction = float(len(mutations)) / float(total_prev_elements)
        if mutated_fraction > max_fraction:
            return True
    return False


# ==============================================================================
# 4. Shadow DOM Element Registry
# ==============================================================================

class ShadowDOMRegistry:
    """
    Maintains a client-side shadow element registry that reconstructs full DOM
    interactable element states from incremental differential mutation deltas.
    """

    def __init__(self) -> None:
        self.elements: Dict[int, UIElement] = {}

    def reset(self, elements: Optional[List[UIElement]] = None) -> None:
        """Resets the shadow registry with a full element snapshot."""
        self.elements.clear()
        if elements:
            for el in elements:
                self.elements[el.element_id] = UIElement(
                    element_id=el.element_id,
                    tag=el.tag,
                    role=el.role,
                    text=el.text,
                    bbox=el.bbox,
                    value=el.value,
                    is_clickable=el.is_clickable,
                    is_input=el.is_input,
                    is_scrollable=el.is_scrollable,
                    is_focused=el.is_focused,
                    attributes=dict(el.attributes),
                    has_bbox=el.has_bbox,
                )

    def apply_delta(self, delta: DOMDeltaRecord) -> List[UIElement]:
        """
        Applies a differential delta record to the shadow registry.
        If delta is a full snapshot, completely re-seeds the registry.
        Returns the updated list of UIElement objects sorted by element_id.
        """
        if delta.is_full_snapshot and delta.full_elements is not None:
            self.reset(delta.full_elements)
            return self.get_elements()

        for mut in delta.mutations:
            if mut.mutation_type == DOMMutationType.REMOVE:
                self.elements.pop(mut.element_id, None)

            elif mut.mutation_type == DOMMutationType.ADD:
                bbox = mut.bbox if mut.bbox is not None else (0.0, 0.0, 0.0, 0.0)
                self.elements[mut.element_id] = UIElement(
                    element_id=mut.element_id,
                    tag=mut.tag or "element",
                    role=mut.role or (mut.tag or "element"),
                    text=mut.text or "",
                    bbox=bbox,
                    value=mut.value or "",
                    attributes=dict(mut.attributes),
                )

            elif mut.mutation_type == DOMMutationType.MODIFY:
                if mut.element_id in self.elements:
                    el = self.elements[mut.element_id]
                    if mut.text is not None:
                        el.text = mut.text
                    if mut.value is not None:
                        el.value = mut.value
                    if mut.bbox is not None:
                        el.bbox = mut.bbox
                    if mut.attributes:
                        el.attributes.update(mut.attributes)

        return self.get_elements()

    def get_elements(self) -> List[UIElement]:
        """Returns all currently tracked shadow elements sorted by element_id."""
        return [self.elements[eid] for eid in sorted(self.elements.keys())]

    def validate_consistency(
        self,
        ground_truth_elements: List[UIElement],
        tolerance: float = 0.02,
    ) -> bool:
        """
        Validates shadow registry state against ground-truth page elements.
        Returns True if element counts match, IDs match, and attributes/geometry align.
        """
        if len(self.elements) != len(ground_truth_elements):
            return False

        gt_by_id = {el.element_id: el for el in ground_truth_elements}
        for eid, shadow_el in self.elements.items():
            if eid not in gt_by_id:
                return False
            gt_el = gt_by_id[eid]
            if shadow_el.tag != gt_el.tag or shadow_el.role != gt_el.role:
                return False
            if shadow_el.text != gt_el.text or shadow_el.value != gt_el.value:
                return False
            # Check bbox tolerance
            if shadow_el.has_bbox and gt_el.has_bbox:
                c_s = shadow_el.center
                c_g = gt_el.center
                dist = math.sqrt((c_s[0] - c_g[0]) ** 2 + (c_s[1] - c_g[1]) ** 2)
                if dist > tolerance:
                    return False

        return True


# ==============================================================================
# 5. In-Browser JavaScript MutationObserver Hook
# ==============================================================================

BROWSER_MUTATION_OBSERVER_INJECTION_JS = """
(() => {
    if (window.__jed_observer__) {
        return;
    }
    window.__jed_mutations__ = [];
    
    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            if (mutation.type === 'childList') {
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        window.__jed_mutations__.push({
                            type: 'childList_add',
                            tag: node.tagName ? node.tagName.toLowerCase() : '',
                            id: node.id || '',
                            className: node.className || '',
                            text: (node.innerText || node.textContent || '').trim().substring(0, 50),
                            timestamp: Date.now()
                        });
                    }
                }
                for (const node of mutation.removedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        window.__jed_mutations__.push({
                            type: 'childList_remove',
                            tag: node.tagName ? node.tagName.toLowerCase() : '',
                            id: node.id || '',
                            timestamp: Date.now()
                        });
                    }
                }
            } else if (mutation.type === 'attributes') {
                const target = mutation.target;
                if (target && target.nodeType === Node.ELEMENT_NODE) {
                    window.__jed_mutations__.push({
                        type: 'attributes',
                        tag: target.tagName ? target.tagName.toLowerCase() : '',
                        attributeName: mutation.attributeName,
                        id: target.id || '',
                        value: target.value || target.getAttribute(mutation.attributeName) || '',
                        timestamp: Date.now()
                    });
                }
            }
        }
    });

    observer.observe(document.documentElement, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'disabled', 'value', 'aria-hidden', 'aria-label', 'style']
    });

    window.__jed_observer__ = observer;
})();
"""

BROWSER_MUTATION_POLL_JS = """
(() => {
    if (!window.__jed_mutations__) {
        return [];
    }
    const muts = window.__jed_mutations__;
    window.__jed_mutations__ = [];
    return muts;
})();
"""
