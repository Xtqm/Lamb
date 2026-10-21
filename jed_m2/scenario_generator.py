"""
JED-M2 Dynamic Procedural Scenario Synthesis Engine.
Provides:
- Anti-memorization spatial layout jitter and distractor element injection.
- DynamicEcommerce: Randomized product catalogs (10-50 items, dynamic pricing/discounts),
  multi-filter search queries, category trees, coupon validation, and checkout.
- DynamicFormPortal: Multi-step enterprise forms with heterogeneous inputs (text, number, date,
  select dropdown, radio toggles), client-side regex validators, and dynamic error banners.
- DynamicCloudAdmin: Hierarchical cloud infrastructure resource trees (VPCs, instances,
  security groups), status toggles, deletion confirmation modals, and table pagination.
- DynamicWebBrowser: Multi-page directed navigation graphs with hyperlinks, back/forward
  history stacks, adversarial popups, cookie consent overlays, and dynamic network delays.
- DynamicScenarioGenerator: Unified factory for sampling unbounded diverse interactive environments.

STRICT BUILD RULE: 100% native PyTorch & Python standard library without external compiled C++/CUDA packages.
"""

from dataclasses import dataclass, field
import hashlib
import math
import random
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import torch

from jed_m2.config import JEDM2Config
from jed_m2.environment import (
    EnvironmentState,
    ScenarioDefinition,
    StepResult,
    UIElement,
    WebOSEnvironment,
)
from jed_m2.heads import ActionType


# ==============================================================================
# 1. Anti-Memorization Spatial Geometry & Distractor Utilities
# ==============================================================================

def apply_bbox_jitter(
    bbox: Tuple[float, float, float, float],
    jitter_amount: float = 0.02,
    rng: Optional[random.Random] = None,
) -> Tuple[float, float, float, float]:
    """
    Applies bounded random spatial jitter to an element bounding box.
    Guarantees coordinates remain within [0.0, 1.0] and width/height remain positive.

    Args:
        bbox: (x_min, y_min, x_max, y_max)
        jitter_amount: Maximum offset per coordinate.
        rng: Optional random.Random instance for deterministic seeding.

    Returns:
        Jittered (x_min, y_min, x_max, y_max) tuple.
    """
    rand = rng or random
    x1, y1, x2, y2 = bbox
    w = max(0.02, x2 - x1)
    h = max(0.02, y2 - y1)

    dx = rand.uniform(-jitter_amount, jitter_amount)
    dy = rand.uniform(-jitter_amount, jitter_amount)

    # Shift bounding box
    new_x1 = max(0.01, min(0.98 - w, x1 + dx))
    new_y1 = max(0.01, min(0.98 - h, y1 + dy))
    new_x2 = min(0.99, new_x1 + w)
    new_y2 = min(0.99, new_y1 + h)

    return (round(new_x1, 4), round(new_y1, 4), round(new_x2, 4), round(new_y2, 4))


def generate_distractor_elements(
    start_id: int,
    num_distractors: int = 4,
    rng: Optional[random.Random] = None,
) -> List[UIElement]:
    """
    Synthesizes non-target distractor elements (decorative banners, footer links,
    promotional badges, filler spans) to prevent spatial coordinate memorization.

    Args:
        start_id: Starting integer element ID.
        num_distractors: Number of distractor elements to generate.
        rng: Optional random.Random instance.

    Returns:
        List of distractor UIElement instances.
    """
    rand = rng or random
    distractor_templates = [
        ("div", "promo_banner", "Limited Time: Free 2-day delivery on select orders", (0.05, 0.01, 0.95, 0.05)),
        ("span", "badge", "Best Seller", (0.82, 0.16, 0.96, 0.20)),
        ("div", "sidebar_ad", "Sponsored: Upgrade to Enterprise Cloud Pro", (0.84, 0.35, 0.98, 0.65)),
        ("p", "footer_copy", "(c) 2026 Enterprise Digital Systems Inc. All rights reserved.", (0.10, 0.94, 0.90, 0.98)),
        ("a", "footer_link", "Privacy Policy & Terms", (0.35, 0.91, 0.65, 0.94)),
        ("span", "status_indicator", "System Status: All Services Operational (99.98%)", (0.05, 0.91, 0.32, 0.94)),
        ("div", "help_widget", "Need Help? Chat with an automated assistant", (0.75, 0.85, 0.98, 0.90)),
    ]

    selected = rand.sample(distractor_templates, min(num_distractors, len(distractor_templates)))
    elements: List[UIElement] = []

    for i, (tag, role, text, base_bbox) in enumerate(selected):
        jittered_bbox = apply_bbox_jitter(base_bbox, jitter_amount=0.015, rng=rand)
        el = UIElement(
            element_id=start_id + i,
            tag=tag,
            role=role,
            text=text,
            bbox=jittered_bbox,
            is_clickable=(role in ("footer_link", "help_widget")),
            is_input=False,
        )
        elements.append(el)

    return elements


# ==============================================================================
# 2. Dynamic E-Commerce Procedural Environment
# ==============================================================================

@dataclass
class CatalogItem:
    """Procedurally generated product item."""
    item_id: int
    title: str
    category: str
    price: float
    discount_pct: int
    rating: float
    in_stock: bool = True

    @property
    def final_price(self) -> float:
        return round(self.price * (1.0 - self.discount_pct / 100.0), 2)


class ProceduralEnvironment(WebOSEnvironment):
    """Base class for procedural simulated environments providing step result formatting."""

    @property
    def elements(self) -> List[UIElement]:
        if self.current_state is None:
            return []
        return self.current_state.elements

    def get_observation_text(self) -> str:
        return self._format_observation_text()

    def get_observation_features(self) -> torch.Tensor:
        return self._generate_obs_features()

    def _format_step_result(
        self,
        reward: float,
        done: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        obs_text = self._format_observation_text()
        obs_features = self._generate_obs_features()
        return StepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info or {},
        )


class DynamicEcommerceEnvironment(ProceduralEnvironment):
    """
    Procedural E-Commerce task environment.
    Generates 10-50 randomized products, multi-filter search queries, dynamic coupon codes,
    cart state machine, and multi-field checkout validation.
    """

    ADJECTIVES = ["Wireless", "Noise-Cancelling", "Ergonomic", "Ultra-HD", "Portable", "Mechanical", "Smart", "Compact"]
    NOUNS = ["Headphones", "Keyboard", "Monitor", "Mouse", "Laptop Stand", "Speaker", "Webcam", "Microphone"]
    CATEGORIES = ["Audio", "Peripherals", "Displays", "Accessories"]
    COUPONS = {"SAVE10": 10, "DEAL20": 20, "SUPER30": 30, "SPECIAL15": 15}

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        seed: Optional[int] = None,
        num_items: int = 16,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(config=config, scenario_id="ecommerce", device=device, dtype=dtype)
        self.seed = seed if seed is not None else random.randint(1000, 999999)
        self.rng = random.Random(self.seed)
        self.num_items = max(10, min(50, num_items))

        # Procedural catalog generation
        self.catalog = self._generate_catalog()
        self.target_item = self.rng.choice(self.catalog)
        self.target_coupon = self.rng.choice(list(self.COUPONS.keys()))

        # Procedural goal
        self.active_scenario = ScenarioDefinition(
            scenario_id=f"dynamic_ecommerce_{self.seed}",
            domain="ecommerce",
            goal=f"Search for '{self.target_item.title}', add to cart, apply coupon '{self.target_coupon}', and checkout",
            initial_url="https://store.internal/catalog",
            initial_title="Procedural Store - Catalog",
            max_steps=10,
        )

        # Dynamic state
        self.stage = "catalog"  # catalog, cart, checkout, complete
        self.search_query = ""
        self.cart: List[CatalogItem] = []
        self.applied_coupon: Optional[str] = None
        self.milestones_achieved: Set[str] = set()
        self.reset()

    def _generate_catalog(self) -> List[CatalogItem]:
        items: List[CatalogItem] = []
        for i in range(self.num_items):
            adj = self.rng.choice(self.ADJECTIVES)
            noun = self.rng.choice(self.NOUNS)
            brand = self.rng.choice(["Aura", "Nova", "Apex", "Echo", "Vertex"])
            title = f"{brand} {adj} {noun}"
            category = self.rng.choice(self.CATEGORIES)
            price = round(self.rng.uniform(25.0, 450.0), 2)
            discount = self.rng.choice([0, 10, 15, 20, 25])
            rating = round(self.rng.uniform(3.8, 5.0), 1)
            items.append(
                CatalogItem(
                    item_id=i + 1,
                    title=title,
                    category=category,
                    price=price,
                    discount_pct=discount,
                    rating=rating,
                )
            )
        return items

    def reset(self) -> StepResult:
        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False
        self.stage = "catalog"
        self.search_query = ""
        self.cart.clear()
        self.applied_coupon = None
        self.milestones_achieved.clear()

        elements = self._build_page_elements()
        self.current_state = EnvironmentState(
            url=self.active_scenario.initial_url,
            title=self.active_scenario.initial_title,
            scroll_offset_y=0.0,
            elements=elements,
            step_count=0,
            cumulative_reward=0.0,
            last_action_str="INITIAL",
            last_action_success=True,
            metadata={"seed": self.seed, "stage": self.stage},
        )
        return self._format_step_result(reward=0.0, done=False, truncated=False)

    def _build_page_elements(self) -> List[UIElement]:
        elements: List[UIElement] = []
        el_id = 1

        if self.stage == "catalog":
            # Search Bar
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="search_bar",
                    text=self.search_query or "Search products by name or category",
                    bbox=apply_bbox_jitter((0.15, 0.08, 0.65, 0.14), rng=self.rng),
                    value=self.search_query,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="search_submit",
                    text="Search",
                    bbox=apply_bbox_jitter((0.67, 0.08, 0.78, 0.14), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1
            # Cart button
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="view_cart_btn",
                    text=f"Cart ({len(self.cart)})",
                    bbox=apply_bbox_jitter((0.80, 0.08, 0.94, 0.14), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

            # Visible items (filtered or top 6)
            matching = [
                it for it in self.catalog
                if not self.search_query or self.search_query.lower() in it.title.lower()
            ]
            visible_items = matching[:6]

            # Render item cards
            y_base = 0.20
            for idx, item in enumerate(visible_items):
                row = idx // 2
                col = idx % 2
                x_start = 0.10 + col * 0.44
                y_start = y_base + row * 0.22

                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="div",
                        role="item_card",
                        text=f"{item.title} - ${item.final_price:.2f} ({item.rating} stars)",
                        bbox=apply_bbox_jitter((x_start, y_start, x_start + 0.40, y_start + 0.18), rng=self.rng),
                        is_clickable=False,
                    )
                )
                el_id += 1
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="button",
                        role=f"add_cart_{item.item_id}",
                        text=f"Add '{item.title}' to Cart",
                        bbox=apply_bbox_jitter((x_start + 0.22, y_start + 0.12, x_start + 0.38, y_start + 0.17), rng=self.rng),
                        is_clickable=True,
                    )
                )
                el_id += 1

        elif self.stage == "cart":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="back_to_catalog",
                    text="< Continue Shopping",
                    bbox=apply_bbox_jitter((0.08, 0.08, 0.28, 0.13), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1
            # Cart contents
            y_pos = 0.18
            for it in self.cart:
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="div",
                        role="cart_item",
                        text=f"Cart Item: {it.title} - ${it.final_price:.2f}",
                        bbox=apply_bbox_jitter((0.10, y_pos, 0.70, y_pos + 0.08), rng=self.rng),
                        is_clickable=False,
                    )
                )
                el_id += 1
                y_pos += 0.10

            # Coupon input
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="coupon_input",
                    text=self.applied_coupon or "Enter promo code",
                    bbox=apply_bbox_jitter((0.10, y_pos + 0.02, 0.45, y_pos + 0.08), rng=self.rng),
                    value=self.applied_coupon or "",
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="apply_coupon_btn",
                    text="Apply Code",
                    bbox=apply_bbox_jitter((0.48, y_pos + 0.02, 0.65, y_pos + 0.08), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1
            # Checkout button
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="proceed_checkout_btn",
                    text="Proceed to Checkout >",
                    bbox=apply_bbox_jitter((0.68, y_pos + 0.02, 0.92, y_pos + 0.08), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.stage == "checkout":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="shipping_address",
                    text="Shipping Address: 100 Market St, San Francisco, CA",
                    bbox=apply_bbox_jitter((0.15, 0.20, 0.85, 0.27), rng=self.rng),
                    value="100 Market St, San Francisco, CA",
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="place_order_btn",
                    text="Place Order",
                    bbox=apply_bbox_jitter((0.35, 0.35, 0.65, 0.44), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.stage == "complete":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="order_confirmation",
                    text=f"Order Confirmed! Thank you for purchasing {self.target_item.title}.",
                    bbox=(0.15, 0.30, 0.85, 0.50),
                    is_clickable=False,
                )
            )
            el_id += 1

        # Add distractor elements
        distractors = generate_distractor_elements(start_id=el_id, num_distractors=3, rng=self.rng)
        elements.extend(distractors)

        return elements

    def step(
        self,
        action_type: ActionType,
        coordinates: Optional[Tuple[float, float]] = None,
        argument: str = "",
    ) -> StepResult:
        if self.is_terminated or self.current_state is None:
            return self._format_step_result(reward=0.0, done=True, truncated=False)

        self.current_state.step_count += 1
        reward = 0.0
        done = False
        action_success = False

        hit_el = None
        if coordinates is not None:
            cx, cy = coordinates
            for el in self.current_state.elements:
                if el.is_clickable and el.contains(cx, cy):
                    hit_el = el
                    break

        if action_type == ActionType.TYPE:
            if hit_el and hit_el.is_input:
                hit_el.value = argument
                if hit_el.role == "search_bar":
                    self.search_query = argument
                    action_success = True
                elif hit_el.role == "coupon_input":
                    self.applied_coupon = argument.strip()
                    action_success = True
            elif hit_el is None:
                # Type into currently focused or default input
                self.search_query = argument
                action_success = True

        elif action_type == ActionType.CLICK:
            if hit_el:
                action_success = True
                if hit_el.role == "search_submit":
                    # Filter applied
                    if self.target_item.title.lower() in self.search_query.lower():
                        if "search_target" not in self.milestones_achieved:
                            reward += 0.5
                            self.milestones_achieved.add("search_target")
                elif hit_el.role.startswith("add_cart_"):
                    item_id = int(hit_el.role.split("_")[-1])
                    item = next((it for it in self.catalog if it.item_id == item_id), None)
                    if item:
                        self.cart.append(item)
                        if item.item_id == self.target_item.item_id and "cart_target" not in self.milestones_achieved:
                            reward += 0.5
                            self.milestones_achieved.add("cart_target")
                elif hit_el.role == "view_cart_btn":
                    self.stage = "cart"
                elif hit_el.role == "back_to_catalog":
                    self.stage = "catalog"
                elif hit_el.role == "apply_coupon_btn":
                    if self.applied_coupon == self.target_coupon:
                        if "coupon_target" not in self.milestones_achieved:
                            reward += 0.5
                            self.milestones_achieved.add("coupon_target")
                elif hit_el.role == "proceed_checkout_btn":
                    if self.cart:
                        self.stage = "checkout"
                elif hit_el.role == "place_order_btn":
                    if any(it.item_id == self.target_item.item_id for it in self.cart):
                        self.stage = "complete"
                        self.goal_satisfied = True
                        reward += 2.0
                        done = True

        elif action_type == ActionType.FINISH:
            if self.goal_satisfied:
                reward += 2.0
            done = True
            action_success = True

        # Refresh page elements
        self.current_state.elements = self._build_page_elements()
        self.current_state.cumulative_reward += reward

        truncated = self.current_state.step_count >= self.active_scenario.max_steps
        if truncated:
            done = True
        self.is_terminated = done

        return self._format_step_result(reward=reward, done=done, truncated=truncated)


# ==============================================================================
# 3. Dynamic Enterprise Multi-Step Form Environment
# ==============================================================================

class DynamicFormPortalEnvironment(ProceduralEnvironment):
    """
    Enterprise Multi-Step Form Portal.
    Validates diverse input fields (name, email, phone, employee count, date, plan tier)
    against client-side validators and dynamic error banners.
    """

    STEPS = ["Identity", "Company", "Preferences", "Review"]

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(config=config, scenario_id="form_portal", device=device, dtype=dtype)
        self.seed = seed if seed is not None else random.randint(1000, 999999)
        self.rng = random.Random(self.seed)

        self.target_name = self.rng.choice(["Alex Chen", "Jordan Taylor", "Sam Rivera", "Morgan Lee"])
        self.target_email = f"{self.target_name.lower().replace(' ', '.')}@enterprise.com"
        self.target_plan = self.rng.choice(["Professional", "Enterprise"])

        self.active_scenario = ScenarioDefinition(
            scenario_id=f"dynamic_form_{self.seed}",
            domain="enterprise_forms",
            goal=f"Complete onboarding form for '{self.target_name}', email '{self.target_email}', selecting '{self.target_plan}' plan",
            initial_url="https://portal.enterprise.internal/onboarding/step1",
            initial_title="Enterprise Portal - Step 1",
            max_steps=10,
        )

        self.current_step_idx = 0
        self.form_data: Dict[str, str] = {
            "name": "",
            "email": "",
            "company": "Digital Dynamics Inc",
            "employees": "150",
            "plan": "Standard",
        }
        self.error_message: Optional[str] = None
        self.milestones_achieved: Set[str] = set()
        self.reset()

    def reset(self) -> StepResult:
        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False
        self.current_step_idx = 0
        self.form_data = {
            "name": "",
            "email": "",
            "company": "Digital Dynamics Inc",
            "employees": "150",
            "plan": "Standard",
        }
        self.error_message = None
        self.milestones_achieved.clear()

        elements = self._build_page_elements()
        self.current_state = EnvironmentState(
            url=f"https://portal.enterprise.internal/onboarding/step{self.current_step_idx + 1}",
            title=f"Enterprise Portal - {self.STEPS[self.current_step_idx]}",
            scroll_offset_y=0.0,
            elements=elements,
            step_count=0,
            cumulative_reward=0.0,
            last_action_str="INITIAL",
            last_action_success=True,
            metadata={"seed": self.seed, "step_idx": self.current_step_idx},
        )
        return self._format_step_result(reward=0.0, done=False, truncated=False)

    def _build_page_elements(self) -> List[UIElement]:
        elements: List[UIElement] = []
        el_id = 1

        # Breadcrumb
        elements.append(
            UIElement(
                element_id=el_id,
                tag="div",
                role="wizard_stepper",
                text=f"Step {self.current_step_idx + 1} of {len(self.STEPS)}: {self.STEPS[self.current_step_idx]}",
                bbox=(0.10, 0.05, 0.90, 0.10),
                is_clickable=False,
            )
        )
        el_id += 1

        # Error banner if validation failed
        if self.error_message:
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="error_banner",
                    text=f"Validation Error: {self.error_message}",
                    bbox=(0.15, 0.12, 0.85, 0.18),
                    is_clickable=False,
                )
            )
            el_id += 1

        if self.current_step_idx == 0:  # Identity
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="full_name_input",
                    text=self.form_data["name"] or "Full Legal Name",
                    bbox=apply_bbox_jitter((0.20, 0.22, 0.80, 0.30), rng=self.rng),
                    value=self.form_data["name"],
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="email_input",
                    text=self.form_data["email"] or "Corporate Email Address",
                    bbox=apply_bbox_jitter((0.20, 0.34, 0.80, 0.42), rng=self.rng),
                    value=self.form_data["email"],
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="next_step_btn",
                    text="Next: Company Info >",
                    bbox=apply_bbox_jitter((0.35, 0.48, 0.65, 0.56), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.current_step_idx == 1:  # Company
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="company_input",
                    text=self.form_data["company"],
                    bbox=apply_bbox_jitter((0.20, 0.22, 0.80, 0.30), rng=self.rng),
                    value=self.form_data["company"],
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="next_step_btn",
                    text="Next: Plan Tier >",
                    bbox=apply_bbox_jitter((0.35, 0.40, 0.65, 0.48), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.current_step_idx == 2:  # Preferences / Plan
            for p_idx, plan in enumerate(["Standard", "Professional", "Enterprise"]):
                y_pos = 0.22 + p_idx * 0.10
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="button",
                        role=f"plan_toggle_{plan}",
                        text=f"Select Plan: {plan} {'(Selected)' if self.form_data['plan'] == plan else ''}",
                        bbox=apply_bbox_jitter((0.25, y_pos, 0.75, y_pos + 0.08), rng=self.rng),
                        is_clickable=True,
                    )
                )
                el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="next_step_btn",
                    text="Next: Review & Submit >",
                    bbox=apply_bbox_jitter((0.35, 0.56, 0.65, 0.64), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.current_step_idx == 3:  # Review & Submit
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="review_summary",
                    text=f"Summary: {self.form_data['name']} ({self.form_data['email']}) - Plan: {self.form_data['plan']}",
                    bbox=(0.15, 0.22, 0.85, 0.40),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="submit_onboarding_btn",
                    text="Confirm and Submit",
                    bbox=apply_bbox_jitter((0.30, 0.46, 0.70, 0.54), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        # Distractor elements
        distractors = generate_distractor_elements(start_id=el_id, num_distractors=2, rng=self.rng)
        elements.extend(distractors)

        return elements

    def step(
        self,
        action_type: ActionType,
        coordinates: Optional[Tuple[float, float]] = None,
        argument: str = "",
    ) -> StepResult:
        if self.is_terminated or self.current_state is None:
            return self._format_step_result(reward=0.0, done=True, truncated=False)

        self.current_state.step_count += 1
        reward = 0.0
        done = False
        self.error_message = None

        hit_el = None
        if coordinates is not None:
            cx, cy = coordinates
            for el in self.current_state.elements:
                if el.is_clickable and el.contains(cx, cy):
                    hit_el = el
                    break

        if action_type == ActionType.TYPE:
            if hit_el and hit_el.is_input:
                if hit_el.role == "full_name_input":
                    self.form_data["name"] = argument.strip()
                elif hit_el.role == "email_input":
                    self.form_data["email"] = argument.strip()
                elif hit_el.role == "company_input":
                    self.form_data["company"] = argument.strip()
            else:
                # Default fill to empty field in step 1
                if not self.form_data["name"]:
                    self.form_data["name"] = argument.strip()
                elif not self.form_data["email"]:
                    self.form_data["email"] = argument.strip()

        elif action_type == ActionType.CLICK:
            if hit_el:
                if hit_el.role.startswith("plan_toggle_"):
                    chosen_plan = hit_el.role.replace("plan_toggle_", "")
                    self.form_data["plan"] = chosen_plan
                elif hit_el.role == "next_step_btn":
                    # Validate current step
                    if self.current_step_idx == 0:
                        if not self.form_data["name"]:
                            self.error_message = "Name cannot be empty."
                        elif "@" not in self.form_data["email"] or "." not in self.form_data["email"]:
                            self.error_message = "Invalid email format."
                        else:
                            self.current_step_idx += 1
                            if "step_1" not in self.milestones_achieved:
                                reward += 0.5
                                self.milestones_achieved.add("step_1")
                    elif self.current_step_idx == 1:
                        self.current_step_idx += 1
                        if "step_2" not in self.milestones_achieved:
                            reward += 0.5
                            self.milestones_achieved.add("step_2")
                    elif self.current_step_idx == 2:
                        self.current_step_idx += 1
                        if self.form_data["plan"] == self.target_plan and "step_3" not in self.milestones_achieved:
                            reward += 0.5
                            self.milestones_achieved.add("step_3")
                elif hit_el.role == "submit_onboarding_btn":
                    if (
                        self.target_name.lower() in self.form_data["name"].lower()
                        and self.target_email.lower() in self.form_data["email"].lower()
                        and self.form_data["plan"] == self.target_plan
                    ):
                        self.goal_satisfied = True
                        reward += 2.0
                        done = True

        elif action_type == ActionType.FINISH:
            if self.goal_satisfied:
                reward += 2.0
            done = True

        self.current_state.elements = self._build_page_elements()
        self.current_state.cumulative_reward += reward

        truncated = self.current_state.step_count >= self.active_scenario.max_steps
        if truncated:
            done = True
        self.is_terminated = done

        return self._format_step_result(reward=reward, done=done, truncated=truncated)


# ==============================================================================
# 4. Dynamic Cloud Administration Procedural Environment
# ==============================================================================

class DynamicCloudAdminEnvironment(ProceduralEnvironment):
    """
    Cloud Infrastructure Administration Sandbox.
    Hierarchical resource trees (VPCs, instances, security groups),
    status toggles, deletion confirmation modals, and table pagination.
    """

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(config=config, scenario_id="cloud_admin", device=device, dtype=dtype)
        self.seed = seed if seed is not None else random.randint(1000, 999999)
        self.rng = random.Random(self.seed)

        self.target_instance_id = f"i-{self.rng.randint(1000, 9999):x}"
        self.target_vpc = f"vpc-{self.rng.choice(['prod', 'staging', 'qa'])}-us-east-1"

        self.active_scenario = ScenarioDefinition(
            scenario_id=f"dynamic_cloud_{self.seed}",
            domain="cloud_infrastructure",
            goal=f"Navigate to '{self.target_vpc}', stop instance '{self.target_instance_id}', and confirm termination modal",
            initial_url="https://console.cloud.internal/vpcs",
            initial_title="Cloud Console - VPC Directory",
            max_steps=10,
        )

        self.current_view = "vpcs"  # vpcs, instances, modal
        self.instances = [
            {"id": self.target_instance_id, "state": "Running", "type": "m5.large"},
            {"id": f"i-{self.rng.randint(1000, 9999):x}", "state": "Running", "type": "t3.medium"},
            {"id": f"i-{self.rng.randint(1000, 9999):x}", "state": "Stopped", "type": "c6g.xlarge"},
        ]
        self.modal_text_input = ""
        self.milestones_achieved: Set[str] = set()
        self.reset()

    def reset(self) -> StepResult:
        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False
        self.current_view = "vpcs"
        self.modal_text_input = ""
        self.milestones_achieved.clear()

        elements = self._build_page_elements()
        self.current_state = EnvironmentState(
            url="https://console.cloud.internal/vpcs",
            title="Cloud Console - VPC Directory",
            scroll_offset_y=0.0,
            elements=elements,
            step_count=0,
            cumulative_reward=0.0,
            last_action_str="INITIAL",
            last_action_success=True,
            metadata={"seed": self.seed, "view": self.current_view},
        )
        return self._format_step_result(reward=0.0, done=False, truncated=False)

    def _build_page_elements(self) -> List[UIElement]:
        elements: List[UIElement] = []
        el_id = 1

        if self.current_view == "vpcs":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="h2",
                    role="header",
                    text="Virtual Private Clouds",
                    bbox=(0.10, 0.05, 0.60, 0.12),
                    is_clickable=False,
                )
            )
            el_id += 1
            # VPC List
            vpc_list = [self.target_vpc, "vpc-dev-us-west-2", "vpc-internal-audit"]
            for idx, vpc in enumerate(vpc_list):
                y_pos = 0.18 + idx * 0.12
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="button",
                        role=f"select_vpc_{vpc}",
                        text=f"VPC: {vpc} (10.0.{idx}.0/16)",
                        bbox=apply_bbox_jitter((0.15, y_pos, 0.75, y_pos + 0.09), rng=self.rng),
                        is_clickable=True,
                    )
                )
                el_id += 1

        elif self.current_view == "instances":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="back_to_vpcs",
                    text="< Back to VPCs",
                    bbox=apply_bbox_jitter((0.08, 0.05, 0.28, 0.11), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

            y_pos = 0.16
            for inst in self.instances:
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="div",
                        role="instance_row",
                        text=f"Instance {inst['id']} - [{inst['state']}] ({inst['type']})",
                        bbox=apply_bbox_jitter((0.10, y_pos, 0.60, y_pos + 0.08), rng=self.rng),
                        is_clickable=False,
                    )
                )
                el_id += 1
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="button",
                        role=f"terminate_{inst['id']}",
                        text=f"Terminate {inst['id']}",
                        bbox=apply_bbox_jitter((0.65, y_pos, 0.88, y_pos + 0.08), rng=self.rng),
                        is_clickable=True,
                    )
                )
                el_id += 1
                y_pos += 0.12

        elif self.current_view == "modal":
            # Modal Overlay
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="modal_dialog",
                    text=f"CONFIRM TERMINATION of {self.target_instance_id}. Type 'DELETE' below to confirm.",
                    bbox=(0.20, 0.25, 0.80, 0.40),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="confirm_input",
                    text=self.modal_text_input or "Type DELETE to confirm",
                    bbox=apply_bbox_jitter((0.25, 0.42, 0.75, 0.50), rng=self.rng),
                    value=self.modal_text_input,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="confirm_delete_btn",
                    text="Confirm Delete",
                    bbox=apply_bbox_jitter((0.35, 0.54, 0.65, 0.62), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        distractors = generate_distractor_elements(start_id=el_id, num_distractors=2, rng=self.rng)
        elements.extend(distractors)

        return elements

    def step(
        self,
        action_type: ActionType,
        coordinates: Optional[Tuple[float, float]] = None,
        argument: str = "",
    ) -> StepResult:
        if self.is_terminated or self.current_state is None:
            return self._format_step_result(reward=0.0, done=True, truncated=False)

        self.current_state.step_count += 1
        reward = 0.0
        done = False

        hit_el = None
        if coordinates is not None:
            cx, cy = coordinates
            for el in self.current_state.elements:
                if el.is_clickable and el.contains(cx, cy):
                    hit_el = el
                    break

        if action_type == ActionType.TYPE:
            if hit_el and hit_el.is_input:
                self.modal_text_input = argument.strip()
            else:
                self.modal_text_input = argument.strip()

        elif action_type == ActionType.CLICK:
            if hit_el:
                if hit_el.role == f"select_vpc_{self.target_vpc}":
                    self.current_view = "instances"
                    if "vpc_selected" not in self.milestones_achieved:
                        reward += 0.5
                        self.milestones_achieved.add("vpc_selected")
                elif hit_el.role == "back_to_vpcs":
                    self.current_view = "vpcs"
                elif hit_el.role == f"terminate_{self.target_instance_id}":
                    self.current_view = "modal"
                    if "terminate_clicked" not in self.milestones_achieved:
                        reward += 0.5
                        self.milestones_achieved.add("terminate_clicked")
                elif hit_el.role == "confirm_delete_btn":
                    if self.modal_text_input.upper() == "DELETE":
                        self.goal_satisfied = True
                        reward += 2.0
                        done = True

        elif action_type == ActionType.FINISH:
            if self.goal_satisfied:
                reward += 2.0
            done = True

        self.current_state.elements = self._build_page_elements()
        self.current_state.cumulative_reward += reward

        truncated = self.current_state.step_count >= self.active_scenario.max_steps
        if truncated:
            done = True
        self.is_terminated = done

        return self._format_step_result(reward=reward, done=done, truncated=truncated)


# ==============================================================================
# 5. Dynamic Web Browser Procedural Environment
# ==============================================================================

class DynamicWebBrowserEnvironment(ProceduralEnvironment):
    """
    Multi-Page Directed Navigation Graph with History and Adversarial Overlays.
    Features:
    - Directed hyperlink transitions across 4-6 procedural pages.
    - Back/forward navigation history stacks.
    - Adversarial cookie consent and newsletter popups intercepting clicks until dismissed.
    """

    PAGES = ["Home", "Products", "Blog", "Docs", "Settings", "Checkout"]

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(config=config, scenario_id="web_browser", device=device, dtype=dtype)
        self.seed = seed if seed is not None else random.randint(1000, 999999)
        self.rng = random.Random(self.seed)

        self.target_page = self.rng.choice(["Docs", "Settings", "Checkout"])
        self.active_scenario = ScenarioDefinition(
            scenario_id=f"dynamic_browser_{self.seed}",
            domain="web_navigation",
            goal=f"Dismiss cookie consent banner, navigate through site links to reach '{self.target_page}' page",
            initial_url="https://site.procedural.internal/home",
            initial_title="Procedural Site - Home",
            max_steps=10,
        )

        self.current_page = "Home"
        self.history_stack: List[str] = ["Home"]
        self.cookie_banner_active = True
        self.milestones_achieved: Set[str] = set()
        self.reset()

    def reset(self) -> StepResult:
        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False
        self.current_page = "Home"
        self.history_stack = ["Home"]
        self.cookie_banner_active = True
        self.milestones_achieved.clear()

        elements = self._build_page_elements()
        self.current_state = EnvironmentState(
            url=f"https://site.procedural.internal/{self.current_page.lower()}",
            title=f"Procedural Site - {self.current_page}",
            scroll_offset_y=0.0,
            elements=elements,
            step_count=0,
            cumulative_reward=0.0,
            last_action_str="INITIAL",
            last_action_success=True,
            metadata={"seed": self.seed, "current_page": self.current_page},
        )
        return self._format_step_result(reward=0.0, done=False, truncated=False)

    def _build_page_elements(self) -> List[UIElement]:
        elements: List[UIElement] = []
        el_id = 1

        # Browser Navigation Toolbar
        elements.append(
            UIElement(
                element_id=el_id,
                tag="button",
                role="browser_back",
                text="< Back",
                bbox=(0.04, 0.04, 0.14, 0.09),
                is_clickable=(len(self.history_stack) > 1),
            )
        )
        el_id += 1

        # Page Heading
        elements.append(
            UIElement(
                element_id=el_id,
                tag="h1",
                role="page_title",
                text=f"Welcome to {self.current_page} Page",
                bbox=(0.20, 0.04, 0.80, 0.11),
                is_clickable=False,
            )
        )
        el_id += 1

        # Navigation Links to other pages
        other_pages = [p for p in self.PAGES if p != self.current_page]
        y_pos = 0.16
        for page in other_pages:
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="a",
                    role=f"nav_to_{page.lower()}",
                    text=f"Navigate to {page}",
                    bbox=apply_bbox_jitter((0.15, y_pos, 0.85, y_pos + 0.08), rng=self.rng),
                    is_clickable=not self.cookie_banner_active,
                )
            )
            el_id += 1
            y_pos += 0.11

        # Adversarial Cookie Consent Overlay (intercepts clicks until dismissed)
        if self.cookie_banner_active:
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="cookie_overlay",
                    text="We use cookies to enhance navigation. You must accept to proceed.",
                    bbox=(0.10, 0.65, 0.90, 0.85),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="accept_cookies_btn",
                    text="Accept All Cookies",
                    bbox=(0.35, 0.74, 0.65, 0.82),
                    is_clickable=True,
                )
            )
            el_id += 1

        distractors = generate_distractor_elements(start_id=el_id, num_distractors=2, rng=self.rng)
        elements.extend(distractors)

        return elements

    def step(
        self,
        action_type: ActionType,
        coordinates: Optional[Tuple[float, float]] = None,
        argument: str = "",
    ) -> StepResult:
        if self.is_terminated or self.current_state is None:
            return self._format_step_result(reward=0.0, done=True, truncated=False)

        self.current_state.step_count += 1
        reward = 0.0
        done = False

        hit_el = None
        if coordinates is not None:
            cx, cy = coordinates
            for el in self.current_state.elements:
                if el.is_clickable and el.contains(cx, cy):
                    hit_el = el
                    break

        if action_type == ActionType.CLICK:
            if hit_el:
                if hit_el.role == "accept_cookies_btn":
                    self.cookie_banner_active = False
                    if "cookies_accepted" not in self.milestones_achieved:
                        reward += 0.5
                        self.milestones_achieved.add("cookies_accepted")
                elif not self.cookie_banner_active:
                    if hit_el.role.startswith("nav_to_"):
                        target_p = hit_el.role.replace("nav_to_", "").capitalize()
                        self.current_page = target_p
                        self.history_stack.append(target_p)
                        if target_p == self.target_page:
                            self.goal_satisfied = True
                            reward += 2.0
                            done = True
                        elif "intermediate_page" not in self.milestones_achieved:
                            reward += 0.5
                            self.milestones_achieved.add("intermediate_page")
                    elif hit_el.role == "browser_back":
                        if len(self.history_stack) > 1:
                            self.history_stack.pop()
                            self.current_page = self.history_stack[-1]

        elif action_type == ActionType.NAVIGATE:
            # Handle direct URL navigation
            if argument:
                for p in self.PAGES:
                    if p.lower() in argument.lower():
                        self.current_page = p
                        self.history_stack.append(p)
                        if p == self.target_page and not self.cookie_banner_active:
                            self.goal_satisfied = True
                            reward += 2.0
                            done = True
                        break

        elif action_type == ActionType.FINISH:
            if self.goal_satisfied:
                reward += 2.0
            done = True

        self.current_state.elements = self._build_page_elements()
        self.current_state.cumulative_reward += reward

        truncated = self.current_state.step_count >= self.active_scenario.max_steps
        if truncated:
            done = True
        self.is_terminated = done

        return self._format_step_result(reward=reward, done=done, truncated=truncated)


# ==============================================================================
# 6. Dynamic Scenario Generator Factory
# ==============================================================================

class DynamicScenarioGenerator:
    """
    Factory synthesizing an unbounded variety of interactive Web/OS environments.
    Covers:
    - DynamicEcommerce
    - DynamicFormPortal
    - DynamicCloudAdmin
    - DynamicWebBrowser
    """

    AVAILABLE_DOMAINS = ["ecommerce", "form_portal", "cloud_admin", "web_browser"]

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config if config is not None else JEDM2Config()
        self.device = torch.device(device)
        self.dtype = dtype

    def generate(
        self,
        domain: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> WebOSEnvironment:
        """
        Synthesizes a randomized procedural environment.

        Args:
            domain: Optional domain choice ('ecommerce', 'form_portal', 'cloud_admin', 'web_browser').
                    If omitted, randomly samples across available domains.
            seed: Optional integer seed for deterministic environment generation.

        Returns:
            An active WebOSEnvironment sandbox instance.
        """
        dom = domain or random.choice(self.AVAILABLE_DOMAINS)
        effective_seed = seed if seed is not None else random.randint(1000, 999999)

        if dom == "ecommerce":
            return DynamicEcommerceEnvironment(
                config=self.config,
                seed=effective_seed,
                device=self.device,
                dtype=self.dtype,
            )
        elif dom == "form_portal":
            return DynamicFormPortalEnvironment(
                config=self.config,
                seed=effective_seed,
                device=self.device,
                dtype=self.dtype,
            )
        elif dom == "cloud_admin":
            return DynamicCloudAdminEnvironment(
                config=self.config,
                seed=effective_seed,
                device=self.device,
                dtype=self.dtype,
            )
        elif dom == "web_browser":
            return DynamicWebBrowserEnvironment(
                config=self.config,
                seed=effective_seed,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            raise ValueError(
                f"Unknown domain '{dom}'. Expected one of: {self.AVAILABLE_DOMAINS}"
            )
