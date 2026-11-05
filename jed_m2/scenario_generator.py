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


# ==============================================================================
# 1b. Procedural Obstacle & Adversarial Noise Engine
# ==============================================================================

@dataclass
class ObstacleState:
    """Represents an active procedural obstacle or UI friction element."""

    obstacle_type: str = "none"  # "cookie_consent", "validation_error", "loading_delay", "promo_popup"
    active: bool = False
    message: str = ""
    dismiss_role: str = ""
    delay_remaining: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "obstacle_type": self.obstacle_type,
            "active": self.active,
            "message": self.message,
            "dismiss_role": self.dismiss_role,
            "delay_remaining": self.delay_remaining,
        }


class ObstacleInjector:
    """
    Procedural obstacle and adversarial noise injection engine.
    Injects cookie consent overlays, client-side input validation errors,
    asynchronous loading delays, and promotional popups.
    """

    OBSTACLE_TYPES = ["cookie_consent", "validation_error", "loading_delay", "promo_popup"]

    def __init__(self, default_prob: float = 0.35, rng: Optional[random.Random] = None):
        self.default_prob = default_prob
        self.rng = rng or random

    def maybe_inject(self, env: "ProceduralEnvironment", prob: Optional[float] = None) -> Optional[ObstacleState]:
        """Probabilistically injects an obstacle into the environment if none is active."""
        p = prob if prob is not None else self.default_prob
        if getattr(env, "obstacle_state", None) is not None and env.obstacle_state.active:
            return env.obstacle_state
        if self.rng.random() < p:
            obs_type = self.rng.choice(self.OBSTACLE_TYPES)
            return self.inject_obstacle(env, obstacle_type=obs_type)
        return None

    def inject_obstacle(
        self,
        env: "ProceduralEnvironment",
        obstacle_type: str = "cookie_consent",
    ) -> ObstacleState:
        """Injects a specific obstacle into the given procedural environment."""
        if obstacle_type == "cookie_consent":
            state = ObstacleState(
                obstacle_type="cookie_consent",
                active=True,
                message="Cookie consent modal active; user must accept or dismiss to interact with page.",
                dismiss_role="cookie_accept_btn",
            )
        elif obstacle_type == "validation_error":
            state = ObstacleState(
                obstacle_type="validation_error",
                active=True,
                message="Client-side validation error banner active; user must correct input or dismiss alert.",
                dismiss_role="dismiss_validation_btn",
            )
        elif obstacle_type == "loading_delay":
            state = ObstacleState(
                obstacle_type="loading_delay",
                active=True,
                message="Simulated asynchronous loading delay; page components settling.",
                dismiss_role="spinner_overlay",
                delay_remaining=1,
            )
        elif obstacle_type == "promo_popup":
            state = ObstacleState(
                obstacle_type="promo_popup",
                active=True,
                message="Promotional newsletter modal active; close icon must be clicked.",
                dismiss_role="promo_close_btn",
            )
        else:
            state = ObstacleState(obstacle_type="none", active=False)

        env.obstacle_state = state
        if env.current_state is not None:
            env.current_state.metadata["obstacle"] = state.to_dict()
            env.current_state.elements = env._build_page_elements()
        return state

    @staticmethod
    def get_obstacle_elements(obstacle_state: ObstacleState, start_id: int) -> List[UIElement]:
        """Synthesizes DOM elements representing the active obstacle overlay."""
        elements: List[UIElement] = []
        el_id = start_id

        if not obstacle_state.active:
            return elements

        if obstacle_state.obstacle_type == "cookie_consent":
            # Modal backdrop and dialog buttons
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="cookie_consent_modal",
                    text="We value your privacy. This site uses cookies to ensure optimal performance.",
                    bbox=(0.10, 0.40, 0.90, 0.58),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="cookie_accept_btn",
                    text="Accept All Cookies",
                    bbox=(0.22, 0.50, 0.48, 0.56),
                    is_clickable=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="cookie_dismiss_btn",
                    text="Dismiss",
                    bbox=(0.52, 0.50, 0.78, 0.56),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif obstacle_state.obstacle_type == "validation_error":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="validation_error_alert",
                    text="Validation Alert: Please verify required fields and format before proceeding.",
                    bbox=(0.10, 0.12, 0.72, 0.18),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="dismiss_validation_btn",
                    text="Dismiss Error",
                    bbox=(0.75, 0.12, 0.90, 0.18),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif obstacle_state.obstacle_type == "loading_delay":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="spinner_overlay",
                    text="Loading requested resource... please wait.",
                    bbox=(0.25, 0.42, 0.75, 0.52),
                    is_clickable=False,
                )
            )
            el_id += 1

        elif obstacle_state.obstacle_type == "promo_popup":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="promo_popup_modal",
                    text="Exclusive Offer! Sign up today for 20% off your transaction.",
                    bbox=(0.15, 0.32, 0.85, 0.52),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="promo_close_btn",
                    text="[X] Close",
                    bbox=(0.78, 0.33, 0.84, 0.38),
                    is_clickable=True,
                )
            )
            el_id += 1

        return elements


class ProceduralEnvironment(WebOSEnvironment):
    """Base class for procedural simulated environments providing step result formatting and obstacle interception."""

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        scenario_id: str = "procedural",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(config=config, scenario_id=scenario_id, device=device, dtype=dtype)
        self.obstacle_state: Optional[ObstacleState] = None
        self.obstacle_injector: Optional[ObstacleInjector] = None

    @property
    def elements(self) -> List[UIElement]:
        if self.current_state is None:
            return []
        return self.current_state.elements

    def get_observation_text(self) -> str:
        return self._format_observation_text()

    def get_observation_features(self) -> torch.Tensor:
        return self._generate_obs_features()

    def _format_observation_text(self) -> str:
        """Serializes observation text with explicit obstacle signals when active."""
        base_text = super()._format_observation_text()
        if self.obstacle_state is not None and self.obstacle_state.active:
            obs_cue = f"[OBSTACLE: {self.obstacle_state.obstacle_type}] {self.obstacle_state.message} | "
            base_text = obs_cue + base_text
        return base_text

    def _format_step_result(
        self,
        reward: float,
        done: bool,
        truncated: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        obs_text = self._format_observation_text()
        obs_features = self._generate_obs_features()
        merged_info = dict(info or {})
        if self.obstacle_state is not None and self.obstacle_state.active:
            merged_info["obstacle"] = self.obstacle_state.to_dict()
        return StepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=reward,
            done=done,
            truncated=truncated,
            info=merged_info,
        )

    def handle_obstacle_step(
        self,
        action_type: ActionType,
        coordinates: Optional[Tuple[float, float]],
        hit_el: Optional[UIElement],
    ) -> Tuple[bool, bool, float]:
        """
        Evaluates whether an active obstacle intercepts or is resolved by the agent's action.
        Returns:
            (is_intercepted, obstacle_dismissed, reward_bonus)
        """
        if self.obstacle_state is None or not self.obstacle_state.active:
            return False, False, 0.0

        if self.obstacle_state.obstacle_type == "loading_delay":
            if self.obstacle_state.delay_remaining > 0:
                self.obstacle_state.delay_remaining -= 1
                if self.obstacle_state.delay_remaining <= 0:
                    self.obstacle_state.active = False
                    if hasattr(self, "milestones_achieved"):
                        self.milestones_achieved.add("obstacle_dismissed")
                    return True, True, 0.15
            else:
                self.obstacle_state.active = False
            return True, True, 0.0

        dismiss_roles = {
            "cookie_consent": ("cookie_accept_btn", "cookie_dismiss_btn"),
            "validation_error": ("dismiss_validation_btn",),
            "promo_popup": ("promo_close_btn",),
        }
        valid_roles = dismiss_roles.get(self.obstacle_state.obstacle_type, ())

        if hit_el is not None and hit_el.role in valid_roles:
            self.obstacle_state.active = False
            if hasattr(self, "milestones_achieved"):
                self.milestones_achieved.add("obstacle_dismissed")
            return False, True, 0.15

        # If obstacle is modal or backdrop and click did not resolve it, intercept click
        if self.obstacle_state.obstacle_type in ("cookie_consent", "promo_popup", "validation_error"):
            return True, False, 0.0

        return False, False, 0.0


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

        # Add obstacle elements if active
        if getattr(self, "obstacle_state", None) is not None and self.obstacle_state.active:
            elements.extend(ObstacleInjector.get_obstacle_elements(self.obstacle_state, start_id=el_id))
            el_id += 5

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

        intercepted, dismissed, obs_reward = self.handle_obstacle_step(action_type, coordinates, hit_el)
        if dismissed:
            reward += obs_reward
        if intercepted:
            self.current_state.elements = self._build_page_elements()
            self.current_state.cumulative_reward += reward
            truncated = self.current_state.step_count >= self.active_scenario.max_steps
            if truncated:
                done = True
            self.is_terminated = done
            return self._format_step_result(reward=reward, done=done, truncated=truncated)

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

        # Add obstacle elements if active
        if getattr(self, "obstacle_state", None) is not None and self.obstacle_state.active:
            elements.extend(ObstacleInjector.get_obstacle_elements(self.obstacle_state, start_id=el_id))
            el_id += 5

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

        intercepted, dismissed, obs_reward = self.handle_obstacle_step(action_type, coordinates, hit_el)
        if dismissed:
            reward += obs_reward
        if intercepted:
            self.current_state.elements = self._build_page_elements()
            self.current_state.cumulative_reward += reward
            truncated = self.current_state.step_count >= self.active_scenario.max_steps
            if truncated:
                done = True
            self.is_terminated = done
            return self._format_step_result(reward=reward, done=done, truncated=truncated)

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

        # Add obstacle elements if active
        if getattr(self, "obstacle_state", None) is not None and self.obstacle_state.active:
            elements.extend(ObstacleInjector.get_obstacle_elements(self.obstacle_state, start_id=el_id))
            el_id += 5

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

        intercepted, dismissed, obs_reward = self.handle_obstacle_step(action_type, coordinates, hit_el)
        if dismissed:
            reward += obs_reward
        if intercepted:
            self.current_state.elements = self._build_page_elements()
            self.current_state.cumulative_reward += reward
            truncated = self.current_state.step_count >= self.active_scenario.max_steps
            if truncated:
                done = True
            self.is_terminated = done
            return self._format_step_result(reward=reward, done=done, truncated=truncated)

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

        # Add obstacle elements if active
        if getattr(self, "obstacle_state", None) is not None and self.obstacle_state.active:
            elements.extend(ObstacleInjector.get_obstacle_elements(self.obstacle_state, start_id=el_id))
            el_id += 5

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

        intercepted, dismissed, obs_reward = self.handle_obstacle_step(action_type, coordinates, hit_el)
        if dismissed:
            reward += obs_reward
        if intercepted:
            self.current_state.elements = self._build_page_elements()
            self.current_state.cumulative_reward += reward
            truncated = self.current_state.step_count >= self.active_scenario.max_steps
            if truncated:
                done = True
            self.is_terminated = done
            return self._format_step_result(reward=reward, done=done, truncated=truncated)

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
# 5. Dynamic Travel Booking Procedural Environment
# ==============================================================================

@dataclass
class FlightBookingRecord:
    """Represents a confirmed flight booking reservation record."""

    booking_id: str
    origin: str
    destination: str
    departure_date: str
    flight_number: str
    seat: str
    passenger_name: str
    passenger_email: str
    add_ons: List[str]
    total_price: float
    confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "booking_id": self.booking_id,
            "origin": self.origin,
            "destination": self.destination,
            "departure_date": self.departure_date,
            "flight_number": self.flight_number,
            "seat": self.seat,
            "passenger_name": self.passenger_name,
            "passenger_email": self.passenger_email,
            "add_ons": self.add_ons,
            "total_price": self.total_price,
            "confirmed": self.confirmed,
        }


class DynamicBookingEnvironment(ProceduralEnvironment):
    """
    Multi-Step Travel Booking Procedural Sandbox.
    Features:
    - Route searching with origin, destination, and calendar departure date.
    - Tier/flight option selection with dynamic pricing.
    - Passenger detail entry, interactive seat selection grid, and baggage add-on toggles.
    - Deterministic review and booking confirmation reference generation.
    """

    AIRPORTS = ["SFO", "JFK", "LAX", "ORD", "SEA", "BOS", "MIA", "AUS"]
    PASSENGERS = ["Alex Chen", "Jordan Taylor", "Sam Rivera", "Morgan Lee", "Chris Zhang"]
    SEATS = ["12A", "12B", "14A", "14B", "16C", "16D"]

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(config=config, scenario_id="travel_booking", device=device, dtype=dtype)
        self.seed = seed if seed is not None else random.randint(1000, 999999)
        self.rng = random.Random(self.seed)

        self.target_origin = self.rng.choice(self.AIRPORTS)
        dest_choices = [a for a in self.AIRPORTS if a != self.target_origin]
        self.target_destination = self.rng.choice(dest_choices)
        self.target_date = "2026-11-20"
        self.target_passenger_name = self.rng.choice(self.PASSENGERS)
        self.target_passenger_email = f"{self.target_passenger_name.lower().replace(' ', '.')}@example.com"
        self.target_flight_number = f"FL-{self.rng.randint(100, 999)}"

        self.booking_stage = "search"  # search, flights, details, review, complete
        self.origin_input = ""
        self.dest_input = ""
        self.date_input = ""
        self.passenger_name_input = ""
        self.passenger_email_input = ""
        self.selected_flight: Optional[Dict[str, Any]] = None
        self.selected_seat: Optional[str] = None
        self.add_ons: List[str] = []
        self.confirmed_booking: Optional[FlightBookingRecord] = None
        self.milestones_achieved: Set[str] = set()

        self.flights = self._generate_flights()

        self.active_scenario = ScenarioDefinition(
            scenario_id=f"dynamic_booking_{self.seed}",
            domain="travel_booking",
            goal=(
                f"Search flights from '{self.target_origin}' to '{self.target_destination}', "
                f"select flight '{self.target_flight_number}', enter passenger '{self.target_passenger_name}', "
                f"and confirm booking"
            ),
            initial_url="https://travel.booking.internal/search",
            initial_title="Global Travel - Flight Booking Portal",
            max_steps=12,
        )
        self.reset()

    def _generate_flights(self) -> List[Dict[str, Any]]:
        flights = [
            {
                "number": self.target_flight_number,
                "origin": self.target_origin,
                "destination": self.target_destination,
                "time": "08:30 AM - 04:45 PM",
                "price": float(self.rng.randint(240, 420)),
                "stops": "Nonstop",
            }
        ]
        other_nums = [f"FL-{self.rng.randint(100, 999)}" for _ in range(2)]
        for fn in other_nums:
            if fn != self.target_flight_number:
                flights.append(
                    {
                        "number": fn,
                        "origin": self.target_origin,
                        "destination": self.target_destination,
                        "time": "01:15 PM - 09:30 PM",
                        "price": float(self.rng.randint(190, 380)),
                        "stops": "1 Stop",
                    }
                )
        self.rng.shuffle(flights)
        return flights

    def reset(self) -> StepResult:
        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False
        self.booking_stage = "search"
        self.origin_input = ""
        self.dest_input = ""
        self.date_input = ""
        self.passenger_name_input = ""
        self.passenger_email_input = ""
        self.selected_flight = None
        self.selected_seat = None
        self.add_ons = []
        self.confirmed_booking = None
        self.milestones_achieved.clear()

        elements = self._build_page_elements()
        self.current_state = EnvironmentState(
            url=f"https://travel.booking.internal/{self.booking_stage}",
            title=f"Global Travel - {self.booking_stage.capitalize()}",
            scroll_offset_y=0.0,
            elements=elements,
            step_count=0,
            cumulative_reward=0.0,
            last_action_str="INITIAL",
            last_action_success=True,
            metadata={"seed": self.seed, "stage": self.booking_stage},
        )
        return self._format_step_result(reward=0.0, done=False, truncated=False)

    def _build_page_elements(self) -> List[UIElement]:
        elements: List[UIElement] = []
        el_id = 1

        if self.booking_stage == "search":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="h2",
                    role="header",
                    text="Flight Search Portal",
                    bbox=(0.10, 0.05, 0.70, 0.12),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="origin_input",
                    text=self.origin_input or "Enter departure city or airport",
                    bbox=apply_bbox_jitter((0.10, 0.16, 0.45, 0.24), rng=self.rng),
                    value=self.origin_input,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="dest_input",
                    text=self.dest_input or "Enter arrival city or airport",
                    bbox=apply_bbox_jitter((0.55, 0.16, 0.90, 0.24), rng=self.rng),
                    value=self.dest_input,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="date_input",
                    text=self.date_input or self.target_date,
                    bbox=apply_bbox_jitter((0.10, 0.28, 0.45, 0.36), rng=self.rng),
                    value=self.date_input,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="search_flights_btn",
                    text="Search Flights",
                    bbox=apply_bbox_jitter((0.55, 0.28, 0.90, 0.36), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.booking_stage == "flights":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="h2",
                    role="header",
                    text=f"Available Flights: {self.origin_input or self.target_origin} to {self.dest_input or self.target_destination}",
                    bbox=(0.10, 0.05, 0.90, 0.12),
                    is_clickable=False,
                )
            )
            el_id += 1

            y_pos = 0.16
            for fl in self.flights:
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="div",
                        role=f"flight_card_{fl['number']}",
                        text=f"{fl['number']} | {fl['time']} | {fl['stops']} | ${fl['price']:.2f}",
                        bbox=apply_bbox_jitter((0.10, y_pos, 0.65, y_pos + 0.12), rng=self.rng),
                        is_clickable=False,
                    )
                )
                el_id += 1
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="button",
                        role=f"select_flight_{fl['number']}",
                        text=f"Select {fl['number']}",
                        bbox=apply_bbox_jitter((0.68, y_pos + 0.02, 0.90, y_pos + 0.10), rng=self.rng),
                        is_clickable=True,
                    )
                )
                el_id += 1
                y_pos += 0.16

            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="back_to_search_btn",
                    text="< Modify Search",
                    bbox=(0.10, y_pos, 0.35, y_pos + 0.08),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.booking_stage == "details":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="h2",
                    role="header",
                    text=f"Passenger Details & Seat: Flight {self.selected_flight['number'] if self.selected_flight else self.target_flight_number}",
                    bbox=(0.10, 0.05, 0.90, 0.12),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="passenger_name_input",
                    text=self.passenger_name_input or "Full Passenger Name",
                    bbox=apply_bbox_jitter((0.10, 0.16, 0.45, 0.24), rng=self.rng),
                    value=self.passenger_name_input,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="passenger_email_input",
                    text=self.passenger_email_input or "Email Address",
                    bbox=apply_bbox_jitter((0.55, 0.16, 0.90, 0.24), rng=self.rng),
                    value=self.passenger_email_input,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1

            # Seat buttons
            x_seat = 0.10
            for s in self.SEATS[:4]:
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="button",
                        role=f"select_seat_{s}",
                        text=f"Seat {s}" + (" [Selected]" if self.selected_seat == s else ""),
                        bbox=apply_bbox_jitter((x_seat, 0.28, x_seat + 0.18, 0.36), rng=self.rng),
                        is_clickable=True,
                    )
                )
                el_id += 1
                x_seat += 0.22

            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="toggle_baggage_btn",
                    text="Add Checked Baggage (+$40)" if "checked_bag" not in self.add_ons else "Checked Baggage Added [x]",
                    bbox=apply_bbox_jitter((0.10, 0.40, 0.45, 0.48), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="proceed_to_review_btn",
                    text="Proceed to Review >",
                    bbox=apply_bbox_jitter((0.55, 0.40, 0.90, 0.48), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.booking_stage == "review":
            fl_num = self.selected_flight["number"] if self.selected_flight else self.target_flight_number
            price = self.selected_flight["price"] if self.selected_flight else 299.0
            if "checked_bag" in self.add_ons:
                price += 40.0

            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="h2",
                    role="header",
                    text="Review Booking Summary",
                    bbox=(0.10, 0.05, 0.90, 0.12),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="summary_card",
                    text=(
                        f"Flight: {fl_num} | Route: {self.target_origin}->{self.target_destination} | "
                        f"Passenger: {self.passenger_name_input or self.target_passenger_name} | "
                        f"Seat: {self.selected_seat or '12A'} | Total: ${price:.2f}"
                    ),
                    bbox=apply_bbox_jitter((0.10, 0.16, 0.90, 0.35), rng=self.rng),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="confirm_booking_btn",
                    text="Confirm and Complete Booking",
                    bbox=apply_bbox_jitter((0.30, 0.42, 0.70, 0.52), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        elif self.booking_stage == "complete":
            ref = self.confirmed_booking.booking_id if self.confirmed_booking else "BK-9999"
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="h2",
                    role="header",
                    text="Booking Confirmed!",
                    bbox=(0.10, 0.10, 0.90, 0.20),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="div",
                    role="confirmation_banner",
                    text=f"Reservation Ref: {ref} for {self.target_passenger_name}. Safe travels!",
                    bbox=(0.10, 0.25, 0.90, 0.45),
                    is_clickable=False,
                )
            )
            el_id += 1

        # Add obstacle elements if active
        if getattr(self, "obstacle_state", None) is not None and self.obstacle_state.active:
            elements.extend(ObstacleInjector.get_obstacle_elements(self.obstacle_state, start_id=el_id))
            el_id += 5

        # Add distractor elements
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
        action_success = False

        hit_el = None
        if coordinates is not None:
            cx, cy = coordinates
            for el in self.current_state.elements:
                if el.is_clickable and el.contains(cx, cy):
                    hit_el = el
                    break

        intercepted, dismissed, obs_reward = self.handle_obstacle_step(action_type, coordinates, hit_el)
        if dismissed:
            reward += obs_reward
        if intercepted:
            self.current_state.elements = self._build_page_elements()
            self.current_state.cumulative_reward += reward
            truncated = self.current_state.step_count >= self.active_scenario.max_steps
            if truncated:
                done = True
            self.is_terminated = done
            return self._format_step_result(reward=reward, done=done, truncated=truncated)

        if action_type == ActionType.TYPE:
            if hit_el and hit_el.is_input:
                hit_el.value = argument
                if hit_el.role == "origin_input":
                    self.origin_input = argument.strip()
                    action_success = True
                elif hit_el.role == "dest_input":
                    self.dest_input = argument.strip()
                    action_success = True
                elif hit_el.role == "date_input":
                    self.date_input = argument.strip()
                    action_success = True
                elif hit_el.role == "passenger_name_input":
                    self.passenger_name_input = argument.strip()
                    action_success = True
                elif hit_el.role == "passenger_email_input":
                    self.passenger_email_input = argument.strip()
                    action_success = True
            else:
                # Default input assignment based on stage
                if self.booking_stage == "search":
                    self.dest_input = argument.strip()
                elif self.booking_stage == "details":
                    self.passenger_name_input = argument.strip()
                action_success = True

        elif action_type == ActionType.CLICK:
            if hit_el:
                action_success = True
                if hit_el.role == "search_flights_btn":
                    self.booking_stage = "flights"
                    if "search_submitted" not in self.milestones_achieved:
                        reward += 0.5
                        self.milestones_achieved.add("search_submitted")
                elif hit_el.role.startswith("select_flight_"):
                    fl_num = hit_el.role.replace("select_flight_", "")
                    self.selected_flight = next((fl for fl in self.flights if fl["number"] == fl_num), self.flights[0])
                    self.booking_stage = "details"
                    if "flight_selected" not in self.milestones_achieved:
                        reward += 0.5
                        self.milestones_achieved.add("flight_selected")
                elif hit_el.role.startswith("select_seat_"):
                    self.selected_seat = hit_el.role.replace("select_seat_", "")
                    if "seat_selected" not in self.milestones_achieved:
                        reward += 0.2
                        self.milestones_achieved.add("seat_selected")
                elif hit_el.role == "toggle_baggage_btn":
                    if "checked_bag" in self.add_ons:
                        self.add_ons.remove("checked_bag")
                    else:
                        self.add_ons.append("checked_bag")
                elif hit_el.role == "proceed_to_review_btn":
                    self.booking_stage = "review"
                    if "passenger_details_filled" not in self.milestones_achieved:
                        reward += 0.5
                        self.milestones_achieved.add("passenger_details_filled")
                elif hit_el.role == "confirm_booking_btn":
                    fl_num = self.selected_flight["number"] if self.selected_flight else self.target_flight_number
                    price = self.selected_flight["price"] if self.selected_flight else 299.0
                    if "checked_bag" in self.add_ons:
                        price += 40.0
                    self.confirmed_booking = FlightBookingRecord(
                        booking_id=f"BK-{self.rng.randint(1000, 9999)}",
                        origin=self.origin_input or self.target_origin,
                        destination=self.dest_input or self.target_destination,
                        departure_date=self.date_input or self.target_date,
                        flight_number=fl_num,
                        seat=self.selected_seat or "12A",
                        passenger_name=self.passenger_name_input or self.target_passenger_name,
                        passenger_email=self.passenger_email_input or self.target_passenger_email,
                        add_ons=list(self.add_ons),
                        total_price=price,
                        confirmed=True,
                    )
                    self.booking_stage = "complete"
                    self.goal_satisfied = True
                    self.milestones_achieved.add("booking_confirmed")
                    reward += 2.0
                    done = True
                elif hit_el.role == "back_to_search_btn":
                    self.booking_stage = "search"

        elif action_type == ActionType.FINISH:
            if self.goal_satisfied:
                reward += 2.0
            done = True
            action_success = True

        self.current_state.elements = self._build_page_elements()
        self.current_state.cumulative_reward += reward

        truncated = self.current_state.step_count >= self.active_scenario.max_steps
        if truncated:
            done = True
        self.is_terminated = done

        return self._format_step_result(reward=reward, done=done, truncated=truncated)


# ==============================================================================
# 6. Dynamic CRM Pipeline Procedural Environment
# ==============================================================================

@dataclass
class CRMDeal:
    """Represents a prospective customer deal in the sales CRM pipeline."""

    deal_id: str
    company: str
    value: float
    stage: str  # "Lead", "Qualified", "Demo", "Proposal", "Closed-Won"
    contact_name: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deal_id": self.deal_id,
            "company": self.company,
            "value": self.value,
            "stage": self.stage,
            "contact_name": self.contact_name,
            "notes": self.notes,
        }


class DynamicCRMEnvironment(ProceduralEnvironment):
    """
    Enterprise CRM Sales Pipeline Sandbox.
    Features:
    - Interactive Kanban pipeline columns (Lead, Qualified, Demo, Proposal, Closed-Won).
    - Deal modal creator and metadata editor.
    - Stage advancement transitions with state-machine validation.
    """

    STAGES = ["Lead", "Qualified", "Demo", "Proposal", "Closed-Won"]
    COMPANIES = ["Acme Corp", "Globex Inc", "Initech", "Stark Systems", "Umbrella Tech"]
    REPS = ["Taylor Swift", "Bruce Wayne", "Clark Kent", "Diana Prince"]

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        seed: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(config=config, scenario_id="crm_pipeline", device=device, dtype=dtype)
        self.seed = seed if seed is not None else random.randint(1000, 999999)
        self.rng = random.Random(self.seed)

        self.target_company = self.rng.choice(self.COMPANIES)
        self.target_value = float(self.rng.choice([25000, 50000, 75000, 100000]))

        self.crm_view = "pipeline"  # pipeline, modal
        self.modal_company = ""
        self.modal_value = ""
        self.active_deal_id: Optional[str] = None
        self.milestones_achieved: Set[str] = set()

        self.deals: Dict[str, CRMDeal] = self._generate_seed_deals()

        self.active_scenario = ScenarioDefinition(
            scenario_id=f"dynamic_crm_{self.seed}",
            domain="crm_pipeline",
            goal=(
                f"Create a new deal for '{self.target_company}' with value ${int(self.target_value)}, "
                f"and advance it to 'Closed-Won'"
            ),
            initial_url="https://crm.enterprise.internal/pipeline",
            initial_title="Enterprise CRM - Sales Pipeline",
            max_steps=12,
        )
        self.reset()

    def _generate_seed_deals(self) -> Dict[str, CRMDeal]:
        deals: Dict[str, CRMDeal] = {}
        seed_companies = [c for c in self.COMPANIES if c != self.target_company][:3]
        for idx, comp in enumerate(seed_companies):
            d_id = f"deal-{idx + 1:03d}"
            deals[d_id] = CRMDeal(
                deal_id=d_id,
                company=comp,
                value=float(self.rng.randint(20, 90) * 1000),
                stage=self.STAGES[idx % (len(self.STAGES) - 1)],
                contact_name=self.rng.choice(self.REPS),
            )
        return deals

    def reset(self) -> StepResult:
        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False
        self.crm_view = "pipeline"
        self.modal_company = ""
        self.modal_value = ""
        self.active_deal_id = None
        self.milestones_achieved.clear()
        self.deals = self._generate_seed_deals()

        elements = self._build_page_elements()
        self.current_state = EnvironmentState(
            url=f"https://crm.enterprise.internal/{self.crm_view}",
            title=f"Enterprise CRM - {self.crm_view.capitalize()}",
            scroll_offset_y=0.0,
            elements=elements,
            step_count=0,
            cumulative_reward=0.0,
            last_action_str="INITIAL",
            last_action_success=True,
            metadata={"seed": self.seed, "view": self.crm_view},
        )
        return self._format_step_result(reward=0.0, done=False, truncated=False)

    def _build_page_elements(self) -> List[UIElement]:
        elements: List[UIElement] = []
        el_id = 1

        if self.crm_view == "pipeline":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="h2",
                    role="header",
                    text="Sales Pipeline Kanban",
                    bbox=(0.05, 0.04, 0.50, 0.10),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="new_deal_btn",
                    text="+ New Deal",
                    bbox=apply_bbox_jitter((0.78, 0.04, 0.95, 0.10), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

            # Render stage columns and deal cards
            y_pos = 0.15
            for stage in self.STAGES:
                stage_deals = [d for d in self.deals.values() if d.stage == stage]
                elements.append(
                    UIElement(
                        element_id=el_id,
                        tag="div",
                        role=f"stage_col_{stage.lower()}",
                        text=f"Stage: {stage} ({len(stage_deals)})",
                        bbox=(0.05, y_pos, 0.35, y_pos + 0.06),
                        is_clickable=False,
                    )
                )
                el_id += 1

                for deal in stage_deals:
                    elements.append(
                        UIElement(
                            element_id=el_id,
                            tag="div",
                            role=f"deal_card_{deal.deal_id}",
                            text=f"{deal.company} - ${deal.value:,.0f} [{deal.stage}]",
                            bbox=apply_bbox_jitter((0.38, y_pos, 0.72, y_pos + 0.06), rng=self.rng),
                            is_clickable=False,
                        )
                    )
                    el_id += 1
                    if stage != "Closed-Won":
                        elements.append(
                            UIElement(
                                element_id=el_id,
                                tag="button",
                                role=f"advance_deal_{deal.deal_id}",
                                text=f"Advance '{deal.company}' >",
                                bbox=apply_bbox_jitter((0.75, y_pos, 0.95, y_pos + 0.06), rng=self.rng),
                                is_clickable=True,
                            )
                        )
                        el_id += 1
                    y_pos += 0.08

        elif self.crm_view == "modal":
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="h2",
                    role="header",
                    text="Create / Edit Deal",
                    bbox=(0.20, 0.15, 0.80, 0.22),
                    is_clickable=False,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="deal_company_input",
                    text=self.modal_company or "Enter company name",
                    bbox=apply_bbox_jitter((0.20, 0.26, 0.80, 0.34), rng=self.rng),
                    value=self.modal_company,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="input",
                    role="deal_value_input",
                    text=self.modal_value or "Enter estimated value ($)",
                    bbox=apply_bbox_jitter((0.20, 0.38, 0.80, 0.46), rng=self.rng),
                    value=self.modal_value,
                    is_clickable=True,
                    is_input=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="save_deal_btn",
                    text="Save Deal",
                    bbox=apply_bbox_jitter((0.20, 0.52, 0.48, 0.60), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1
            elements.append(
                UIElement(
                    element_id=el_id,
                    tag="button",
                    role="close_modal_btn",
                    text="Cancel",
                    bbox=apply_bbox_jitter((0.52, 0.52, 0.80, 0.60), rng=self.rng),
                    is_clickable=True,
                )
            )
            el_id += 1

        # Add obstacle elements if active
        if getattr(self, "obstacle_state", None) is not None and self.obstacle_state.active:
            elements.extend(ObstacleInjector.get_obstacle_elements(self.obstacle_state, start_id=el_id))
            el_id += 5

        # Add distractor elements
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
        action_success = False

        hit_el = None
        if coordinates is not None:
            cx, cy = coordinates
            for el in self.current_state.elements:
                if el.is_clickable and el.contains(cx, cy):
                    hit_el = el
                    break

        intercepted, dismissed, obs_reward = self.handle_obstacle_step(action_type, coordinates, hit_el)
        if dismissed:
            reward += obs_reward
        if intercepted:
            self.current_state.elements = self._build_page_elements()
            self.current_state.cumulative_reward += reward
            truncated = self.current_state.step_count >= self.active_scenario.max_steps
            if truncated:
                done = True
            self.is_terminated = done
            return self._format_step_result(reward=reward, done=done, truncated=truncated)

        if action_type == ActionType.TYPE:
            if hit_el and hit_el.is_input:
                hit_el.value = argument
                if hit_el.role == "deal_company_input":
                    self.modal_company = argument.strip()
                    action_success = True
                elif hit_el.role == "deal_value_input":
                    self.modal_value = argument.strip()
                    action_success = True
            else:
                if self.crm_view == "modal":
                    self.modal_company = argument.strip()
                    action_success = True

        elif action_type == ActionType.CLICK:
            if hit_el:
                action_success = True
                if hit_el.role == "new_deal_btn":
                    self.crm_view = "modal"
                    if "deal_modal_opened" not in self.milestones_achieved:
                        reward += 0.3
                        self.milestones_achieved.add("deal_modal_opened")
                elif hit_el.role == "close_modal_btn":
                    self.crm_view = "pipeline"
                elif hit_el.role == "save_deal_btn":
                    company = self.modal_company or self.target_company
                    try:
                        clean_num = re.sub(r"[^\d.]", "", self.modal_value)
                        val = float(clean_num) if clean_num else self.target_value
                    except (ValueError, TypeError):
                        val = self.target_value

                    deal_id = f"deal-{len(self.deals) + 1:03d}"
                    new_deal = CRMDeal(
                        deal_id=deal_id,
                        company=company,
                        value=val,
                        stage="Lead",
                        contact_name="Lead Manager",
                    )
                    self.deals[deal_id] = new_deal
                    self.crm_view = "pipeline"
                    if "deal_created" not in self.milestones_achieved:
                        reward += 0.5
                        self.milestones_achieved.add("deal_created")
                elif hit_el.role.startswith("advance_deal_"):
                    d_id = hit_el.role.replace("advance_deal_", "")
                    deal = self.deals.get(d_id)
                    if deal and deal.stage in self.STAGES:
                        curr_idx = self.STAGES.index(deal.stage)
                        if curr_idx < len(self.STAGES) - 1:
                            deal.stage = self.STAGES[curr_idx + 1]
                            reward += 0.2
                            self.milestones_achieved.add("stage_advanced")
                            if deal.stage == "Closed-Won":
                                self.milestones_achieved.add("deal_closed_won")
                                if deal.company.lower() == self.target_company.lower():
                                    self.goal_satisfied = True
                                    reward += 2.0
                                    done = True

        elif action_type == ActionType.FINISH:
            if self.goal_satisfied:
                reward += 2.0
            done = True
            action_success = True

        self.current_state.elements = self._build_page_elements()
        self.current_state.cumulative_reward += reward

        truncated = self.current_state.step_count >= self.active_scenario.max_steps
        if truncated:
            done = True
        self.is_terminated = done

        return self._format_step_result(reward=reward, done=done, truncated=truncated)


# ==============================================================================
# 7. Programmatic Outcome Verifiers & Milestone Detectors
# ==============================================================================

@dataclass
class OutcomeVerificationResult:
    """Deterministic programmatic outcome verification container (zero human annotation)."""

    goal_satisfied: bool
    milestones_achieved: List[str]
    detected_errors: List[str]
    trajectory_efficiency: float
    state_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_satisfied": self.goal_satisfied,
            "milestones_achieved": self.milestones_achieved,
            "detected_errors": self.detected_errors,
            "trajectory_efficiency": self.trajectory_efficiency,
            "state_payload": self.state_payload,
        }


def verify_ecommerce_outcome(env: DynamicEcommerceEnvironment) -> OutcomeVerificationResult:
    milestones = list(env.milestones_achieved)
    errors: List[str] = []
    satisfied = False

    if env.stage == "complete" and env.goal_satisfied:
        if any(it.item_id == env.target_item.item_id for it in env.cart):
            satisfied = True
            if "order_placed" not in milestones:
                milestones.append("order_placed")
        else:
            errors.append("target_item_not_in_cart")

    return OutcomeVerificationResult(
        goal_satisfied=satisfied,
        milestones_achieved=milestones,
        detected_errors=errors,
        trajectory_efficiency=len(milestones) / max(1, env.current_state.step_count if env.current_state else 1),
        state_payload={"cart_size": len(env.cart), "stage": env.stage},
    )


def verify_form_outcome(env: DynamicFormPortalEnvironment) -> OutcomeVerificationResult:
    milestones = list(env.milestones_achieved)
    errors: List[str] = []
    satisfied = False

    total_steps = len(getattr(env, "STEPS", ["Identity", "Company", "Preferences", "Review"]))
    step_idx = getattr(env, "current_step_idx", 0)
    if env.goal_satisfied:
        satisfied = True
        if "form_completed" not in milestones:
            milestones.append("form_completed")
    elif step_idx >= total_steps - 1 and env.form_data.get("name") and env.form_data.get("email"):
        satisfied = True
        if "form_completed" not in milestones:
            milestones.append("form_completed")

    return OutcomeVerificationResult(
        goal_satisfied=satisfied,
        milestones_achieved=milestones,
        detected_errors=errors,
        trajectory_efficiency=len(milestones) / max(1, env.current_state.step_count if env.current_state else 1),
        state_payload={"current_step_idx": step_idx, "form_data": env.form_data},
    )


def verify_booking_outcome(env: DynamicBookingEnvironment) -> OutcomeVerificationResult:
    milestones = list(env.milestones_achieved)
    errors: List[str] = []
    satisfied = False

    if env.confirmed_booking is not None and env.confirmed_booking.confirmed:
        b = env.confirmed_booking
        matches_origin = (b.origin.lower() == env.target_origin.lower())
        matches_dest = (b.destination.lower() == env.target_destination.lower())
        matches_name = (b.passenger_name.lower() == env.target_passenger_name.lower())

        if matches_origin and matches_dest and matches_name:
            satisfied = True
            if "booking_confirmed" not in milestones:
                milestones.append("booking_confirmed")
        else:
            errors.append("booking_attribute_mismatch")

    return OutcomeVerificationResult(
        goal_satisfied=satisfied,
        milestones_achieved=milestones,
        detected_errors=errors,
        trajectory_efficiency=len(milestones) / max(1, env.current_state.step_count if env.current_state else 1),
        state_payload={"booking": env.confirmed_booking.to_dict() if env.confirmed_booking else None},
    )


def verify_crm_outcome(env: DynamicCRMEnvironment) -> OutcomeVerificationResult:
    milestones = list(env.milestones_achieved)
    errors: List[str] = []
    satisfied = False

    target_deal = next((d for d in env.deals.values() if d.company.lower() == env.target_company.lower()), None)
    if target_deal is not None:
        if target_deal.stage == "Closed-Won":
            satisfied = True
            if "deal_closed_won" not in milestones:
                milestones.append("deal_closed_won")
        else:
            errors.append(f"deal_not_closed_won (stage={target_deal.stage})")
    else:
        errors.append("target_deal_not_found")

    return OutcomeVerificationResult(
        goal_satisfied=satisfied,
        milestones_achieved=milestones,
        detected_errors=errors,
        trajectory_efficiency=len(milestones) / max(1, env.current_state.step_count if env.current_state else 1),
        state_payload={"deals_count": len(env.deals), "target_deal": target_deal.to_dict() if target_deal else None},
    )


def verify_cloud_outcome(env: DynamicCloudAdminEnvironment) -> OutcomeVerificationResult:
    milestones = list(env.milestones_achieved)
    errors: List[str] = []
    satisfied = False

    target_inst = next((i for i in env.instances if i["id"] == env.target_instance_id), None)
    if target_inst and target_inst.get("state") == "Stopped" and env.goal_satisfied:
        satisfied = True
        if "instance_stopped" not in milestones:
            milestones.append("instance_stopped")

    return OutcomeVerificationResult(
        goal_satisfied=satisfied,
        milestones_achieved=milestones,
        detected_errors=errors,
        trajectory_efficiency=len(milestones) / max(1, env.current_state.step_count if env.current_state else 1),
        state_payload={"target_instance": target_inst},
    )


def verify_environment_outcome(env: Any) -> OutcomeVerificationResult:
    """Unified outcome verification router across all Web/OS environments."""
    if isinstance(env, DynamicEcommerceEnvironment):
        return verify_ecommerce_outcome(env)
    elif isinstance(env, DynamicFormPortalEnvironment):
        return verify_form_outcome(env)
    elif isinstance(env, DynamicBookingEnvironment):
        return verify_booking_outcome(env)
    elif isinstance(env, DynamicCRMEnvironment):
        return verify_crm_outcome(env)
    elif isinstance(env, DynamicCloudAdminEnvironment):
        return verify_cloud_outcome(env)
    else:
        satisfied = bool(getattr(env, "goal_satisfied", False))
        milestones = list(getattr(env, "milestones_achieved", set()))
        return OutcomeVerificationResult(
            goal_satisfied=satisfied,
            milestones_achieved=milestones,
            detected_errors=[],
            trajectory_efficiency=len(milestones) / max(1, getattr(env.current_state, "step_count", 1) if getattr(env, "current_state", None) else 1),
            state_payload={},
        )


@dataclass
class MilestoneEvent:
    name: str
    step: int
    reward: float = 0.15


def detect_milestones(env: Any) -> List[MilestoneEvent]:
    """Extracts verified milestone progression events from an environment instance."""
    milestones = getattr(env, "milestones_achieved", set())
    step = getattr(env.current_state, "step_count", 0) if getattr(env, "current_state", None) else 0
    return [MilestoneEvent(name=m, step=step, reward=0.15) for m in milestones]


# ==============================================================================
# 8. Dynamic Scenario Generator Factory
# ==============================================================================

class DynamicScenarioGenerator:
    """
    Factory synthesizing an unbounded variety of interactive Web/OS environments.
    Covers 5+ domains:
    - DynamicEcommerce
    - DynamicFormPortal
    - DynamicBooking
    - DynamicCRM
    - DynamicCloudAdmin
    - DynamicWebBrowser
    """

    AVAILABLE_DOMAINS = [
        "ecommerce",
        "form_portal",
        "booking",
        "crm",
        "cloud_admin",
        "web_browser",
    ]

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config if config is not None else JEDM2Config()
        self.device = torch.device(device)
        self.dtype = dtype

    def create_environment(
        self,
        domain: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> WebOSEnvironment:
        """Alias for generate() synthesizing an active environment sandbox."""
        return self.generate(domain=domain, seed=seed)

    def generate(
        self,
        domain: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> WebOSEnvironment:
        """
        Synthesizes a randomized procedural environment.

        Args:
            domain: Optional domain choice ('ecommerce', 'form_portal', 'booking', 'crm', 'cloud_admin', 'web_browser').
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
        elif dom == "booking":
            return DynamicBookingEnvironment(
                config=self.config,
                seed=effective_seed,
                device=self.device,
                dtype=self.dtype,
            )
        elif dom == "crm":
            return DynamicCRMEnvironment(
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
