"""
JED-M2 Simulated Interactive Web/OS Environment Sandbox.
Provides:
- UIElement: Interactable DOM / GUI element with spatial bounding box in [0, 1].
- EnvironmentState: Snapshot of current page/desktop state, active element tree, and viewport.
- StepResult: Environment feedback tuple (obs_text, obs_features, reward, done, truncated, info).
- BenchmarkScenario: Multi-turn benchmark workflow definitions (ecommerce, travel, os_admin).
- WebOSEnvironment: Stateful digital sandbox implementing realistic multi-turn interaction
  transitions for discrete primitives (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH),
  spatial hit-testing in [0, 1] x [0, 1], and deterministic ground-truth goal evaluation.

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
"""

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch

from jed_m2.config import JEDM2Config
from jed_m2.core.types import UIElement
from jed_m2.heads import ActionType


# ==============================================================================
# 1. UI Element & Environment State Data Structures
# ==============================================================================

@dataclass
class EnvironmentState:
    """Represents a discrete snapshot of the interactive environment."""

    url: str
    title: str
    scroll_offset_y: float
    elements: List[UIElement]
    focused_element_id: Optional[int] = None
    step_count: int = 0
    cumulative_reward: float = 0.0
    last_action_str: str = "INITIAL"
    last_action_success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Feedback returned after taking an environment step."""

    obs_text: str  # Serialized observation string for LM reasoning
    obs_features: torch.Tensor  # Dense state embedding [hidden_dim] for JEPA world model
    reward: float  # Step-level reward signal
    done: bool  # Whether task has terminated
    truncated: bool  # Whether step horizon limit was reached
    info: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 2. Benchmark Scenario Definitions
# ==============================================================================

@dataclass
class ScenarioDefinition:
    """Specification of a simulated multi-turn interactive task."""

    scenario_id: str
    domain: str
    goal: str
    initial_url: str
    initial_title: str
    max_steps: int = 10


def get_default_scenarios() -> Dict[str, ScenarioDefinition]:
    """Returns catalog of built-in benchmark task scenarios."""
    return {
        "ecommerce": ScenarioDefinition(
            scenario_id="ecommerce",
            domain="shopping",
            goal="Search for noise-cancelling headphones and add top result to cart",
            initial_url="https://shop.internal/home",
            initial_title="Global Marketplace - Home",
            max_steps=8,
        ),
        "travel": ScenarioDefinition(
            scenario_id="travel",
            domain="travel",
            goal="Book hotel in San Francisco for upcoming weekend",
            initial_url="https://travel.booking.internal/hotels",
            initial_title="Travel Booking Portal",
            max_steps=8,
        ),
        "os_admin": ScenarioDefinition(
            scenario_id="os_admin",
            domain="cloud_infrastructure",
            goal="Update security group rules in cloud administration console",
            initial_url="https://console.cloud.internal/vpc",
            initial_title="Cloud Administration Console",
            max_steps=8,
        ),
    }


# ==============================================================================
# 3. Simulated Web/OS Environment State Machine
# ==============================================================================

class WebOSEnvironment:
    """
    Simulated interactive Web/OS environment sandbox.
    Models DOM / GUI state machines, spatial bounding boxes, action execution,
    deterministic reward calculation, and goal verification.
    """

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        scenario_id: str = "ecommerce",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config if config is not None else JEDM2Config()
        self.hidden_dim = self.config.hidden_dim
        self.device = torch.device(device)
        self.dtype = dtype

        self.scenarios = get_default_scenarios()
        self.active_scenario = self.scenarios.get(
            scenario_id, self.scenarios["ecommerce"]
        )

        # Internal state machine attributes
        self.current_state: Optional[EnvironmentState] = None
        self.action_history: List[Dict[str, Any]] = []
        self.goal_satisfied: bool = False
        self.is_terminated: bool = False

    # --------------------------------------------------------------------------
    # Scenario Page Builders
    # --------------------------------------------------------------------------

    def _build_ecommerce_page(self, stage: str) -> Tuple[str, str, List[UIElement]]:
        """Generates realistic UI element trees for ecommerce scenario stages."""
        elements: List[UIElement] = []

        if stage == "home":
            url = "https://shop.internal/home"
            title = "Global Marketplace - Home"
            elements.append(
                UIElement(
                    element_id=1,
                    tag="input",
                    role="search_bar",
                    text="Search products, brands, and categories",
                    bbox=(0.20, 0.08, 0.70, 0.14),
                    is_clickable=True,
                    is_input=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=2,
                    tag="button",
                    role="submit_btn",
                    text="Search",
                    bbox=(0.72, 0.08, 0.85, 0.14),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=3,
                    tag="a",
                    role="nav_link",
                    text="Deals of the Day",
                    bbox=(0.10, 0.18, 0.30, 0.23),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=4,
                    tag="div",
                    role="item_card",
                    text="Trending: Wireless Earbuds",
                    bbox=(0.15, 0.30, 0.45, 0.55),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=5,
                    tag="div",
                    role="item_card",
                    text="Featured: Smart Home Hub",
                    bbox=(0.55, 0.30, 0.85, 0.55),
                    is_clickable=True,
                )
            )

        elif stage == "results":
            url = "https://shop.internal/search?q=headphones"
            title = "Search Results - Headphones"
            elements.append(
                UIElement(
                    element_id=1,
                    tag="input",
                    role="search_bar",
                    text="Sony WH-1000XM5",
                    bbox=(0.20, 0.08, 0.70, 0.14),
                    value="Sony WH-1000XM5",
                    is_clickable=True,
                    is_input=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=2,
                    tag="button",
                    role="submit_btn",
                    text="Search",
                    bbox=(0.72, 0.08, 0.85, 0.14),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=10,
                    tag="a",
                    role="first_product_link",
                    text="Sony WH-1000XM5 Wireless Noise-Cancelling Headphones ($398.00)",
                    bbox=(0.15, 0.25, 0.85, 0.45),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=11,
                    tag="a",
                    role="second_product_link",
                    text="Bose QuietComfort 45 Headphones ($329.00)",
                    bbox=(0.15, 0.50, 0.85, 0.70),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=12,
                    tag="div",
                    role="pagination",
                    text="Page 1 of 5 | Next >",
                    bbox=(0.40, 0.85, 0.60, 0.92),
                    is_scrollable=True,
                )
            )

        elif stage == "product_details":
            url = "https://shop.internal/products/sony-wh1000xm5"
            title = "Sony WH-1000XM5 - Details"
            elements.append(
                UIElement(
                    element_id=20,
                    tag="h1",
                    role="product_title",
                    text="Sony WH-1000XM5 Noise Cancelling Headphones",
                    bbox=(0.10, 0.10, 0.80, 0.20),
                    is_clickable=False,
                )
            )
            elements.append(
                UIElement(
                    element_id=21,
                    tag="span",
                    role="price_tag",
                    text="$398.00 In Stock",
                    bbox=(0.10, 0.22, 0.35, 0.28),
                    is_clickable=False,
                )
            )
            elements.append(
                UIElement(
                    element_id=22,
                    tag="button",
                    role="checkout_btn",
                    text="Add to Cart",
                    bbox=(0.10, 0.35, 0.40, 0.44),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=23,
                    tag="button",
                    role="buy_now_btn",
                    text="Buy Now with 1-Click",
                    bbox=(0.45, 0.35, 0.75, 0.44),
                    is_clickable=True,
                )
            )

        elif stage == "cart_confirmed":
            url = "https://shop.internal/cart"
            title = "Shopping Cart (1 Item)"
            elements.append(
                UIElement(
                    element_id=30,
                    tag="div",
                    role="alert_banner",
                    text="Added to Cart: Sony WH-1000XM5 Wireless Headphones",
                    bbox=(0.10, 0.12, 0.90, 0.22),
                    is_clickable=False,
                )
            )
            elements.append(
                UIElement(
                    element_id=31,
                    tag="button",
                    role="proceed_checkout",
                    text="Proceed to Checkout ($398.00)",
                    bbox=(0.25, 0.35, 0.75, 0.45),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=32,
                    tag="a",
                    role="continue_shopping",
                    text="Continue Shopping",
                    bbox=(0.35, 0.50, 0.65, 0.56),
                    is_clickable=True,
                )
            )
        else:
            url = "https://shop.internal/error"
            title = "Not Found"
            elements = []

        return url, title, elements

    def _build_travel_page(self, stage: str) -> Tuple[str, str, List[UIElement]]:
        """Generates realistic UI element trees for travel booking scenario stages."""
        elements: List[UIElement] = []

        if stage == "home":
            url = "https://travel.booking.internal/hotels"
            title = "Travel Booking Portal - Hotels"
            elements.append(
                UIElement(
                    element_id=1,
                    tag="input",
                    role="location_input",
                    text="Where are you going?",
                    bbox=(0.15, 0.15, 0.55, 0.23),
                    is_clickable=True,
                    is_input=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=2,
                    tag="input",
                    role="date_picker",
                    text="Check-in - Check-out Dates",
                    bbox=(0.60, 0.15, 0.85, 0.23),
                    is_clickable=True,
                    is_input=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=3,
                    tag="button",
                    role="submit_btn",
                    text="Search Hotels",
                    bbox=(0.35, 0.28, 0.65, 0.36),
                    is_clickable=True,
                )
            )
        elif stage == "results":
            url = "https://travel.booking.internal/hotels?city=san_francisco"
            title = "Hotels in San Francisco, CA"
            elements.append(
                UIElement(
                    element_id=10,
                    tag="div",
                    role="item_card",
                    text="Grand Bayview Hotel SF - $220/night",
                    bbox=(0.10, 0.15, 0.90, 0.35),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=11,
                    tag="button",
                    role="reserve_room",
                    text="Reserve Room",
                    bbox=(0.65, 0.25, 0.85, 0.32),
                    is_clickable=True,
                )
            )
        elif stage == "booked":
            url = "https://travel.booking.internal/confirmation"
            title = "Reservation Confirmed"
            elements.append(
                UIElement(
                    element_id=20,
                    tag="div",
                    role="confirmation_box",
                    text="Booking Confirmed: Grand Bayview Hotel SF",
                    bbox=(0.15, 0.20, 0.85, 0.50),
                    is_clickable=False,
                )
            )
        else:
            url = "https://travel.booking.internal/error"
            title = "Error"

        return url, title, elements

    def _build_os_admin_page(self, stage: str) -> Tuple[str, str, List[UIElement]]:
        """Generates realistic UI element trees for cloud administration scenario stages."""
        elements: List[UIElement] = []

        if stage == "home":
            url = "https://console.cloud.internal/vpc"
            title = "Cloud Administration Console - VPC Dashboard"
            elements.append(
                UIElement(
                    element_id=1,
                    tag="a",
                    role="nav_security",
                    text="Security Groups",
                    bbox=(0.05, 0.15, 0.25, 0.22),
                    is_clickable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=2,
                    tag="div",
                    role="vpc_overview",
                    text="VPC default-vpc-01 (10.0.0.0/16)",
                    bbox=(0.30, 0.15, 0.90, 0.40),
                )
            )
        elif stage == "security_groups":
            url = "https://console.cloud.internal/vpc/security-groups"
            title = "Security Groups Management"
            elements.append(
                UIElement(
                    element_id=10,
                    tag="div",
                    role="group_table",
                    text="sg-01a2b: Default Web Tier Inbound Rules",
                    bbox=(0.10, 0.20, 0.90, 0.50),
                    is_scrollable=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=11,
                    tag="button",
                    role="edit_inbound_rules",
                    text="Edit Inbound Rules",
                    bbox=(0.70, 0.25, 0.88, 0.32),
                    is_clickable=True,
                )
            )
        elif stage == "edit_rules":
            url = "https://console.cloud.internal/vpc/security-groups/edit"
            title = "Edit Security Group Inbound Rules"
            elements.append(
                UIElement(
                    element_id=20,
                    tag="input",
                    role="rule_api",
                    text="Enter CLI / API rule command",
                    bbox=(0.15, 0.20, 0.85, 0.30),
                    is_clickable=True,
                    is_input=True,
                )
            )
            elements.append(
                UIElement(
                    element_id=21,
                    tag="button",
                    role="save_rules",
                    text="Save Inbound Rules",
                    bbox=(0.40, 0.40, 0.60, 0.48),
                    is_clickable=True,
                )
            )
        elif stage == "rules_saved":
            url = "https://console.cloud.internal/vpc/security-groups"
            title = "Security Groups - Updated"
            elements.append(
                UIElement(
                    element_id=30,
                    tag="div",
                    role="success_alert",
                    text="Security group rules updated successfully.",
                    bbox=(0.15, 0.15, 0.85, 0.25),
                )
            )
        else:
            url = "https://console.cloud.internal/error"
            title = "Console Error"

        return url, title, elements

    # --------------------------------------------------------------------------
    # Observation Generator & Spatial Hit-Testing
    # --------------------------------------------------------------------------

    def hit_test(self, x: float, y: float) -> Optional[UIElement]:
        """
        Performs continuous spatial hit-testing across active UI elements.
        Returns the top-most (smallest area) element intersecting (x, y).
        """
        if self.current_state is None:
            return None

        hits = [el for el in self.current_state.elements if el.contains(x, y)]
        if not hits:
            return None

        # Return the most specific element (smallest bounding box area)
        hits.sort(key=lambda el: el.area)
        return hits[0]

    def _generate_obs_features(self) -> torch.Tensor:
        """
        Generates deterministic dense observation embedding [hidden_dim]
        for JEPA latent world model conditioning.
        """
        state_key = f"{self.active_scenario.scenario_id}_{self.current_state.url}_{self.current_state.step_count}"
        seed = int(hashlib.md5(state_key.encode("utf-8")).hexdigest()[:8], 16)
        gen = torch.Generator().manual_seed(seed)
        # Random normal feature vector: [hidden_dim]
        features = torch.randn(
            self.hidden_dim, generator=gen
        ).to(device=self.device, dtype=self.dtype)
        return features

    def _format_observation_text(self) -> str:
        """Serializes current state and visible elements into clean prompt observation."""
        if self.current_state is None:
            return ""

        dom_repr = " ".join(
            el.to_dom_string() for el in self.current_state.elements[:8]
        )
        obs_text = (
            f"Goal: {self.active_scenario.goal} | "
            f"URL: {self.current_state.url} | "
            f"Title: {self.current_state.title} | "
            f"Screen State: {dom_repr}"
        )
        return obs_text

    # --------------------------------------------------------------------------
    # Environment Lifecycle: Reset & Step
    # --------------------------------------------------------------------------

    def reset(
        self,
        scenario_id: Optional[str] = None,
        custom_goal: Optional[str] = None,
    ) -> StepResult:
        """
        Resets environment to initial state of chosen scenario.
        """
        if scenario_id is not None and scenario_id in self.scenarios:
            self.active_scenario = self.scenarios[scenario_id]

        if custom_goal is not None:
            self.active_scenario.goal = custom_goal

        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False

        # Load initial stage based on scenario domain
        if self.active_scenario.scenario_id == "ecommerce":
            url, title, elements = self._build_ecommerce_page("home")
            meta = {"stage": "home", "cart": []}
        elif self.active_scenario.scenario_id == "travel":
            url, title, elements = self._build_travel_page("home")
            meta = {"stage": "home", "destination": "", "booked": False}
        elif self.active_scenario.scenario_id == "os_admin":
            url, title, elements = self._build_os_admin_page("home")
            meta = {"stage": "home", "rules": []}
        else:
            url, title, elements = self._build_ecommerce_page("home")
            meta = {"stage": "home"}

        self.current_state = EnvironmentState(
            url=url,
            title=title,
            scroll_offset_y=0.0,
            elements=elements,
            focused_element_id=None,
            step_count=0,
            cumulative_reward=0.0,
            last_action_str="RESET",
            last_action_success=True,
            metadata=meta,
        )

        obs_text = self._format_observation_text()
        obs_features = self._generate_obs_features()

        return StepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=0.0,
            done=False,
            truncated=False,
            info={
                "step": 0,
                "goal": self.active_scenario.goal,
                "scenario": self.active_scenario.scenario_id,
                "url": url,
            },
        )

    def step(
        self,
        action_type: Union[ActionType, int, str],
        coordinates: Optional[Tuple[float, float]] = None,
        argument: Optional[str] = None,
    ) -> StepResult:
        """
        Executes an agent action primitive in the simulated sandbox.

        Args:
            action_type: Primitive identifier (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH).
            coordinates: Continuous (x, y) coordinates in [0, 1].
            argument: String payload argument (search text, URL, API query).

        Returns:
            StepResult containing observation, reward, and completion status.
        """
        if self.current_state is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if self.is_terminated:
            return StepResult(
                obs_text="Episode already terminated.",
                obs_features=self._generate_obs_features(),
                reward=0.0,
                done=True,
                truncated=False,
                info={"error": "step_called_on_terminated_env"},
            )

        # Normalize action primitive
        if isinstance(action_type, str):
            act_enum = ActionType[action_type.upper()]
        elif isinstance(action_type, int):
            act_enum = ActionType(action_type)
        else:
            act_enum = action_type

        coords = coordinates if coordinates is not None else (0.0, 0.0)
        arg_str = argument if argument is not None else ""

        self.current_state.step_count += 1
        current_step = self.current_state.step_count
        step_reward = -0.05  # Slight step cost to encourage rapid goal achievement
        action_success = False
        hit_element: Optional[UIElement] = None

        # Record action in trajectory history
        action_log = {
            "step": current_step,
            "action_type": act_enum.name,
            "coordinates": coords,
            "argument": arg_str,
        }
        self.action_history.append(action_log)

        # ----------------------------------------------------------------------
        # Action Primitive Transition Handlers
        # ----------------------------------------------------------------------

        # 1. CLICK Transition
        if act_enum == ActionType.CLICK:
            hit_element = self.hit_test(coords[0], coords[1])
            if hit_element is not None:
                action_success = True
                step_reward += 0.10

                # Focus input if element is an input
                if hit_element.is_input:
                    self.current_state.focused_element_id = hit_element.element_id
                    for el in self.current_state.elements:
                        el.is_focused = (el.element_id == hit_element.element_id)

                # Scenario specific click transitions
                domain = self.active_scenario.scenario_id
                stage = self.current_state.metadata.get("stage", "")

                if domain == "ecommerce":
                    if hit_element.role == "submit_btn" and stage == "home":
                        url, title, elements = self._build_ecommerce_page("results")
                        self.current_state.url = url
                        self.current_state.title = title
                        self.current_state.elements = elements
                        self.current_state.metadata["stage"] = "results"
                        step_reward += 0.15

                    elif hit_element.role in ("first_product_link", "item_card") and stage == "results":
                        url, title, elements = self._build_ecommerce_page("product_details")
                        self.current_state.url = url
                        self.current_state.title = title
                        self.current_state.elements = elements
                        self.current_state.metadata["stage"] = "product_details"
                        step_reward += 0.20

                    elif hit_element.role in ("checkout_btn", "proceed_checkout") and stage == "product_details":
                        url, title, elements = self._build_ecommerce_page("cart_confirmed")
                        self.current_state.url = url
                        self.current_state.title = title
                        self.current_state.elements = elements
                        self.current_state.metadata["stage"] = "cart_confirmed"
                        self.current_state.metadata["cart"].append("Sony WH-1000XM5")
                        step_reward += 0.30

                elif domain == "travel":
                    if hit_element.role == "submit_btn" and stage == "home":
                        url, title, elements = self._build_travel_page("results")
                        self.current_state.url = url
                        self.current_state.title = title
                        self.current_state.elements = elements
                        self.current_state.metadata["stage"] = "results"
                        step_reward += 0.20

                    elif hit_element.role == "reserve_room" and stage == "results":
                        url, title, elements = self._build_travel_page("booked")
                        self.current_state.url = url
                        self.current_state.title = title
                        self.current_state.elements = elements
                        self.current_state.metadata["stage"] = "booked"
                        self.current_state.metadata["booked"] = True
                        step_reward += 0.30

                elif domain == "os_admin":
                    if hit_element.role == "nav_security" and stage == "home":
                        url, title, elements = self._build_os_admin_page("security_groups")
                        self.current_state.url = url
                        self.current_state.title = title
                        self.current_state.elements = elements
                        self.current_state.metadata["stage"] = "security_groups"
                        step_reward += 0.20

                    elif hit_element.role == "edit_inbound_rules" and stage == "security_groups":
                        url, title, elements = self._build_os_admin_page("edit_rules")
                        self.current_state.url = url
                        self.current_state.title = title
                        self.current_state.elements = elements
                        self.current_state.metadata["stage"] = "edit_rules"
                        step_reward += 0.20

                    elif hit_element.role == "save_rules" and stage == "edit_rules":
                        url, title, elements = self._build_os_admin_page("rules_saved")
                        self.current_state.url = url
                        self.current_state.title = title
                        self.current_state.elements = elements
                        self.current_state.metadata["stage"] = "rules_saved"
                        step_reward += 0.30
            else:
                # Missed spatial hit
                step_reward -= 0.10

        # 2. SELECT Transition
        elif act_enum == ActionType.SELECT:
            hit_element = self.hit_test(coords[0], coords[1])
            if hit_element is not None and (hit_element.tag == "select" or hit_element.role in ("combobox", "listbox")):
                hit_element.value = arg_str
                action_success = True
                step_reward += 0.15
            else:
                step_reward -= 0.10

        # 3. TYPE Transition
        elif act_enum == ActionType.TYPE:
            hit_element = self.hit_test(coords[0], coords[1])
            target_el = hit_element if (hit_element and hit_element.is_input) else None

            # Fallback to currently focused element
            if target_el is None and self.current_state.focused_element_id is not None:
                for el in self.current_state.elements:
                    if el.element_id == self.current_state.focused_element_id:
                        target_el = el
                        break

            if target_el is not None:
                action_success = True
                target_el.value = arg_str
                step_reward += 0.15

                # Auto-transition for typed queries
                domain = self.active_scenario.scenario_id
                if domain == "ecommerce" and "headphone" in arg_str.lower():
                    self.current_state.metadata["typed_query"] = arg_str
                elif domain == "travel" and "francisco" in arg_str.lower():
                    self.current_state.metadata["destination"] = arg_str
            else:
                step_reward -= 0.10

        # 3. SCROLL Transition
        elif act_enum == ActionType.SCROLL:
            action_success = True
            delta_y = 0.20 if "down" in arg_str.lower() or coords[1] >= 0.5 else -0.20
            self.current_state.scroll_offset_y = max(
                0.0, min(1.0, self.current_state.scroll_offset_y + delta_y)
            )
            step_reward += 0.05

        # 4. NAVIGATE Transition
        elif act_enum == ActionType.NAVIGATE:
            action_success = True
            self.current_state.url = arg_str
            step_reward += 0.10

            # Match URL to pages
            domain = self.active_scenario.scenario_id
            if domain == "os_admin" and "security-groups" in arg_str:
                url, title, elements = self._build_os_admin_page("security_groups")
                self.current_state.url = url
                self.current_state.title = title
                self.current_state.elements = elements
                self.current_state.metadata["stage"] = "security_groups"

        # 5. API_CALL Transition
        elif act_enum == ActionType.API_CALL:
            action_success = True
            step_reward += 0.15
            domain = self.active_scenario.scenario_id
            if domain == "os_admin" and "authorize_ingress" in arg_str:
                self.current_state.metadata["rules"].append(arg_str)
                url, title, elements = self._build_os_admin_page("rules_saved")
                self.current_state.url = url
                self.current_state.title = title
                self.current_state.elements = elements
                self.current_state.metadata["stage"] = "rules_saved"

        # 6. FINISH Transition & Goal Verification
        elif act_enum == ActionType.FINISH:
            action_success = True
            self.is_terminated = True

            # Evaluate goal criteria
            domain = self.active_scenario.scenario_id
            stage = self.current_state.metadata.get("stage", "")

            if domain == "ecommerce":
                if stage == "cart_confirmed" or len(self.current_state.metadata.get("cart", [])) > 0:
                    self.goal_satisfied = True
            elif domain == "travel":
                if stage == "booked" or self.current_state.metadata.get("booked", False):
                    self.goal_satisfied = True
            elif domain == "os_admin":
                if stage == "rules_saved" or len(self.current_state.metadata.get("rules", [])) > 0:
                    self.goal_satisfied = True
            else:
                self.goal_satisfied = True

            if self.goal_satisfied:
                step_reward += 1.0  # Terminal success bonus
            else:
                step_reward -= 0.5  # Premature termination penalty

        # Step limit truncation check
        truncated = False
        if current_step >= self.active_scenario.max_steps and not self.is_terminated:
            self.is_terminated = True
            truncated = True

        self.current_state.last_action_str = f"{act_enum.name} ({coords[0]:.2f}, {coords[1]:.2f}) '{arg_str}'"
        self.current_state.last_action_success = action_success
        self.current_state.cumulative_reward += step_reward

        obs_text = self._format_observation_text()
        obs_features = self._generate_obs_features()

        info = {
            "step": current_step,
            "action": act_enum.name,
            "hit_element": hit_element.text if hit_element else None,
            "action_success": action_success,
            "goal_satisfied": self.goal_satisfied,
            "cumulative_reward": self.current_state.cumulative_reward,
        }

        return StepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=step_reward,
            done=self.is_terminated,
            truncated=truncated,
            info=info,
        )
