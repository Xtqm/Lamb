"""
JED-M2 Headless Interactive Browser Sandbox & Programmatic Task Oracles (Track B).
Implements:
- WebTaskOracle: Declarative deterministic programmatic verification engine evaluating URL patterns,
  DOM selector presence/absence, text substrings, and attribute matches to compute milestone and terminal rewards.
- PlaywrightWebEnvironment: Interactive browser environment matching the WebOSEnvironment interface
  (reset(), step(action_type, coordinates, argument)), wrapping Playwright Chromium at 1280x800 viewport,
  with reproducible local HTML fixtures, normalized coordinate mapping, live accessibility tree extraction,
  and graceful DOM emulation fallback when Playwright is unavailable.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

from dataclasses import dataclass, field
import hashlib
import io
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    from playwright.sync_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        sync_playwright,
    )
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False

from jed_m2.config import JEDM2Config
from jed_m2.environment import (
    EnvironmentState,
    ScenarioDefinition,
    StepResult,
    UIElement,
    get_default_scenarios,
)
from jed_m2.heads import ActionType

logger = logging.getLogger("jed_m2.playwright_env")


# ==============================================================================
# 1. Declarative Programmatic Oracle Verifier Engine
# ==============================================================================

@dataclass
class WebTaskOracle:
    """
    Deterministic programmatic task verifier that inspects live browser DOM state
    to award milestone and terminal outcome rewards.
    """

    scenario_id: str
    url_pattern: Optional[str] = None  # Expected regex or substring in current URL
    required_selectors: List[str] = field(default_factory=list)  # Selectors that must be present
    forbidden_selectors: List[str] = field(default_factory=list)  # Selectors that must be absent
    text_assertions: List[Tuple[str, str]] = field(default_factory=list)  # (selector, expected_substring)
    milestone_selectors: Dict[str, float] = field(default_factory=dict)  # selector -> milestone reward
    terminal_reward: float = 1.0

    def evaluate_page(self, page: Optional[Any], current_url: str, current_dom_text: str = "") -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluates page state against oracle conditions using Playwright DOM queries
        or text fallback.

        Returns:
            Tuple of (reward_delta, goal_satisfied, details_dict).
        """
        reward_delta = 0.0
        details: Dict[str, Any] = {"checks": []}
        all_passed = True

        # 1. URL pattern check
        if self.url_pattern:
            url_match = bool(re.search(self.url_pattern, current_url))
            details["checks"].append({"type": "url_pattern", "passed": url_match, "pattern": self.url_pattern})
            if not url_match:
                all_passed = False

        if page is not None and hasattr(page, "query_selector"):
            # 2. Required DOM selectors
            for sel in self.required_selectors:
                try:
                    el = page.query_selector(sel)
                    passed = (el is not None)
                except Exception:
                    passed = False
                details["checks"].append({"type": "required_selector", "passed": passed, "selector": sel})
                if not passed:
                    all_passed = False

            # 3. Forbidden DOM selectors
            for sel in self.forbidden_selectors:
                try:
                    el = page.query_selector(sel)
                    passed = (el is None)
                except Exception:
                    passed = True
                details["checks"].append({"type": "forbidden_selector", "passed": passed, "selector": sel})
                if not passed:
                    all_passed = False

            # 4. Text content assertions
            for sel, expected_text in self.text_assertions:
                try:
                    el = page.query_selector(sel)
                    if el is not None:
                        actual_text = (el.inner_text() or el.text_content() or "").strip()
                        passed = (expected_text.lower() in actual_text.lower())
                    else:
                        passed = False
                except Exception:
                    passed = False
                details["checks"].append({"type": "text_assertion", "passed": passed, "selector": sel, "expected": expected_text})
                if not passed:
                    all_passed = False

            # 5. Milestone rewards
            for sel, r_val in self.milestone_selectors.items():
                try:
                    el = page.query_selector(sel)
                    if el is not None:
                        reward_delta += r_val
                        details["checks"].append({"type": "milestone", "awarded": r_val, "selector": sel})
                except Exception:
                    pass
        else:
            # Fallback DOM evaluation using text inspection
            for sel in self.required_selectors:
                clean_sel = sel.strip("#.").replace("-", " ")
                passed = (clean_sel.lower() in current_dom_text.lower()) or (sel.lower() in current_dom_text.lower())
                details["checks"].append({"type": "required_selector_fallback", "passed": passed, "selector": sel})
                if not passed:
                    all_passed = False

            for sel, expected_text in self.text_assertions:
                passed = expected_text.lower() in current_dom_text.lower()
                details["checks"].append({"type": "text_assertion_fallback", "passed": passed, "selector": sel, "expected": expected_text})
                if not passed:
                    all_passed = False

            for sel, r_val in self.milestone_selectors.items():
                clean_sel = sel.strip("#.").replace("-", " ")
                if clean_sel.lower() in current_dom_text.lower() or sel.lower() in current_dom_text.lower():
                    reward_delta += r_val

        goal_satisfied = all_passed
        if goal_satisfied:
            reward_delta += self.terminal_reward

        return reward_delta, goal_satisfied, details


def get_default_web_oracles() -> Dict[str, WebTaskOracle]:
    """Returns standard programmatic oracles for benchmark scenarios."""
    return {
        "ecommerce": WebTaskOracle(
            scenario_id="ecommerce",
            required_selectors=["#cart-badge"],
            text_assertions=[("#cart-badge", "1")],
            milestone_selectors={"#cart-status": 0.5},
            terminal_reward=1.0,
        ),
        "form_portal": WebTaskOracle(
            scenario_id="form_portal",
            required_selectors=["#confirm-banner"],
            text_assertions=[("#confirm-banner", "submitted successfully")],
            terminal_reward=1.0,
        ),
        "travel": WebTaskOracle(
            scenario_id="travel",
            required_selectors=["#reservation-badge"],
            text_assertions=[("#reservation-badge", "Confirmed")],
            terminal_reward=1.0,
        ),
        "os_admin": WebTaskOracle(
            scenario_id="os_admin",
            required_selectors=["#confirm-banner", "#reservation-badge"],
            terminal_reward=1.0,
        ),
    }


# ==============================================================================
# 2. Local Fixture Resolver
# ==============================================================================

def get_fixture_path(scenario_id: str) -> Optional[Path]:
    """Resolves local HTML fixture file for reproducible benchmark tasks."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    mapping = {
        "ecommerce": fixtures_dir / "ecommerce.html",
        "form_portal": fixtures_dir / "form_portal.html",
        "travel": fixtures_dir / "travel.html",
        "os_admin": fixtures_dir / "form_portal.html",
    }
    fixture_file = mapping.get(scenario_id)
    if fixture_file and fixture_file.exists():
        return fixture_file
    return None


# ==============================================================================
# 3. Headless Interactive Browser Environment
# ==============================================================================

class PlaywrightWebEnvironment:
    """
    Live browser automation sandbox wrapping Playwright Chromium.
    Implements the exact WebOSEnvironment interface:
    - reset(scenario_id: Optional[str] = None) -> StepResult
    - step(action_type, coordinates, argument) -> StepResult

    Features:
    - Normalized coordinate scaling in [0, 1] mapped to 1280x800 viewport pixels.
    - Native mouse click, keyboard typing, wheel scrolling, and page navigation.
    - Live DOM accessibility tree and element bounding box extraction.
    - Deterministic WebTaskOracle evaluation for milestone and terminal rewards.
    - Resilient graceful fallback when Playwright Chromium is not available.
    """

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        scenario_id: str = "ecommerce",
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        timeout_ms: int = 5000,
        max_steps: int = 15,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config if config is not None else JEDM2Config()
        self.hidden_dim = self.config.hidden_dim
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout_ms = timeout_ms
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.dtype = dtype

        # Scenario and task oracles catalog
        self.scenarios = get_default_scenarios()
        if "form_portal" not in self.scenarios:
            self.scenarios["form_portal"] = ScenarioDefinition(
                scenario_id="form_portal",
                domain="form_flow",
                goal="Fill out employee registration form and submit access request",
                initial_url="file:///fixtures/form_portal.html",
                initial_title="Corporate Admin Portal",
                max_steps=8,
            )

        if scenario_id in self.scenarios:
            self.active_scenario = self.scenarios[scenario_id]
        else:
            self.active_scenario = self.scenarios["ecommerce"]

        self.oracles = get_default_web_oracles()
        self.oracle = self.oracles.get(self.active_scenario.scenario_id, self.oracles["ecommerce"])

        # Playwright handle references
        self._playwright_mgr: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._browser_available = _HAS_PLAYWRIGHT

        # State machine tracking
        self.current_state: Optional[EnvironmentState] = None
        self.action_history: List[Dict[str, Any]] = []
        self.goal_satisfied: bool = False
        self.is_terminated: bool = False
        self._synthetic_dom_fallback = False

    # --------------------------------------------------------------------------
    # Browser Lifecycle Management
    # --------------------------------------------------------------------------

    def _ensure_browser(self) -> Optional[Page]:
        """Initializes Chromium browser instance or falls back to synthetic DOM."""
        if not self._browser_available:
            self._synthetic_dom_fallback = True
            return None

        if self._page is not None and not self._page.is_closed():
            return self._page

        try:
            if self._playwright_mgr is None:
                self._playwright_mgr = sync_playwright().start()

            if self._browser is None:
                self._browser = self._playwright_mgr.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--disable-dev-shm-usage",
                        "--no-sandbox",
                    ],
                )

            if self._context is None:
                self._context = self._browser.new_context(
                    viewport={"width": self.viewport_width, "height": self.viewport_height},
                    device_scale_factor=1.0,
                )
                self._context.set_default_timeout(self.timeout_ms)

            self._page = self._context.new_page()
            return self._page
        except Exception as exc:
            logger.warning("Could not launch Playwright Chromium: %s. Using synthetic DOM fallback.", exc)
            self._synthetic_dom_fallback = True
            return None

    def close(self) -> None:
        """Closes browser handles gracefully."""
        try:
            if self._page is not None and not self._page.is_closed():
                self._page.close()
                self._page = None
        except Exception:
            pass
        try:
            if self._context is not None:
                self._context.close()
                self._context = None
        except Exception:
            pass
        try:
            if self._browser is not None:
                self._browser.close()
                self._browser = None
        except Exception:
            pass
        try:
            if self._playwright_mgr is not None:
                self._playwright_mgr.stop()
                self._playwright_mgr = None
        except Exception:
            pass

    def __enter__(self) -> "PlaywrightWebEnvironment":
        self._ensure_browser()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # --------------------------------------------------------------------------
    # Observation & Element Extraction
    # --------------------------------------------------------------------------

    def extract_interactive_elements(self) -> List[UIElement]:
        """Extracts visible interactive DOM elements with normalized bounding boxes."""
        page = self._ensure_browser()
        if page is None:
            return self._get_fallback_elements()

        js_query = """
        () => {
            const elements = [];
            const selector = 'button, a[href], input, select, textarea, [role="button"], [role="link"], [role="textbox"], [tabindex="0"]';
            const candidates = Array.from(document.querySelectorAll(selector));
            const vw = window.innerWidth || 1280;
            const vh = window.innerHeight || 800;

            let elemId = 1;
            for (const el of candidates) {
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
                    continue;
                }
                const rect = el.getBoundingClientRect();
                if (rect.width <= 0 || rect.height <= 0) continue;
                if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) continue;

                const x_min = Math.max(0.0, Math.min(1.0, rect.left / vw));
                const y_min = Math.max(0.0, Math.min(1.0, rect.top / vh));
                const x_max = Math.max(0.0, Math.min(1.0, rect.right / vw));
                const y_max = Math.max(0.0, Math.min(1.0, rect.bottom / vh));

                const tag = el.tagName.toLowerCase();
                let role = el.getAttribute('role') || tag;
                let text = (el.innerText || el.textContent || el.getAttribute('placeholder') || el.getAttribute('aria-label') || '').trim();
                text = text.substring(0, 45).replace(/\\s+/g, ' ');

                elements.push({
                    element_id: elemId++,
                    tag: tag,
                    role: role,
                    text: text || tag,
                    bbox: [x_min, y_min, x_max, y_max],
                    is_clickable: true,
                    is_input: (tag === 'input' || tag === 'textarea' || role === 'textbox'),
                });
                if (elements.length >= 35) break;
            }
            return elements;
        }
        """
        try:
            raw_elements = page.evaluate(js_query)
            elements: List[UIElement] = []
            for r in raw_elements:
                elements.append(
                    UIElement(
                        element_id=int(r["element_id"]),
                        tag=str(r["tag"]),
                        role=str(r["role"]),
                        text=str(r["text"]),
                        bbox=(float(r["bbox"][0]), float(r["bbox"][1]), float(r["bbox"][2]), float(r["bbox"][3])),
                        is_clickable=bool(r.get("is_clickable", True)),
                        is_input=bool(r.get("is_input", False)),
                    )
                )
            return elements
        except Exception:
            return self._get_fallback_elements()

    def _get_fallback_elements(self) -> List[UIElement]:
        """Provides deterministic fallback UI elements when browser evaluation is unavailable."""
        if self.active_scenario.scenario_id == "ecommerce":
            return [
                UIElement(1, "input", "search_bar", "Search products...", (0.20, 0.08, 0.70, 0.14), is_clickable=True, is_input=True),
                UIElement(2, "button", "search_btn", "Search", (0.72, 0.08, 0.85, 0.14), is_clickable=True),
                UIElement(3, "button", "add_to_cart", "Add to Cart", (0.35, 0.40, 0.50, 0.48), is_clickable=True),
            ]
        elif self.active_scenario.scenario_id == "form_portal":
            return [
                UIElement(1, "input", "name_input", "Full Name", (0.20, 0.15, 0.60, 0.22), is_clickable=True, is_input=True),
                UIElement(2, "input", "email_input", "Email Address", (0.20, 0.25, 0.60, 0.32), is_clickable=True, is_input=True),
                UIElement(3, "button", "submit_btn", "Submit Request", (0.20, 0.45, 0.40, 0.52), is_clickable=True),
            ]
        else:
            return [
                UIElement(1, "input", "destination_input", "Enter City", (0.20, 0.10, 0.60, 0.18), is_clickable=True, is_input=True),
                UIElement(2, "button", "reserve_btn", "Reserve Room", (0.65, 0.30, 0.85, 0.38), is_clickable=True),
            ]

    def _format_observation_text(self) -> str:
        if self.current_state is None:
            return "[Page: None]"
        parts = [
            f"[URL] {self.current_state.url}",
            f"[Title] {self.current_state.title}",
            f"[Goal] {self.active_scenario.goal}",
            "[Interactive Elements]",
        ]
        for el in self.current_state.elements[:20]:
            parts.append(el.to_dom_string())
        return "\n".join(parts)

    def _generate_obs_features(self) -> torch.Tensor:
        """Deterministic state embedding [hidden_dim] for JEPA world model conditioning."""
        key = f"{self.active_scenario.scenario_id}_{self.current_state.url if self.current_state else ''}_{self.current_state.step_count if self.current_state else 0}"
        h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)
        gen = torch.Generator().manual_seed(h)
        feat = torch.randn(self.hidden_dim, generator=gen).to(device=self.device, dtype=self.dtype)
        return feat

    # --------------------------------------------------------------------------
    # Environment Reset & Step
    # --------------------------------------------------------------------------

    def reset(
        self,
        scenario_id: Optional[str] = None,
        custom_goal: Optional[str] = None,
    ) -> StepResult:
        """
        Resets environment and navigates to local fixture or target scenario.
        """
        if scenario_id is not None and scenario_id in self.scenarios:
            self.active_scenario = self.scenarios[scenario_id]
        if custom_goal is not None:
            self.active_scenario.goal = custom_goal

        self.oracle = self.oracles.get(self.active_scenario.scenario_id, self.oracles["ecommerce"])
        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False

        page = self._ensure_browser()
        url = self.active_scenario.initial_url
        title = self.active_scenario.initial_title

        # Check for local HTML fixture
        fixture_path = get_fixture_path(self.active_scenario.scenario_id)
        if page is not None and fixture_path is not None:
            try:
                # Load local HTML fixture directly
                with open(fixture_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                page.set_content(html_content, wait_until="domcontentloaded")
                url = f"file:///{fixture_path.resolve().as_posix()}"
                title = page.title() or title
            except Exception as exc:
                logger.warning("Fixture load error: %s", exc)

        elements = self.extract_interactive_elements()
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
            metadata={"scenario": self.active_scenario.scenario_id},
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
                "url": url,
                "scenario": self.active_scenario.scenario_id,
                "goal": self.active_scenario.goal,
            },
        )

    def step(
        self,
        action_type: Union[ActionType, int, str],
        coordinates: Optional[Tuple[float, float]] = None,
        argument: Optional[str] = None,
    ) -> StepResult:
        """
        Dispatches action primitive into Chromium browser and evaluates programmatic oracle.
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

        coords = coordinates if coordinates is not None else (0.5, 0.5)
        arg_str = argument if argument is not None else ""

        self.current_state.step_count += 1
        current_step = self.current_state.step_count
        step_reward = -0.05  # Step time-decay penalty
        action_success = True
        page = self._ensure_browser()

        # Spatial hit testing against visible UI elements
        hit_element: Optional[UIElement] = None
        for el in self.current_state.elements:
            if el.contains(coords[0], coords[1]):
                hit_element = el
                break

        # ----------------------------------------------------------------------
        # Native Browser Primitive Dispatch
        # ----------------------------------------------------------------------
        if page is not None and not self._synthetic_dom_fallback:
            try:
                pixel_x = float(coords[0] * self.viewport_width)
                pixel_y = float(coords[1] * self.viewport_height)

                if act_enum == ActionType.CLICK:
                    page.mouse.click(pixel_x, pixel_y)
                    if hit_element is not None:
                        step_reward += 0.15  # Reward hitting interactable element
                elif act_enum == ActionType.TYPE:
                    page.keyboard.type(arg_str or "test", delay=20)
                    step_reward += 0.10
                elif act_enum == ActionType.SCROLL:
                    delta_y = 300.0 if coords[1] >= 0.5 else -300.0
                    page.mouse.wheel(0, delta_y)
                elif act_enum == ActionType.NAVIGATE:
                    if arg_str.startswith("http") or arg_str.startswith("file:"):
                        page.goto(arg_str, wait_until="domcontentloaded", timeout=self.timeout_ms)
                elif act_enum == ActionType.FINISH:
                    pass

                # Allow DOM animations/mutations to settle
                page.wait_for_timeout(100)
            except Exception as exc:
                action_success = False
                logger.warning("Browser action execution warning: %s", exc)
        else:
            # Emulated action execution in fallback mode
            if act_enum == ActionType.CLICK and hit_element is not None:
                step_reward += 0.15
            elif act_enum == ActionType.TYPE:
                step_reward += 0.10

        # ----------------------------------------------------------------------
        # Evaluate Programmatic Task Oracle
        # ----------------------------------------------------------------------
        dom_text = self._format_observation_text()
        oracle_r, goal_sat, oracle_details = self.oracle.evaluate_page(
            page=page,
            current_url=self.current_state.url,
            current_dom_text=dom_text,
        )

        step_reward += oracle_r
        if goal_sat or act_enum == ActionType.FINISH:
            self.goal_satisfied = goal_sat
            self.is_terminated = True

        # Check step horizon truncation
        truncated = current_step >= self.max_steps
        if truncated:
            self.is_terminated = True

        # Refresh state
        elements = self.extract_interactive_elements()
        self.current_state.elements = elements
        self.current_state.cumulative_reward += step_reward
        self.current_state.last_action_str = f"{act_enum.name}({coords[0]:.2f}, {coords[1]:.2f}, '{arg_str}')"
        self.current_state.last_action_success = action_success

        self.action_history.append({
            "step": current_step,
            "action": act_enum.name,
            "coords": coords,
            "arg": arg_str,
            "success": action_success,
            "oracle_checks": oracle_details,
        })

        obs_text = self._format_observation_text()
        obs_features = self._generate_obs_features()

        return StepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=step_reward,
            done=self.is_terminated,
            truncated=truncated,
            info={
                "step": current_step,
                "url": self.current_state.url,
                "goal_satisfied": self.goal_satisfied,
                "action_success": action_success,
                "cumulative_reward": self.current_state.cumulative_reward,
                "oracle_details": oracle_details,
            },
        )
