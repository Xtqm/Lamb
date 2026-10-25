"""
JED-M2 Unified Live Browser Automation Environment & Task Oracles.
Provides:
- WebTaskOracle: Declarative deterministic programmatic verification engine.
- get_default_web_oracles: Catalog of default scenario verification oracles.
- get_fixture_path: Path resolver for reproducible local HTML benchmark fixtures.
- PlaywrightStepResult: Unified StepResult carrying viewport screenshot tensors, accessibility tree, and oracle details.
- PlaywrightEnvironment: Canonical live browser automation sandbox wrapping Playwright Chromium
  with full backward-compatible aliases:
    PlaywrightBrowserEnvironment = PlaywrightEnvironment
    PlaywrightWebEnvironment = PlaywrightEnvironment

STRICT BUILD RULE: Pure PyTorch without compiled C++/CUDA extensions.
STRICT CONVENTION: Pure ASCII console output (no multi-byte Unicode box-drawing characters).
"""

from dataclasses import dataclass, field
import hashlib
import io
import logging
import math
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
from jed_m2.core.types import UIElement
from jed_m2.environment import (
    EnvironmentState,
    ScenarioDefinition,
    StepResult,
    get_default_scenarios,
)
from jed_m2.heads import ActionType

logger = logging.getLogger("jed_m2.browser")


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
    target_bounding_boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)  # (x1, y1, x2, y2)
    terminal_reward: float = 1.0

    def compute_spatial_reward(self, coords: Tuple[float, float]) -> float:
        """
        Computes dense spatial IoU reward for CLICK coordinates against known target elements.
        Returns continuous reward in [0.0, 0.5] if inside or close to target bounding box.
        """
        if not self.target_bounding_boxes:
            return 0.0
        cx, cy = coords
        best_reward = 0.0
        for (x1, y1, x2, y2) in self.target_bounding_boxes:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return 0.5
            box_cx = (x1 + x2) / 2.0
            box_cy = (y1 + y2) / 2.0
            dist = math.sqrt((cx - box_cx) ** 2 + (cy - box_cy) ** 2)
            if dist < 0.15:
                proximity_reward = 0.5 * (1.0 - dist / 0.15)
                best_reward = max(best_reward, proximity_reward)
        return round(best_reward, 3)

    def evaluate_page(
        self,
        page: Optional[Any],
        current_url: str,
        current_dom_text: str = "",
    ) -> Tuple[float, bool, Dict[str, Any]]:
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
                details["checks"].append({
                    "type": "text_assertion",
                    "passed": passed,
                    "selector": sel,
                    "expected": expected_text,
                })
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
                details["checks"].append({
                    "type": "text_assertion_fallback",
                    "passed": passed,
                    "selector": sel,
                    "expected": expected_text,
                })
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
            target_bounding_boxes=[(0.35, 0.40, 0.50, 0.48), (0.72, 0.08, 0.85, 0.14)],
        ),
        "form_portal": WebTaskOracle(
            scenario_id="form_portal",
            required_selectors=["#confirm-banner"],
            text_assertions=[("#confirm-banner", "submitted successfully")],
            terminal_reward=1.0,
            target_bounding_boxes=[(0.20, 0.45, 0.40, 0.52), (0.20, 0.15, 0.60, 0.22)],
        ),
        "travel": WebTaskOracle(
            scenario_id="travel",
            required_selectors=["#reservation-badge"],
            text_assertions=[("#reservation-badge", "Confirmed")],
            terminal_reward=1.0,
            target_bounding_boxes=[(0.65, 0.30, 0.85, 0.38), (0.20, 0.10, 0.60, 0.18)],
        ),
        "os_admin": WebTaskOracle(
            scenario_id="os_admin",
            required_selectors=["#confirm-banner", "#reservation-badge"],
            terminal_reward=1.0,
            target_bounding_boxes=[(0.20, 0.45, 0.40, 0.52)],
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
# 3. Extended Step Result with Viewport Screenshot & Oracle Details
# ==============================================================================

@dataclass
class PlaywrightStepResult(StepResult):
    """
    Extended StepResult carrying live viewport screenshot tensor [3, H, W],
    raw accessibility data, and programmatic task oracle verification details.
    """

    screenshot: Optional[torch.Tensor] = None
    raw_accessibility_tree: Optional[Union[Dict[str, Any], str]] = None
    oracle_details: Optional[Dict[str, Any]] = None


# ==============================================================================
# 4. Canonical Playwright Environment
# ==============================================================================

class PlaywrightEnvironment:
    """
    Canonical live browser automation sandbox wrapping Playwright Chromium.
    Navigates real websites or local HTML benchmark fixtures, extracts live DOM
    accessibility trees, computes normalized UI element bounding boxes, captures
    real viewport screenshot tensors, evaluates programmatic task oracles, and
    dispatches native browser interaction primitives.
    """

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        scenario_id: Optional[str] = None,
        oracle: Optional[WebTaskOracle] = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        screenshot_size: int = 224,
        timeout_ms: int = 10000,
        max_steps: int = 15,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        mock_mode: bool = False,
        fixture_path: Optional[Union[str, Path]] = None,
        local_benchmark_url: Optional[str] = None,
    ):
        self.config = config if config is not None else JEDM2Config()
        self.hidden_dim = self.config.hidden_dim
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.screenshot_size = screenshot_size
        self.timeout_ms = timeout_ms
        self.max_steps = max_steps
        self.device = torch.device(device)
        self.dtype = dtype
        self.mock_mode = mock_mode or (os.getenv("JED_M2_MOCK_BROWSER", "0") == "1")
        self.fixture_path = Path(fixture_path) if fixture_path else None
        self.local_benchmark_url = local_benchmark_url

        # Scenarios catalog
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

        active_id = scenario_id or "ecommerce"
        if active_id in self.scenarios:
            self.active_scenario = self.scenarios[active_id]
        else:
            self.active_scenario = ScenarioDefinition(
                scenario_id=active_id,
                domain="web",
                goal="Explore web application and complete browsing tasks",
                initial_url=local_benchmark_url or "about:blank",
                initial_title="Browser Page",
                max_steps=max_steps,
            )

        # Oracles catalog
        self.oracles = get_default_web_oracles()
        if oracle is not None:
            self.oracle = oracle
        elif active_id in self.oracles:
            self.oracle = self.oracles[active_id]
        else:
            self.oracle = self.oracles["ecommerce"]

        # Playwright handle references
        self._playwright_mgr: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._browser_available = _HAS_PLAYWRIGHT and not self.mock_mode

        # Episode state tracking
        self.current_state: Optional[EnvironmentState] = None
        self.action_history: List[Dict[str, Any]] = []
        self.goal_satisfied: bool = False
        self.is_terminated: bool = False
        self.current_goal: str = self.active_scenario.goal
        self._synthetic_dom_fallback: bool = self.mock_mode or not self._browser_available

    # --------------------------------------------------------------------------
    # Browser Lifecycle Management
    # --------------------------------------------------------------------------

    def _ensure_browser(self) -> Optional[Page]:
        """
        Lazily initializes the Chromium browser, browser context, and main page.
        Falls back safely to synthetic DOM emulation if Playwright is unavailable or fails.
        """
        if self._synthetic_dom_fallback or self.mock_mode or not self._browser_available:
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
        """Gracefully closes page, browser context, and Playwright session."""
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

    def __enter__(self) -> "PlaywrightEnvironment":
        self._ensure_browser()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    # --------------------------------------------------------------------------
    # Observation Extraction & DOM Parsing
    # --------------------------------------------------------------------------

    def get_accessibility_snapshot(self) -> Optional[Union[Dict[str, Any], str]]:
        """
        Retrieves the Chromium accessibility tree snapshot for the current page.
        Returns synthetic mock structure if page is unavailable.
        """
        if self._synthetic_dom_fallback:
            return {
                "role": "WebArea",
                "name": self.current_state.title if self.current_state else "Mock Page",
                "children": [
                    {"role": el.role, "name": el.text, "element_id": el.element_id}
                    for el in (self.current_state.elements if self.current_state else [])
                ],
            }

        try:
            page = self._ensure_browser()
            if page is None:
                return None
            if hasattr(page, "aria_snapshot"):
                return page.aria_snapshot()
            elif hasattr(page, "accessibility") and hasattr(page.accessibility, "snapshot"):
                return page.accessibility.snapshot()
            return None
        except Exception:
            return None

    def extract_interactive_elements(self) -> List[UIElement]:
        """
        Scans DOM for visible interactive elements (buttons, inputs, links, selects, textareas).
        Computes client bounding boxes and normalizes them to [0, 1] relative to the viewport.
        """
        page = self._ensure_browser()
        if page is None or self._synthetic_dom_fallback:
            return self._get_fallback_elements()

        js_query = """
        () => {
            const elements = [];
            const selector = 'button, a[href], input, select, textarea, [role="button"], [role="link"], [role="textbox"], [role="checkbox"], [role="radio"], [role="menuitem"], [tabindex="0"]';
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
                if (rect.width <= 0 || rect.height <= 0) {
                    continue;
                }
                
                if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) {
                    continue;
                }

                const x_min = Math.max(0.0, Math.min(1.0, rect.left / vw));
                const y_min = Math.max(0.0, Math.min(1.0, rect.top / vh));
                const x_max = Math.max(0.0, Math.min(1.0, rect.right / vw));
                const y_max = Math.max(0.0, Math.min(1.0, rect.bottom / vh));

                const tag = el.tagName.toLowerCase();
                let role = el.getAttribute('role') || '';
                if (!role) {
                    if (tag === 'button') role = 'button';
                    else if (tag === 'a') role = 'link';
                    else if (tag === 'input') {
                        const type = (el.getAttribute('type') || 'text').toLowerCase();
                        role = (type === 'submit' || type === 'button') ? 'button' : 'input';
                    }
                    else if (tag === 'textarea') role = 'input';
                    else if (tag === 'select') role = 'dropdown';
                    else role = 'element';
                }

                let text = (el.innerText || el.textContent || '').trim();
                if (!text && el.getAttribute('aria-label')) {
                    text = el.getAttribute('aria-label').trim();
                }
                if (!text && el.getAttribute('placeholder')) {
                    text = el.getAttribute('placeholder').trim();
                }
                if (!text && el.getAttribute('title')) {
                    text = el.getAttribute('title').trim();
                }
                if (!text && el.getAttribute('value')) {
                    text = el.getAttribute('value').trim();
                }
                if (!text) {
                    text = role || tag;
                }
                text = text.substring(0, 50).replace(/\\s+/g, ' ');

                const is_input = (tag === 'input' || tag === 'textarea' || role === 'textbox' || role === 'input');
                const is_clickable = true;
                const value = (el.value || '').substring(0, 50);

                elements.push({
                    element_id: elemId++,
                    tag: tag,
                    role: role,
                    text: text,
                    bbox: [x_min, y_min, x_max, y_max],
                    value: value,
                    is_clickable: is_clickable,
                    is_input: is_input,
                });

                if (elements.length >= 40) {
                    break;
                }
            }
            return elements;
        }
        """

        try:
            raw_elements = page.evaluate(js_query)
        except Exception:
            raw_elements = []

        if not raw_elements:
            return self._get_fallback_elements()

        ui_elements: List[UIElement] = []
        for raw in raw_elements:
            bbox = (
                float(raw["bbox"][0]),
                float(raw["bbox"][1]),
                float(raw["bbox"][2]),
                float(raw["bbox"][3]),
            )
            ui_elements.append(
                UIElement(
                    element_id=int(raw["element_id"]),
                    tag=str(raw["tag"]),
                    role=str(raw["role"]),
                    text=str(raw["text"]),
                    bbox=bbox,
                    value=str(raw.get("value", "")),
                    is_clickable=bool(raw.get("is_clickable", True)),
                    is_input=bool(raw.get("is_input", False)),
                )
            )

        return ui_elements

    def _get_fallback_elements(self) -> List[UIElement]:
        """Provides deterministic fallback UI elements when browser evaluation is unavailable."""
        scenario_id = (
            getattr(self.active_scenario, "scenario_id", "ecommerce")
            if hasattr(self, "active_scenario") and self.active_scenario
            else "ecommerce"
        )
        if scenario_id == "ecommerce":
            return [
                UIElement(1, "input", "search_bar", "Search products...", (0.20, 0.08, 0.70, 0.14), is_clickable=True, is_input=True),
                UIElement(2, "button", "search_btn", "Search", (0.72, 0.08, 0.85, 0.14), is_clickable=True),
                UIElement(3, "button", "add_to_cart", "Add to Cart", (0.35, 0.40, 0.50, 0.48), is_clickable=True),
            ]
        elif scenario_id == "form_portal":
            return [
                UIElement(1, "input", "name_input", "Full Name", (0.20, 0.15, 0.60, 0.22), is_clickable=True, is_input=True),
                UIElement(2, "input", "email_input", "Email Address", (0.20, 0.25, 0.60, 0.32), is_clickable=True, is_input=True),
                UIElement(3, "button", "submit_btn", "Submit Request", (0.20, 0.45, 0.40, 0.52), is_clickable=True),
            ]
        elif scenario_id == "travel":
            return [
                UIElement(1, "input", "destination_input", "Enter City", (0.20, 0.10, 0.60, 0.18), is_clickable=True, is_input=True),
                UIElement(2, "button", "reserve_btn", "Reserve Room", (0.65, 0.30, 0.85, 0.38), is_clickable=True),
            ]
        else:
            return [
                UIElement(1, "button", "btn_1", "Interactive Button 1", (0.1, 0.1, 0.3, 0.2), is_clickable=True),
                UIElement(2, "input", "inp_1", "Search or input", (0.4, 0.1, 0.7, 0.2), is_clickable=True, is_input=True),
                UIElement(3, "a", "link_1", "Home Link", (0.8, 0.1, 0.95, 0.2), is_clickable=True),
            ]

    def capture_screenshot(
        self,
        as_tensor: bool = True,
    ) -> Optional[Union[torch.Tensor, Any]]:
        """
        Captures full viewport screenshot as normalized [3, H, W] float tensor
        or PIL Image.
        """
        if self._synthetic_dom_fallback:
            if as_tensor:
                return torch.zeros(
                    (3, self.screenshot_size, self.screenshot_size),
                    device=self.device,
                    dtype=torch.float32,
                )
            if _HAS_PIL:
                return Image.new("RGB", (self.screenshot_size, self.screenshot_size), color=(240, 240, 240))
            return None

        page = self._ensure_browser()
        if page is None:
            if as_tensor:
                return torch.zeros(
                    (3, self.screenshot_size, self.screenshot_size),
                    device=self.device,
                    dtype=torch.float32,
                )
            return None

        try:
            png_bytes = page.screenshot(type="png", full_page=False)
        except Exception:
            if as_tensor:
                return torch.zeros(
                    (3, self.screenshot_size, self.screenshot_size),
                    device=self.device,
                    dtype=torch.float32,
                )
            return None

        if not _HAS_PIL:
            if as_tensor:
                return torch.zeros(
                    (3, self.screenshot_size, self.screenshot_size),
                    device=self.device,
                    dtype=torch.float32,
                )
            return None

        try:
            image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            if image.size != (self.screenshot_size, self.screenshot_size):
                image = image.resize(
                    (self.screenshot_size, self.screenshot_size),
                    resample=Image.BILINEAR,
                )

            if not as_tensor:
                return image

            import numpy as np
            img_np = np.array(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            return tensor.to(device=self.device, dtype=torch.float32)
        except Exception:
            if as_tensor:
                return torch.zeros(
                    (3, self.screenshot_size, self.screenshot_size),
                    device=self.device,
                    dtype=torch.float32,
                )
            return None

    def _format_observation_text(self) -> str:
        """Formats concise text observation matching pretraining grammar: URL: ... | DOM: <button ...>."""
        if self.current_state is None:
            return "[Page: None]"

        url = self.current_state.url
        elements = self.current_state.elements

        if elements:
            dom_str = " ".join([el.to_dom_string() for el in elements[:64]])
            return f"URL: {url} | DOM: {dom_str}"
        return f"URL: {url} | DOM: [No interactive elements]"

    def _generate_obs_features(self) -> torch.Tensor:
        """Generates deterministic dense representation vector [hidden_dim] for JEPA world model."""
        if self.current_state is None:
            return torch.zeros(self.hidden_dim, device=self.device, dtype=self.dtype)

        obs_str = f"{self.current_state.url}::{self.current_state.title}::{len(self.current_state.elements)}"
        hash_digest = hashlib.sha256(obs_str.encode("utf-8")).digest()
        seed_ints = [hash_digest[i % len(hash_digest)] for i in range(self.hidden_dim)]
        raw_feat = torch.tensor(seed_ints, device=self.device, dtype=torch.float32)
        mean = raw_feat.mean()
        std = raw_feat.std().clamp(min=1e-5)
        norm_feat = (raw_feat - mean) / std
        return norm_feat.to(dtype=self.dtype)

    # --------------------------------------------------------------------------
    # Environment Step & Reset Lifecycle
    # --------------------------------------------------------------------------

    def reset(
        self,
        url: Optional[str] = None,
        goal: Optional[str] = None,
        scenario_id: Optional[str] = None,
        custom_goal: Optional[str] = None,
        **kwargs: Any,
    ) -> PlaywrightStepResult:
        """
        Resets environment and navigates to target URL or local benchmark fixture.
        Supports both reset(url, goal) and reset(scenario_id, custom_goal) calling styles.
        """
        raw_url = url or kwargs.get("url") or kwargs.get("initial_url")
        target_scenario_id = scenario_id or kwargs.get("scenario_id")
        target_url = None

        if raw_url is not None:
            if raw_url in self.scenarios or raw_url in self.oracles:
                target_scenario_id = raw_url
            else:
                target_url = raw_url

        if target_scenario_id is not None:
            if target_scenario_id in self.scenarios:
                self.active_scenario = self.scenarios[target_scenario_id]
            else:
                self.active_scenario.scenario_id = target_scenario_id
            self.oracle = self.oracles.get(target_scenario_id, self.oracles.get("ecommerce"))

        effective_goal = custom_goal or goal or (self.active_scenario.goal if self.active_scenario else None)
        if effective_goal is not None:
            self.current_goal = effective_goal
            if self.active_scenario:
                self.active_scenario.goal = effective_goal

        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False

        page = self._ensure_browser()

        # Resolve starting URL or local fixture
        url = target_url or (self.active_scenario.initial_url if self.active_scenario else "about:blank")
        title = self.active_scenario.initial_title if self.active_scenario else "Browser Page"
        nav_error: Optional[str] = None

        fixture_file = None
        if target_url is None:
            fixture_file = self.fixture_path or (
                get_fixture_path(self.active_scenario.scenario_id) if (self.active_scenario and target_scenario_id is not None) else None
            )

        if page is not None and not self._synthetic_dom_fallback:
            if target_url and target_url != "about:blank":
                try:
                    page.goto(target_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    url = page.url or target_url
                    title = page.title() or title
                except Exception as exc:
                    nav_error = str(exc)
            elif fixture_file and fixture_file.exists():
                try:
                    with open(fixture_file, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    page.set_content(html_content, wait_until="domcontentloaded")
                    url = f"file:///{fixture_file.resolve().as_posix()}"
                    title = page.title() or title
                except Exception as exc:
                    nav_error = str(exc)
            elif url and url != "about:blank":
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                    url = page.url or url
                    title = page.title() or title
                except Exception as exc:
                    nav_error = str(exc)
        else:
            if target_url:
                url = target_url
            elif fixture_file and fixture_file.exists():
                url = f"file:///{fixture_file.resolve().as_posix()}"

        elements = self.extract_interactive_elements()
        screenshot = self.capture_screenshot(as_tensor=True)
        raw_tree = self.get_accessibility_snapshot()

        meta = {
            "stage": "reset",
            "navigation_error": nav_error,
            "scenario": self.active_scenario.scenario_id if self.active_scenario else "default",
        }

        self.current_state = EnvironmentState(
            url=url,
            title=title,
            scroll_offset_y=0.0,
            elements=elements,
            focused_element_id=None,
            step_count=0,
            cumulative_reward=0.0,
            last_action_str="RESET",
            last_action_success=(nav_error is None),
            metadata=meta,
        )

        obs_text = self._format_observation_text()
        if nav_error:
            obs_text = f"[Navigation Error: {nav_error}]\n{obs_text}"
        obs_features = self._generate_obs_features()

        info = {
            "step": 0,
            "url": url,
            "scenario": self.active_scenario.scenario_id if self.active_scenario else "default",
            "goal": self.current_goal,
            "screenshot": screenshot,
            "elements": elements,
            "navigation_error": nav_error,
        }

        return PlaywrightStepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=0.0,
            done=False,
            truncated=False,
            screenshot=screenshot,
            raw_accessibility_tree=raw_tree,
            oracle_details=None,
            info=info,
        )

    def step(
        self,
        action_type: Union[ActionType, int, str],
        coordinates: Optional[Tuple[float, float]] = None,
        argument: Optional[str] = None,
    ) -> PlaywrightStepResult:
        """
        Dispatches native browser action primitive and evaluates programmatic task oracle.
        """
        if self.current_state is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if self.is_terminated:
            screenshot = self.capture_screenshot(as_tensor=True)
            return PlaywrightStepResult(
                obs_text="Episode already terminated.",
                obs_features=self._generate_obs_features(),
                reward=0.0,
                done=True,
                truncated=False,
                screenshot=screenshot,
                raw_accessibility_tree=None,
                oracle_details=None,
                info={"error": "step_called_on_terminated_env", "screenshot": screenshot, "step": self.current_state.step_count},
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
        step_reward = -0.05
        action_success = False
        action_error: Optional[str] = None
        hit_element: Optional[UIElement] = None

        page = self._ensure_browser()

        # Spatial hit testing
        cx, cy = coords
        for el in self.current_state.elements:
            if el.contains(cx, cy):
                hit_element = el
                break

        # Action dispatch
        if page is not None and not self._synthetic_dom_fallback:
            try:
                pixel_x = float(coords[0] * self.viewport_width)
                pixel_y = float(coords[1] * self.viewport_height)

                if act_enum == ActionType.CLICK:
                    page.mouse.click(pixel_x, pixel_y)
                    action_success = True
                    if hit_element is not None:
                        step_reward += 0.15
                elif act_enum == ActionType.TYPE:
                    text_to_type = arg_str if arg_str else "test"
                    page.keyboard.type(text_to_type, delay=20)
                    action_success = True
                    step_reward += 0.10
                elif act_enum == ActionType.SCROLL:
                    delta_y = 300.0 if coords[1] >= 0.5 else -300.0
                    page.mouse.wheel(0, delta_y)
                    action_success = True
                elif act_enum == ActionType.NAVIGATE:
                    target_url = arg_str.strip()
                    if target_url:
                        page.goto(target_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                        action_success = True
                    else:
                        action_error = "Empty URL provided for NAVIGATE action"
                elif act_enum == ActionType.FINISH:
                    action_success = True
                else:
                    action_error = f"Unsupported browser action primitive: {act_enum.name}"

                try:
                    page.wait_for_timeout(100)
                except Exception:
                    pass
            except Exception as exc:
                action_success = False
                action_error = str(exc)
        else:
            # Emulated mock execution
            action_success = True
            if act_enum == ActionType.CLICK and hit_element is not None:
                step_reward += 0.15
            elif act_enum == ActionType.TYPE:
                step_reward += 0.10

        # Programmatic Oracle Evaluation
        oracle_details: Optional[Dict[str, Any]] = None
        if self.oracle is not None:
            # Dense spatial IoU reward shaping if CLICK action
            if act_enum == ActionType.CLICK and hasattr(self.oracle, "compute_spatial_reward"):
                spatial_r = self.oracle.compute_spatial_reward(coords)
                step_reward += spatial_r

            dom_text = self._format_observation_text()
            oracle_r, goal_sat, oracle_details = self.oracle.evaluate_page(
                page=page,
                current_url=self.current_state.url,
                current_dom_text=dom_text,
            )
            step_reward += oracle_r
            if goal_sat:
                self.goal_satisfied = True
                self.is_terminated = True

        if act_enum == ActionType.FINISH:
            self.is_terminated = True
            if not self.goal_satisfied:
                self.goal_satisfied = True
            step_reward += 1.0

        truncated = current_step >= self.max_steps
        if truncated:
            self.is_terminated = True

        # Refresh page state
        try:
            curr_url = (page.url if page else None) or self.current_state.url
            curr_title = (page.title() if page else None) or self.current_state.title
        except Exception:
            curr_url = self.current_state.url
            curr_title = self.current_state.title

        elements = self.extract_interactive_elements()
        screenshot = self.capture_screenshot(as_tensor=True)
        raw_tree = self.get_accessibility_snapshot()

        self.current_state.url = curr_url
        self.current_state.title = curr_title
        self.current_state.elements = elements
        self.current_state.cumulative_reward += step_reward
        self.current_state.last_action_str = f"{act_enum.name}({coords[0]:.2f}, {coords[1]:.2f}, '{arg_str}')"
        self.current_state.last_action_success = action_success

        action_log = {
            "step": current_step,
            "action_type": act_enum.name,
            "coordinates": coords,
            "argument": arg_str,
            "success": action_success,
            "hit_element": hit_element.text if hit_element else None,
            "error": action_error,
            "oracle_checks": oracle_details,
        }
        self.action_history.append(action_log)

        obs_text = self._format_observation_text()
        if action_error:
            obs_text = f"[Action Error: {action_error}]\n{obs_text}"
        obs_features = self._generate_obs_features()

        info = {
            "step": current_step,
            "url": curr_url,
            "goal": self.current_goal,
            "action_type": act_enum.name,
            "action_success": action_success,
            "hit_element": hit_element.text if hit_element else None,
            "action_error": action_error,
            "goal_satisfied": self.goal_satisfied,
            "cumulative_reward": self.current_state.cumulative_reward,
            "screenshot": screenshot,
            "elements": elements,
            "oracle_details": oracle_details,
            "scenario": self.active_scenario.scenario_id if self.active_scenario else None,
        }

        return PlaywrightStepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=step_reward,
            done=self.is_terminated,
            truncated=truncated,
            screenshot=screenshot,
            raw_accessibility_tree=raw_tree,
            oracle_details=oracle_details,
            info=info,
        )


# ==============================================================================
# 5. Backward-Compatible Aliases
# ==============================================================================

PlaywrightBrowserEnvironment = PlaywrightEnvironment
PlaywrightWebEnvironment = PlaywrightEnvironment
