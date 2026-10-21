"""
JED-M2 Live Browser Automation Environment (Phase 7).
Provides:
- PlaywrightStepResult: Subclass of StepResult containing live viewport screenshot tensor.
- PlaywrightBrowserEnvironment: Live browser automation sandbox wrapping Playwright Chromium:
  - Manages headless/headed Chromium browser instance and persistent page contexts.
  - Extracts live DOM accessibility snapshots, interactive element candidates, and normalized bounding boxes in [0, 1] x [0, 1].
  - Captures real viewport screenshots as [3, H, W] normalized float RGB tensors or PIL images for multimodal visual grounding.
  - Dispatches native browser interaction primitives:
    * CLICK at normalized (x, y) coordinates via page.mouse.click.
    * TYPE at active/target element via page.keyboard.type.
    * SCROLL via page.mouse.wheel.
    * NAVIGATE via page.goto.
    * FINISH to complete episode.
  - Resilient error handling for network timeouts, unresolvable domains, and navigation failures.

STRICT BUILD RULE: Pure PyTorch without compiled C++/CUDA extensions.
"""

from dataclasses import dataclass, field
import hashlib
import io
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
from jed_m2.environment import EnvironmentState, StepResult, UIElement
from jed_m2.heads import ActionType


# ==============================================================================
# 1. Extended Step Result with Viewport Screenshot
# ==============================================================================

@dataclass
class PlaywrightStepResult(StepResult):
    """
    Extended StepResult carrying live viewport screenshot tensor [3, H, W]
    and raw accessibility data.
    """
    screenshot: Optional[torch.Tensor] = None
    raw_accessibility_tree: Optional[Union[Dict[str, Any], str]] = None


# ==============================================================================
# 2. Live Playwright Browser Environment
# ==============================================================================

class PlaywrightBrowserEnvironment:
    """
    Live browser automation environment wrapping Playwright Chromium.
    Navigates real web pages, extracts live DOM accessibility trees,
    computes normalized UI element bounding boxes, captures screenshot tensors,
    and dispatches native browser action primitives.
    """

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        headless: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 800,
        screenshot_size: int = 224,
        timeout_ms: int = 10000,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        if not _HAS_PLAYWRIGHT:
            raise ImportError(
                "Playwright is not installed. Please install it via:\n"
                "  pip install playwright && playwright install chromium"
            )

        self.config = config if config is not None else JEDM2Config()
        self.hidden_dim = self.config.hidden_dim
        self.headless = headless
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.screenshot_size = screenshot_size
        self.timeout_ms = timeout_ms
        self.device = torch.device(device)
        self.dtype = dtype

        # Playwright handle references
        self._playwright_mgr: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        # Episode state tracking
        self.current_state: Optional[EnvironmentState] = None
        self.action_history: List[Dict[str, Any]] = []
        self.goal_satisfied: bool = False
        self.is_terminated: bool = False
        self.current_goal: str = "Explore web application and complete browsing tasks"
        self.max_steps: int = 15

    # --------------------------------------------------------------------------
    # Browser Lifecycle Management
    # --------------------------------------------------------------------------

    def _ensure_browser(self) -> Page:
        """
        Lazily initializes the Chromium browser, browser context, and main page.
        Configures fixed viewport size and device scale factor for deterministic coordinates.
        """
        if self._page is not None and not self._page.is_closed():
            return self._page

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
            # Set default navigation timeout
            self._context.set_default_timeout(self.timeout_ms)

        self._page = self._context.new_page()
        return self._page

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

    def __enter__(self) -> "PlaywrightBrowserEnvironment":
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
        Uses page.aria_snapshot() (Playwright 1.49+) or page.accessibility.snapshot() (legacy).
        Returns None if page is unavailable or extraction fails.
        """
        try:
            page = self._ensure_browser()
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

        # JavaScript evaluation query to locate visible interactive elements
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
                
                // Check if element is within or overlapping the current viewport
                if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) {
                    continue;
                }

                // Compute normalized bounding box in [0, 1]
                const x_min = Math.max(0.0, Math.min(1.0, rect.left / vw));
                const y_min = Math.max(0.0, Math.min(1.0, rect.top / vh));
                const x_max = Math.max(0.0, Math.min(1.0, rect.right / vw));
                const y_max = Math.max(0.0, Math.min(1.0, rect.bottom / vh));

                // Normalize tag and role
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

                // Extract visible text representation
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
                // Cap text length to prevent prompt inflation
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

    def capture_screenshot(
        self,
        as_tensor: bool = True,
    ) -> Optional[Union[torch.Tensor, Any]]:
        """
        Captures full viewport screenshot as normalized [3, H, W] float tensor
        or PIL Image.
        """
        page = self._ensure_browser()
        try:
            png_bytes = page.screenshot(type="png", full_page=False)
        except Exception:
            if as_tensor:
                # Return empty black canvas tensor if capture fails
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

            # Convert to PyTorch float tensor normalized to [0, 1]: [3, H, W]
            import numpy as np
            img_np = np.array(image, dtype=np.float32) / 255.0
            # [H, W, 3] -> [3, H, W]
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
        """Formats concise text observation summarizing current URL, title, and interactive DOM elements."""
        if self.current_state is None:
            return "[Page: None]"

        url = self.current_state.url
        title = self.current_state.title
        elements = self.current_state.elements

        parts: List[str] = [
            f"[URL] {url}",
            f"[Title] {title}",
            f"[Goal] {self.current_goal}",
        ]

        if elements:
            parts.append("[Interactive Elements]")
            # Cap at top 25 elements to stay within prompt context budget
            for el in elements[:25]:
                parts.append(el.to_dom_string())
        else:
            parts.append("[Interactive Elements] (None found in active viewport)")

        return "\n".join(parts)

    def _generate_obs_features(self) -> torch.Tensor:
        """Generates deterministic dense representation vector [hidden_dim] for JEPA world model."""
        if self.current_state is None:
            return torch.zeros(self.hidden_dim, device=self.device, dtype=self.dtype)

        obs_str = f"{self.current_state.url}::{self.current_state.title}::{len(self.current_state.elements)}"
        hash_digest = hashlib.sha256(obs_str.encode("utf-8")).digest()
        seed_ints = [hash_digest[i % len(hash_digest)] for i in range(self.hidden_dim)]
        raw_feat = torch.tensor(seed_ints, device=self.device, dtype=torch.float32)
        # Normalize to zero mean, unit variance
        mean = raw_feat.mean()
        std = raw_feat.std().clamp(min=1e-5)
        norm_feat = (raw_feat - mean) / std
        return norm_feat.to(dtype=self.dtype)

    # --------------------------------------------------------------------------
    # Environment Step & Reset Lifecycle
    # --------------------------------------------------------------------------

    def reset(
        self,
        url: str = "about:blank",
        goal: Optional[str] = None,
    ) -> PlaywrightStepResult:
        """
        Resets environment and navigates to target URL.

        Args:
            url: Starting URL or HTML string ('about:blank', 'http://...', 'data:text/html,...').
            goal: Optional task goal string.

        Returns:
            PlaywrightStepResult containing initial observation and screenshot tensor.
        """
        self.action_history.clear()
        self.goal_satisfied = False
        self.is_terminated = False
        if goal is not None:
            self.current_goal = goal

        page = self._ensure_browser()

        # Navigate to requested target
        nav_error: Optional[str] = None
        if url and url != "about:blank":
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            except Exception as exc:
                nav_error = str(exc)

        # Retrieve current page metadata
        try:
            curr_url = page.url or url
            curr_title = page.title() or "Untitled"
        except Exception:
            curr_url = url
            curr_title = "Error Page"

        # Extract DOM elements
        elements = self.extract_interactive_elements()
        screenshot = self.capture_screenshot(as_tensor=True)
        raw_tree = self.get_accessibility_snapshot()

        meta = {
            "stage": "reset",
            "navigation_error": nav_error,
        }

        self.current_state = EnvironmentState(
            url=curr_url,
            title=curr_title,
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

        return PlaywrightStepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=0.0,
            done=False,
            truncated=False,
            screenshot=screenshot,
            raw_accessibility_tree=raw_tree,
            info={
                "step": 0,
                "url": curr_url,
                "goal": self.current_goal,
                "navigation_error": nav_error,
                "screenshot": screenshot,
                "elements": elements,
            },
        )

    def step(
        self,
        action_type: Union[ActionType, int, str],
        coordinates: Optional[Tuple[float, float]] = None,
        argument: Optional[str] = None,
    ) -> PlaywrightStepResult:
        """
        Dispatches native browser action primitive into the live Chromium session:
        - CLICK: Denormalizes (x, y) coordinates to viewport pixels via page.mouse.click.
        - TYPE: Types string argument into active input via page.keyboard.type.
        - SCROLL: Dispatches vertical mouse wheel scroll via page.mouse.wheel.
        - NAVIGATE: Navigates page to argument URL via page.goto.
        - FINISH: Flags episode completion.

        Args:
            action_type: Primitive identifier (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH).
            coordinates: Continuous (x, y) coordinates in [0, 1].
            argument: String payload argument.

        Returns:
            PlaywrightStepResult with updated observation, screenshot, reward, and done flag.
        """
        if self.current_state is None:
            raise RuntimeError("Environment must be reset() before calling step().")

        if self.is_terminated:
            return PlaywrightStepResult(
                obs_text="Episode already terminated.",
                obs_features=self._generate_obs_features(),
                reward=0.0,
                done=True,
                truncated=False,
                screenshot=self.capture_screenshot(as_tensor=True),
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
        step_reward = -0.05  # Step penalty to promote rapid task completion
        action_success = False
        action_error: Optional[str] = None
        hit_element: Optional[UIElement] = None

        page = self._ensure_browser()

        # Hit test against current UI elements for spatial tracking
        cx, cy = coords
        for el in self.current_state.elements:
            if el.contains(cx, cy):
                hit_element = el
                break

        # ----------------------------------------------------------------------
        # Native Browser Action Dispatch
        # ----------------------------------------------------------------------
        try:
            if act_enum == ActionType.CLICK:
                # Denormalize coordinates to viewport pixels
                pixel_x = float(coords[0] * self.viewport_width)
                pixel_y = float(coords[1] * self.viewport_height)
                page.mouse.click(pixel_x, pixel_y)
                action_success = True
                if hit_element is not None:
                    step_reward += 0.1  # Reward hitting interactable element

            elif act_enum == ActionType.TYPE:
                # Type argument string into active element
                text_to_type = arg_str if arg_str else "test"
                page.keyboard.type(text_to_type, delay=20)
                action_success = True
                step_reward += 0.05

            elif act_enum == ActionType.SCROLL:
                # Continuous scroll: vertical delta
                # Normalized coordinates can dictate scroll direction: cy > 0.5 scrolls down
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
                self.is_terminated = True
                self.goal_satisfied = True
                step_reward += 1.0  # Reward completing task

            else:
                action_error = f"Unsupported browser action primitive: {act_enum.name}"

            # Wait briefly for page mutations/animations to settle
            try:
                page.wait_for_timeout(250)
            except Exception:
                pass

        except Exception as exc:
            action_success = False
            action_error = str(exc)

        # Truncate if maximum step horizon reached
        truncated = current_step >= self.max_steps
        if truncated:
            self.is_terminated = True

        # Refresh page state
        try:
            curr_url = page.url or self.current_state.url
            curr_title = page.title() or self.current_state.title
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

        # Log action history
        action_log = {
            "step": current_step,
            "action_type": act_enum.name,
            "coordinates": coords,
            "argument": arg_str,
            "success": action_success,
            "hit_element": hit_element.text if hit_element else None,
            "error": action_error,
        }
        self.action_history.append(action_log)

        obs_text = self._format_observation_text()
        if action_error:
            obs_text = f"[Action Error: {action_error}]\n{obs_text}"
        obs_features = self._generate_obs_features()

        return PlaywrightStepResult(
            obs_text=obs_text,
            obs_features=obs_features,
            reward=step_reward,
            done=self.is_terminated,
            truncated=truncated,
            screenshot=screenshot,
            raw_accessibility_tree=raw_tree,
            info={
                "step": current_step,
                "url": curr_url,
                "action_type": act_enum.name,
                "action_success": action_success,
                "hit_element": hit_element.text if hit_element else None,
                "action_error": action_error,
                "goal_satisfied": self.goal_satisfied,
                "cumulative_reward": self.current_state.cumulative_reward,
                "screenshot": screenshot,
                "elements": elements,
            },
        )
