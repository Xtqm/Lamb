"""
JED-M2 Windows OS Automation Controller.
Provides:
- Windows DPI Awareness via ctypes.windll.shcore.SetProcessDpiAwareness(2).
- Continuous normalized coordinate mapping [0, 1] -> physical screen pixels (X, Y).
- Native mouse dispatch (left, right, double click, smooth Bezier curves, drag-and-drop, scroll).
- Native keyboard dispatch (jittered text typing, special keys, hotkeys).
- High-speed screen perception capturing RGB tensors directly ingestible by VisionPatchEncoder.
- Mock mode support for headless testing and verification.
"""

import ctypes
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageGrab

from jed_m2.heads import ActionType

# Check if PyAutoGUI is available
try:
    import pyautogui
    # Configure pyautogui fail-safe
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.01
    HAS_PYAUTOGUI = True
except ImportError:
    pyautogui = None
    HAS_PYAUTOGUI = False


# ==============================================================================
# Windows API Constants & Type Definitions
# ==============================================================================

# DPI Awareness Constants
PROCESS_DPI_UNAWARE = 0
PROCESS_SYSTEM_DPI_AWARE = 1
PROCESS_PER_MONITOR_DPI_AWARE = 2

# Mouse Event Flags
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

# Virtual Key Codes
VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12  # Alt
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_DELETE = 0x2E

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

SPECIAL_KEY_MAP = {
    "enter": VK_RETURN,
    "return": VK_RETURN,
    "tab": VK_TAB,
    "backspace": VK_BACK,
    "escape": VK_ESCAPE,
    "esc": VK_ESCAPE,
    "space": VK_SPACE,
    "ctrl": VK_CONTROL,
    "control": VK_CONTROL,
    "alt": VK_MENU,
    "shift": VK_SHIFT,
    "left": VK_LEFT,
    "up": VK_UP,
    "right": VK_RIGHT,
    "down": VK_DOWN,
    "delete": VK_DELETE,
}


# ==============================================================================
# Windows OS Automation Controller
# ==============================================================================

class WindowsOSController:
    """
    Pure Python Windows OS Automation Controller.
    Ensures per-monitor DPI awareness and maps continuous LAM predictions
    to physical screen pixels and peripheral hardware events.
    """

    def __init__(
        self,
        screen_size: Optional[Tuple[int, int]] = None,
        display_scale: float = 1.0,
        mock_mode: bool = False,
    ):
        """
        Initializes the Windows OS controller.

        Args:
            screen_size: Optional fixed (width, height) physical pixel bounds.
            display_scale: Display scaling factor (e.g. 1.0, 1.25, 1.5).
            mock_mode: When True, records actions without issuing OS-level events.
        """
        self.mock_mode = mock_mode
        self.display_scale = display_scale
        self.action_history: List[Dict[str, Any]] = []

        # Initialize Windows DPI Awareness
        self.dpi_aware = self._init_dpi_awareness()

        # Resolve screen dimensions
        if screen_size is not None:
            self.screen_width, self.screen_height = screen_size
        else:
            self.screen_width, self.screen_height = self._query_screen_dimensions()

        # Simulated cursor position for mock or relative queries
        self._cursor_x = self.screen_width // 2
        self._cursor_y = self.screen_height // 2

    def _init_dpi_awareness(self) -> bool:
        """
        Registers Per-Monitor DPI Awareness (mode 2) via Windows shcore.dll.
        Guarantees physical pixel coordinates across 100%, 125%, 150%+ display scales.
        """
        if sys.platform != "win32":
            return False

        try:
            # SetProcessDpiAwareness(2) -> PROCESS_PER_MONITOR_DPI_AWARE
            shcore = ctypes.windll.shcore
            hr = shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)
            # S_OK = 0, E_ACCESSDENIED = 0x80070005 (already set)
            return hr in (0, -2147024891)
        except (AttributeError, OSError):
            try:
                # Fallback to SetProcessDPIAware in user32
                user32 = ctypes.windll.user32
                return bool(user32.SetProcessDPIAware())
            except (AttributeError, OSError):
                return False

    def _query_screen_dimensions(self) -> Tuple[int, int]:
        """Queries physical screen width and height from Windows user32."""
        if sys.platform == "win32":
            try:
                user32 = ctypes.windll.user32
                w = user32.GetSystemMetrics(0)  # SM_CXSCREEN
                h = user32.GetSystemMetrics(1)  # SM_CYSCREEN
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass

        if HAS_PYAUTOGUI:
            try:
                w, h = pyautogui.size()
                return int(w), int(h)
            except Exception:
                pass

        return 1920, 1080

    def set_display_bounds(self, width: int, height: int, display_scale: float = 1.0) -> None:
        """Manually sets virtual display boundaries and scaling for calibration or testing."""
        self.screen_width = width
        self.screen_height = height
        self.display_scale = display_scale

    # --------------------------------------------------------------------------
    # Coordinate Mapping
    # --------------------------------------------------------------------------

    def normalized_to_screen(
        self,
        x: float,
        y: float,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Translates continuous normalized model predictions [0, 1] to physical screen pixels.

        Args:
            x: Normalized horizontal coordinate in [0.0, 1.0].
            y: Normalized vertical coordinate in [0.0, 1.0].
            bounds: Optional (left, top, right, bottom) pixel bounding box.

        Returns:
            Tuple of discrete (X_screen, Y_screen) physical pixels.
        """
        # Clamp inputs safely into [0.0, 1.0]
        x_clamped = max(0.0, min(1.0, float(x)))
        y_clamped = max(0.0, min(1.0, float(y)))

        if bounds is not None:
            left, top, right, bottom = bounds
            w = max(1, right - left)
            h = max(1, bottom - top)
            px = int(round(left + x_clamped * (w - 1)))
            py = int(round(top + y_clamped * (h - 1)))
            return max(left, min(right - 1, px)), max(top, min(bottom - 1, py))

        # Default full screen mapping
        px = int(round(x_clamped * (self.screen_width - 1)))
        py = int(round(y_clamped * (self.screen_height - 1)))
        return max(0, min(self.screen_width - 1, px)), max(0, min(self.screen_height - 1, py))

    def screen_to_normalized(
        self,
        px: int,
        py: int,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[float, float]:
        """Inverse mapping: translates physical screen pixels to normalized coordinates [0, 1]."""
        if bounds is not None:
            left, top, right, bottom = bounds
            w = max(1, right - left)
            h = max(1, bottom - top)
            nx = (px - left) / float(w - 1) if w > 1 else 0.0
            ny = (py - top) / float(h - 1) if h > 1 else 0.0
            return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

        nx = px / float(self.screen_width - 1) if self.screen_width > 1 else 0.0
        ny = py / float(self.screen_height - 1) if self.screen_height > 1 else 0.0
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    # --------------------------------------------------------------------------
    # Mouse Dispatch
    # --------------------------------------------------------------------------

    def get_cursor_pos(self) -> Tuple[int, int]:
        """Returns the current mouse cursor coordinates (px, py)."""
        if self.mock_mode or sys.platform != "win32":
            return self._cursor_x, self._cursor_y

        try:
            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            pt = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return pt.x, pt.y
        except Exception:
            return self._cursor_x, self._cursor_y

    def _set_cursor_pos(self, px: int, py: int) -> None:
        """Sets cursor position directly via Windows user32 or internal state."""
        self._cursor_x = px
        self._cursor_y = py
        if not self.mock_mode and sys.platform == "win32":
            try:
                ctypes.windll.user32.SetCursorPos(px, py)
            except Exception:
                pass

    def move_to(
        self,
        x: float,
        y: float,
        duration: float = 0.1,
        human_like: bool = True,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Moves the mouse smoothly to the normalized target coordinates (x, y).

        Args:
            x: Target normalized X in [0, 1].
            y: Target normalized Y in [0, 1].
            duration: Duration of movement in seconds.
            human_like: When True, follows a perturbed Bezier curve.
            bounds: Optional bounding rect.

        Returns:
            Tuple of final (target_px, target_py).
        """
        target_px, target_py = self.normalized_to_screen(x, y, bounds=bounds)
        start_px, start_py = self.get_cursor_pos()

        record = {
            "type": "move",
            "from": (start_px, start_py),
            "to": (target_px, target_py),
            "normalized": (x, y),
            "human_like": human_like,
        }
        self.action_history.append(record)

        if self.mock_mode:
            self._set_cursor_pos(target_px, target_py)
            return target_px, target_py

        if HAS_PYAUTOGUI and not human_like:
            pyautogui.moveTo(target_px, target_py, duration=duration)
            self._set_cursor_pos(target_px, target_py)
            return target_px, target_py

        # Generate smooth Bezier curve points
        steps = max(5, int(duration * 60))
        # Control points for quadratic Bezier with slight random deviation
        mid_x = (start_px + target_px) / 2.0 + (random.uniform(-20, 20) if human_like else 0)
        mid_y = (start_py + target_py) / 2.0 + (random.uniform(-20, 20) if human_like else 0)

        dt = duration / float(steps)
        for i in range(1, steps + 1):
            t = i / float(steps)
            # Quadratic Bezier formula: (1-t)^2 * P0 + 2*(1-t)*t * P1 + t^2 * P2
            curr_x = int(round((1 - t) ** 2 * start_px + 2 * (1 - t) * t * mid_x + t ** 2 * target_px))
            curr_y = int(round((1 - t) ** 2 * start_py + 2 * (1 - t) * t * mid_y + t ** 2 * target_py))
            self._set_cursor_pos(curr_x, curr_y)
            if dt > 0.001:
                time.sleep(dt)

        self._set_cursor_pos(target_px, target_py)
        return target_px, target_py

    def click(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        button: str = "left",
        clicks: int = 1,
        bounds: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[int, int]:
        """
        Executes a mouse click (left, right, or double click) at specified or current coordinates.
        """
        if x is not None and y is not None:
            px, py = self.move_to(x, y, duration=0.05, bounds=bounds)
        else:
            px, py = self.get_cursor_pos()

        record = {
            "type": "click",
            "pos": (px, py),
            "button": button,
            "clicks": clicks,
        }
        self.action_history.append(record)

        if self.mock_mode:
            return px, py

        if HAS_PYAUTOGUI:
            pyautogui.click(x=px, y=py, clicks=clicks, button=button)
            return px, py

        # Pure ctypes fallback for Windows
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            down_flag, up_flag = {
                "left": (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
                "right": (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
                "middle": (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
            }.get(button, (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP))

            for _ in range(clicks):
                user32.mouse_event(down_flag, 0, 0, 0, 0)
                time.sleep(random.uniform(0.015, 0.035))
                user32.mouse_event(up_flag, 0, 0, 0, 0)
                if clicks > 1:
                    time.sleep(random.uniform(0.05, 0.1))

        return px, py

    def double_click(self, x: Optional[float] = None, y: Optional[float] = None) -> Tuple[int, int]:
        """Executes a native double click."""
        return self.click(x=x, y=y, button="left", clicks=2)

    def right_click(self, x: Optional[float] = None, y: Optional[float] = None) -> Tuple[int, int]:
        """Executes a native right click."""
        return self.click(x=x, y=y, button="right", clicks=1)

    def drag_to(
        self,
        start_x: float,
        start_y: float,
        end_x: float,
        end_y: float,
        duration: float = 0.2,
    ) -> None:
        """Executes a drag-and-drop mouse gesture between normalized coordinates."""
        spx, spy = self.move_to(start_x, start_y, duration=0.05)
        epx, epy = self.normalized_to_screen(end_x, end_y)

        self.action_history.append({
            "type": "drag",
            "from": (spx, spy),
            "to": (epx, epy),
        })

        if self.mock_mode:
            self._set_cursor_pos(epx, epy)
            return

        if HAS_PYAUTOGUI:
            pyautogui.dragTo(epx, epy, duration=duration, button="left")
            return

        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            self.move_to(end_x, end_y, duration=duration, human_like=True)
            time.sleep(0.05)
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def scroll(self, clicks: int, x: Optional[float] = None, y: Optional[float] = None) -> None:
        """Dispatches mouse wheel vertical scroll events."""
        if x is not None and y is not None:
            self.move_to(x, y, duration=0.05)

        self.action_history.append({"type": "scroll", "clicks": clicks})

        if self.mock_mode:
            return

        if HAS_PYAUTOGUI:
            pyautogui.scroll(clicks)
            return

        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            # Each click is typically 120 delta units
            wheel_delta = clicks * 120
            user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_delta, 0)

    # --------------------------------------------------------------------------
    # Keyboard Dispatch
    # --------------------------------------------------------------------------

    def type_text(self, text: str, min_delay: float = 0.01, max_delay: float = 0.04) -> None:
        """
        Emits text keystrokes with natural inter-key jitter delays.

        Args:
            text: String to type.
            min_delay: Minimum delay between keystrokes in seconds.
            max_delay: Maximum delay between keystrokes in seconds.
        """
        self.action_history.append({"type": "type_text", "text": text})

        if self.mock_mode:
            return

        if HAS_PYAUTOGUI:
            for char in text:
                pyautogui.write(char)
                delay = random.uniform(min_delay, max_delay)
                time.sleep(delay)
            return

        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            for char in text:
                # Send Unicode character via user32.keybd_event
                vk_code = ord(char)
                user32.keybd_event(0, vk_code, KEYEVENTF_UNICODE, 0)
                time.sleep(random.uniform(0.005, 0.015))
                user32.keybd_event(0, vk_code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0)
                time.sleep(random.uniform(min_delay, max_delay))

    def press_key(self, key_name: str) -> None:
        """Emits a single key press (e.g. 'enter', 'tab', 'escape', 'backspace')."""
        normalized_key = key_name.lower().strip()
        self.action_history.append({"type": "press_key", "key": normalized_key})

        if self.mock_mode:
            return

        if HAS_PYAUTOGUI:
            pyautogui.press(normalized_key)
            return

        if sys.platform == "win32" and normalized_key in SPECIAL_KEY_MAP:
            user32 = ctypes.windll.user32
            vk = SPECIAL_KEY_MAP[normalized_key]
            user32.keybd_event(vk, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

    def hotkey(self, *keys: str) -> None:
        """Dispatches key combinations simultaneously (e.g. hotkey('ctrl', 'c'))."""
        normalized_keys = [k.lower().strip() for k in keys]
        self.action_history.append({"type": "hotkey", "keys": normalized_keys})

        if self.mock_mode:
            return

        if HAS_PYAUTOGUI:
            pyautogui.hotkey(*normalized_keys)
            return

        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            vk_list = [SPECIAL_KEY_MAP.get(k, ord(k.upper())) for k in normalized_keys]
            # Key down in sequence
            for vk in vk_list:
                user32.keybd_event(vk, 0, 0, 0)
                time.sleep(0.01)
            time.sleep(0.03)
            # Key up in reverse sequence
            for vk in reversed(vk_list):
                user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
                time.sleep(0.01)

    # --------------------------------------------------------------------------
    # Screen Perception to PyTorch Tensors
    # --------------------------------------------------------------------------

    def capture_screen(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_size: Optional[Tuple[int, int]] = (224, 224),
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Captures the primary screen or target bounding region and converts it
        into a normalized PyTorch tensor directly ingestible by VisionPatchEncoder.

        Args:
            region: Optional (left, top, right, bottom) pixel bbox.
            target_size: Target (H, W) spatial resolution for patch encoding (default: 224x224).
            dtype: Tensor data type (torch.float32 or torch.bfloat16).
            device: Target torch device (CPU or CUDA).

        Returns:
            RGB PyTorch tensor of shape [1, 3, target_H, target_W] with values in [0.0, 1.0].
        """
        if self.mock_mode:
            # Generate deterministic synthetic observation frame for testing
            h, w = target_size if target_size is not None else (224, 224)
            tensor = torch.zeros((1, 3, h, w), dtype=dtype, device=device)
            # Add gradient pattern to verify spatial information
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1, h, device=device),
                torch.linspace(0, 1, w, device=device),
                indexing="ij",
            )
            tensor[0, 0] = grid_x
            tensor[0, 1] = grid_y
            tensor[0, 2] = 0.5
            return tensor

        # Capture live screen using PIL.ImageGrab
        try:
            bbox = region if region is not None else None
            img = ImageGrab.grab(bbox=bbox, all_screens=False)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to numpy and then torch tensor
            img_np = np.array(img, dtype=np.uint8)  # [H, W, 3]
            img.close()  # Clean up PIL image resource immediately

            # [H, W, 3] -> [1, 3, H, W]
            tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)
            tensor = tensor / 255.0  # Normalize to [0.0, 1.0]

            if device is not None:
                tensor = tensor.to(device=device)

            if target_size is not None:
                th, tw = target_size
                if tensor.shape[-2:] != (th, tw):
                    tensor = F.interpolate(
                        tensor,
                        size=(th, tw),
                        mode="bilinear",
                        align_corners=False,
                    )

            return tensor
        except Exception as e:
            # Fallback to zero observation frame on capture failure
            h, w = target_size if target_size is not None else (224, 224)
            return torch.zeros((1, 3, h, w), dtype=dtype, device=device)

    # --------------------------------------------------------------------------
    # Dispatch Action from Decoded LAM Action
    # --------------------------------------------------------------------------

    def dispatch_action(
        self,
        action_type: Union[ActionType, int, str],
        coordinates: Tuple[float, float],
        argument: str = "",
    ) -> Dict[str, Any]:
        """
        Executes a high-level action decision emitted by JEDM2Agent.

        Args:
            action_type: ActionType enum, integer, or name.
            coordinates: Continuous (x, y) coordinates in [0, 1].
            argument: Optional string argument (e.g. text to type or URL to navigate).

        Returns:
            Execution result dictionary.
        """
        # Resolve action type
        if isinstance(action_type, int):
            try:
                act = ActionType(action_type)
            except ValueError:
                act = ActionType.CLICK
        elif isinstance(action_type, str):
            act_name = action_type.upper()
            act = getattr(ActionType, act_name, ActionType.CLICK)
        else:
            act = action_type

        x, y = coordinates
        result = {"action": act.name, "coords": (x, y), "arg": argument, "success": True}

        if act == ActionType.CLICK:
            px, py = self.click(x=x, y=y)
            result["screen_coords"] = (px, py)
        elif act == ActionType.TYPE:
            if x is not None and y is not None:
                self.click(x=x, y=y)
            if argument:
                self.type_text(argument)
                self.press_key("enter")
        elif act == ActionType.SCROLL:
            try:
                clicks = int(argument) if argument else -3
            except ValueError:
                clicks = -3
            self.scroll(clicks, x=x, y=y)
        elif act == ActionType.NAVIGATE:
            # Hotkey to open address/run bar: Ctrl+L or Win+R, then type URL
            self.hotkey("ctrl", "l")
            time.sleep(0.05)
            if argument:
                self.type_text(argument)
                self.press_key("enter")
        elif act == ActionType.API_CALL:
            # Non-UI action or hotkey payload
            if argument:
                keys = [k.strip() for k in argument.split("+")]
                self.hotkey(*keys)
        elif act == ActionType.FINISH:
            result["status"] = "task_finished"

        return result
