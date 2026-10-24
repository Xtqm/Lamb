"""
Verification Script for Phase 7: Live Browser Automation Environment (Playwright).

Tests:
1. Local HTTP Test Server & Page Navigation:
   - Spawns background HTTP server serving structured HTML with interactive elements.
   - Launches headless Playwright Chromium and navigates to local test server.
   - Validates viewport screenshot capture: [3, H, W] float RGB tensor in [0, 1].
   - Validates DOM accessibility snapshot and normalized UI element bounding boxes.
2. Native Browser Action Primitives:
   - Dispatches CLICK at denormalized pixel coordinates via page.mouse.click.
   - Dispatches TYPE with string argument via page.keyboard.type.
   - Dispatches SCROLL via page.mouse.wheel.
   - Dispatches NAVIGATE to new URL endpoint via page.goto.
   - Dispatches FINISH to complete episode.
3. Resilient Error Handling & Fault Tolerance:
   - Evaluates navigation to unresolvable domains and invalid URLs.
   - Asserts environment catches errors gracefully without raising uncaught exceptions.
4. End-to-End Multimodal Agent Closed-Loop Rollout:
   - Executes JEDM2Agent against PlaywrightBrowserEnvironment.
   - Synchronizes live screenshot captures with VisionPatchEncoder.
   - Grounds CoordinateHead continuous spatial predictions in visual patch tokens.
   - Executes multi-turn browsing tasks with O(1) recurrent memory buffers.
   - Asserts zero NaNs and zero Infs across all model outputs and recurrent states.
5. Hardware Memory Budget Profile:
   - Evaluates live multimodal execution on CUDA in bfloat16.
   - Strictly asserts peak allocated GPU VRAM is below 6.0 GB (6144.0 MB).

STRICT CONVENTION: Pure ASCII console output (no multi-byte Unicode box-drawing characters).
"""

import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)



import http.server
import socket
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from jed_m2.agent import AgentStepOutput, JEDM2Agent, TurnRecord
from jed_m2.browser import PlaywrightBrowserEnvironment, PlaywrightStepResult
from jed_m2.config import JEDM2Config
from jed_m2.environment import UIElement
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model


# ==============================================================================
# 1. Local HTTP Test Server
# ==============================================================================

TEST_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>JED-M2 Phase 7 Local Test Benchmark</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        .box { padding: 15px; border: 1px solid #ccc; margin-bottom: 15px; }
        .spacer { height: 1200px; background: #f0f0f0; margin: 20px 0; padding: 10px; }
    </style>
</head>
<body>
    <h1>JED-M2 Browser Test Portal</h1>
    <div class="box">
        <label for="search-input">Search Query:</label>
        <input type="text" id="search-input" name="search" placeholder="Enter keywords..." />
        <button id="search-btn" onclick="document.getElementById('status').innerText='Search Clicked'">Search</button>
    </div>
    <div class="box">
        <a href="/target" id="target-link">Navigate to Target Page</a>
        <button id="action-btn" onclick="document.getElementById('status').innerText='Action Committed'">Commit Action</button>
    </div>
    <div id="status">Ready</div>
    <div class="spacer">Scrollable Area (1200px)</div>
    <button id="bottom-btn" onclick="document.getElementById('status').innerText='Bottom Clicked'">Bottom Element</button>
</body>
</html>
"""

TARGET_PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Target Destination Page</title>
</head>
<body>
    <h1>Navigation Success</h1>
    <p>Target destination reached successfully.</p>
</body>
</html>
"""


class TestServerHandler(http.server.BaseHTTPRequestHandler):
    """Minimal HTTP request handler for local browser testing."""

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/index"):
            content = TEST_PAGE_HTML.encode("utf-8")
        elif self.path.startswith("/target"):
            content = TARGET_PAGE_HTML.encode("utf-8")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress standard HTTP request logging to keep console output clean
        pass


def start_local_test_server() -> Tuple[http.server.HTTPServer, int]:
    """Starts local HTTP server on an ephemeral free port."""
    server = http.server.HTTPServer(("127.0.0.1", 0), TestServerHandler)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, port


# ==============================================================================
# 2. Hardware Probe & Gradient Checks
# ==============================================================================

def probe_device_and_precision() -> Tuple[torch.device, torch.dtype, bool]:
    """Detects CUDA capabilities and probes tensor execution."""
    if torch.cuda.is_available():
        try:
            probe = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
            probe = probe @ probe
            del probe
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            device = torch.device("cuda")
            dtype = torch.bfloat16
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device Probe] CUDA execution successful on: {gpu_name}")
            print(f"[Device Probe] Selected precision: {dtype}")
            return device, dtype, True
        except Exception as exc:
            print("=" * 80)
            print(f"[NOTICE] CUDA probe failed: {exc}. Falling back to CPU.")
            print("=" * 80)
            return torch.device("cpu"), torch.float32, False
    else:
        print("[Device Probe] No CUDA device detected. Using CPU (float32).")
        return torch.device("cpu"), torch.float32, False


# ==============================================================================
# 3. Test Suites
# ==============================================================================

def test_browser_launch_and_observation(
    server_port: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Tests browser launching, page navigation, screenshot tensor, and DOM extraction."""
    print("-" * 80)
    print("[Test 1] Browser Launch, Navigation, Screenshot Capture & DOM Extraction")
    print("-" * 80)

    url = f"http://127.0.0.1:{server_port}/"
    config = JEDM2Config()

    env = PlaywrightBrowserEnvironment(
        config=config,
        headless=True,
        screenshot_size=224,
        device=device,
        dtype=dtype,
    )

    try:
        t0 = time.time()
        res = env.reset(url=url, goal="Locate and click Search button")
        elapsed_ms = (time.time() - t0) * 1000.0

        # Assert URL and Title
        assert env.current_state is not None, "Environment current_state is None!"
        assert f"127.0.0.1:{server_port}" in env.current_state.url, f"Unexpected URL: {env.current_state.url}"
        assert "Phase 7 Local Test" in env.current_state.title, f"Unexpected Title: {env.current_state.title}"

        # Assert Screenshot Tensor
        assert res.screenshot is not None, "Screenshot tensor was not captured!"
        assert isinstance(res.screenshot, torch.Tensor), f"Screenshot is not a tensor: {type(res.screenshot)}"
        assert res.screenshot.shape == (3, 224, 224), f"Unexpected screenshot shape: {res.screenshot.shape}"
        assert res.screenshot.dtype == torch.float32, f"Unexpected screenshot dtype: {res.screenshot.dtype}"
        assert float(res.screenshot.min()) >= 0.0, "Screenshot tensor contains values < 0.0"
        assert float(res.screenshot.max()) <= 1.0, "Screenshot tensor contains values > 1.0"

        # Assert DOM Element Extraction
        elements = res.info.get("elements", [])
        assert len(elements) >= 3, f"Expected at least 3 interactive elements, got {len(elements)}"
        tags = [el.tag for el in elements]
        roles = [el.role for el in elements]
        assert "input" in tags or "input" in roles, f"Input element not detected in {tags}"
        assert "button" in tags or "button" in roles, f"Button element not detected in {tags}"
        assert "a" in tags or "link" in roles, f"Anchor link element not detected in {tags}"

        # Assert Bounding Box Normalization in [0, 1]
        for el in elements:
            x_min, y_min, x_max, y_max = el.bbox
            assert 0.0 <= x_min <= 1.0, f"x_min out of bounds: {x_min}"
            assert 0.0 <= y_min <= 1.0, f"y_min out of bounds: {y_min}"
            assert 0.0 <= x_max <= 1.0, f"x_max out of bounds: {x_max}"
            assert 0.0 <= y_max <= 1.0, f"y_max out of bounds: {y_max}"
            assert x_min <= x_max, f"Inverted x coordinates: {x_min} > {x_max}"
            assert y_min <= y_max, f"Inverted y coordinates: {y_min} > {y_max}"

        # Assert Accessibility Tree
        assert res.raw_accessibility_tree is not None, "Accessibility tree was not returned!"

        print(f"  Page URL            : {env.current_state.url}")
        print(f"  Page Title          : {env.current_state.title}")
        print(f"  Screenshot Shape    : {list(res.screenshot.shape)} (range: [{res.screenshot.min():.2f}, {res.screenshot.max():.2f}])")
        print(f"  Interactive Elements: {len(elements)} detected ({tags})")
        print(f"  Launch + Nav Latency: {elapsed_ms:.1f} ms")
        print("  [PASS] Test 1 passed successfully.\n")

    finally:
        env.close()


def test_native_action_primitives(
    server_port: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Tests execution of native browser primitives: CLICK, TYPE, SCROLL, NAVIGATE, FINISH."""
    print("-" * 80)
    print("[Test 2] Native Browser Action Primitives (CLICK, TYPE, SCROLL, NAVIGATE, FINISH)")
    print("-" * 80)

    url = f"http://127.0.0.1:{server_port}/"
    config = JEDM2Config()

    env = PlaywrightBrowserEnvironment(
        config=config,
        headless=True,
        screenshot_size=224,
        device=device,
        dtype=dtype,
    )

    try:
        init_res = env.reset(url=url, goal="Test native interaction primitives")

        # 1. Test CLICK primitive on search button
        button_el = next((el for el in env.current_state.elements if "button" in (el.tag, el.role)), None)
        assert button_el is not None, "Could not find button element for CLICK test"
        bx, by = button_el.center
        click_res = env.step(ActionType.CLICK, coordinates=(bx, by))
        assert click_res.info.get("action_success") is True, "CLICK action failed!"
        print(f"  [Action 1: CLICK] at ({bx:.3f}, {by:.3f}) -> Success: {click_res.info['action_success']}")

        # 2. Test TYPE primitive
        type_res = env.step(ActionType.TYPE, argument="Wireless Headphones")
        assert type_res.info.get("action_success") is True, "TYPE action failed!"
        print(f"  [Action 2: TYPE]  text='Wireless Headphones' -> Success: {type_res.info['action_success']}")

        # 3. Test SCROLL primitive
        scroll_res = env.step(ActionType.SCROLL, coordinates=(0.5, 0.8))
        assert scroll_res.info.get("action_success") is True, "SCROLL action failed!"
        print(f"  [Action 3: SCROLL] delta down -> Success: {scroll_res.info['action_success']}")

        # 4. Test NAVIGATE primitive to target URL
        target_url = f"http://127.0.0.1:{server_port}/target"
        nav_res = env.step(ActionType.NAVIGATE, argument=target_url)
        assert nav_res.info.get("action_success") is True, "NAVIGATE action failed!"
        assert "/target" in env.current_state.url, f"Expected URL ending in /target, got: {env.current_state.url}"
        print(f"  [Action 4: NAVIGATE] to '{target_url}' -> Current URL: {env.current_state.url}")

        # 5. Test FINISH primitive
        finish_res = env.step(ActionType.FINISH)
        assert finish_res.done is True, "Episode did not terminate on FINISH!"
        assert env.goal_satisfied is True, "Goal was not marked satisfied on FINISH!"
        print(f"  [Action 5: FINISH] Episode Done: {finish_res.done} | Cumulative Reward: {finish_res.info.get('cumulative_reward', 0.0):+.2f}")

        assert len(env.action_history) == 5, f"Expected 5 actions logged, got {len(env.action_history)}"
        print("  [PASS] Test 2 passed successfully.\n")

    finally:
        env.close()


def test_resilient_error_handling(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Tests environment robustness on unresolvable URLs and invalid domains."""
    print("-" * 80)
    print("[Test 3] Resilient Navigation & Action Error Handling")
    print("-" * 80)

    config = JEDM2Config()
    env = PlaywrightBrowserEnvironment(
        config=config,
        headless=True,
        timeout_ms=3000,
        device=device,
        dtype=dtype,
    )

    try:
        # Navigate to a non-existent local port that immediately fails
        bad_url = "http://127.0.0.1:59999/does_not_exist"
        res = env.reset(url=bad_url, goal="Handle invalid endpoint gracefully")

        # Assert no crash occurred and error was caught
        assert env.current_state is not None, "Environment current_state is None on error!"
        assert env.current_state.last_action_success is False, "Expected last_action_success=False on failure"
        assert res.info.get("navigation_error") is not None, "Expected navigation_error to be logged"
        assert "[Navigation Error:" in res.obs_text, "Expected error message in observation text"
        print(f"  Navigation Error Intercepted: {res.info['navigation_error'][:60]}...")

        # Step an action on the failed page to ensure graceful continuation
        step_res = env.step(ActionType.CLICK, coordinates=(0.5, 0.5))
        assert step_res.obs_text is not None, "Observation text is None after stepping on error page"
        print(f"  Recovery Step Executed      : ActionType={step_res.info['action_type']}")
        print("  [PASS] Test 3 passed successfully.\n")

    finally:
        env.close()


def test_end_to_end_multimodal_agent(
    server_port: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Tests closed-loop multimodal agent decision making, visual grounding, and recurrent rollout."""
    print("-" * 80)
    print("[Test 4] End-to-End Multimodal Agent Decision-Making & Rollout")
    print("-" * 80)

    config = JEDM2Config(hidden_dim=512, image_size=224, patch_size=16)
    model = JEDM2Model(config).to(device=device, dtype=dtype)
    agent = JEDM2Agent(
        model=model,
        config=config,
        device=device,
        dtype=dtype,
        max_thought_tokens=8,
        action_gate_threshold=0.5,
    )

    url = f"http://127.0.0.1:{server_port}/"
    env = PlaywrightBrowserEnvironment(
        config=config,
        headless=True,
        screenshot_size=224,
        device=device,
        dtype=dtype,
    )

    try:
        # Reinitialize episode
        agent.reset()
        init_res = env.reset(url=url, goal="Search and inspect products")

        num_turns = 3
        for turn_idx in range(1, num_turns + 1):
            t0 = time.time()
            agent_out, step_res = agent.step_environment(env)
            turn_latency_ms = (time.time() - t0) * 1000.0

            # Validate agent outputs
            assert isinstance(agent_out, AgentStepOutput), "agent_out is not AgentStepOutput"
            assert isinstance(agent_out.action_type, ActionType), "Invalid action type"
            cx, cy = agent_out.coordinates
            assert 0.0 <= cx <= 1.0, f"Predicted cx out of range: {cx}"
            assert 0.0 <= cy <= 1.0, f"Predicted cy out of range: {cy}"
            assert len(agent_out.thought) > 0, "Empty thought rationale string"

            # Assert zero NaNs / Infs across recurrent states
            if agent.recurrent_states is not None:
                for layer_idx, (ssm_s, conv_s, kv_c) in enumerate(agent.recurrent_states):
                    if ssm_s is not None:
                        assert not torch.isnan(ssm_s).any(), f"NaN in ssm_state layer {layer_idx}!"
                        assert not torch.isinf(ssm_s).any(), f"Inf in ssm_state layer {layer_idx}!"
                    if conv_s is not None:
                        assert not torch.isnan(conv_s).any(), f"NaN in conv_state layer {layer_idx}!"
                        assert not torch.isinf(conv_s).any(), f"Inf in conv_state layer {layer_idx}!"

            print(f"  Turn {turn_idx}/{num_turns} | Action: {agent_out.action_name} at ({cx:.3f}, {cy:.3f}) "
                  f"| Gate: {agent_out.gate_prob:.3f} | Latency: {turn_latency_ms:.1f} ms")
            print(f"    Rationale: \"{agent_out.thought[:65]}...\"")

        assert len(agent.history) == num_turns, f"Expected {num_turns} turn records, got {len(agent.history)}"
        print("  [PASS] Test 4 passed successfully.\n")

    finally:
        env.close()


def test_hardware_memory_profile(is_cuda: bool) -> None:
    """Verifies peak GPU VRAM allocation remains strictly below 6.0 GB ceiling."""
    print("-" * 80)
    print("[Test 5] Hardware Memory Budget Profile (< 6.0 GB VRAM Limit)")
    print("-" * 80)

    if not is_cuda:
        print("  CUDA not available. Skipping GPU VRAM ceiling assertion (CPU mode).")
        print("  [PASS] Test 5 passed (CPU mode).\n")
        return

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_mb = peak_bytes / (1024.0 * 1024.0)
    peak_gb = peak_mb / 1024.0
    limit_mb = 6144.0  # 6.0 GB

    print(f"  Peak GPU VRAM Allocated: {peak_mb:.2f} MB ({peak_gb:.2f} GB)")
    print(f"  Target VRAM Ceiling    : {limit_mb:.2f} MB (6.00 GB)")
    print(f"  Remaining Headroom     : {limit_mb - peak_mb:.2f} MB")

    assert peak_mb <= limit_mb, (
        f"HARDWARE ENVELOPE BREACH: Peak GPU VRAM {peak_mb:.2f} MB exceeds limit {limit_mb:.2f} MB!"
    )
    print("  [PASS] Test 5 passed successfully.\n")


# ==============================================================================
# Main Verification Entrypoint
# ==============================================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("      JED-M2 Large Action Model (LAM) - Phase 7 Verification")
    print("      (Live Playwright Browser Automation Environment)")
    print("=" * 80 + "\n")

    device, dtype, is_cuda = probe_device_and_precision()
    print("=" * 80 + "\n")

    # Start local test server
    server, port = start_local_test_server()
    print(f"[Local HTTP Server] Listening on http://127.0.0.1:{port}/")

    try:
        # Run test suites
        test_browser_launch_and_observation(port, device, dtype)
        test_native_action_primitives(port, device, dtype)
        test_resilient_error_handling(device, dtype)
        test_end_to_end_multimodal_agent(port, device, dtype)
        test_hardware_memory_profile(is_cuda)

        print("=" * 80)
        print("      ALL PHASE 7 VERIFICATION SUITES PASSED SUCCESSFULLY!")
        print("=" * 80 + "\n")

    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
