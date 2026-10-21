"""
JED-M2 Interactive Agent CLI Script.
Provides an interactive terminal interface for running JED-M2 Large Action Model
in simulated Web/OS sandbox environments:
- Pure ASCII visual canvas rendering screen elements and pointer targets (Windows CP1252 safe).
- Step-by-step inspection of System 2 latent rationales and System 1 action primitives.
- Interactive mode with manual step pausing or automated benchmark execution.

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
"""

import argparse
import sys
from typing import List, Optional, Tuple
import torch

from jed_m2.agent import JEDM2Agent
from jed_m2.browser import PlaywrightBrowserEnvironment
from jed_m2.config import JEDM2Config
from jed_m2.environment import UIElement, WebOSEnvironment
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model


# ==============================================================================
# 1. Pure ASCII Screen Canvas Renderer
# ==============================================================================

class ASCIIScreenCanvas:
    """
    Renders 2D visual screen representations into pure 7-bit ASCII grids.
    Guarantees zero multi-byte Unicode characters to prevent Windows CP1252 exceptions.
    """

    def __init__(self, width: int = 70, height: int = 18):
        self.width = width
        self.height = height

    def render(
        self,
        elements: List[UIElement],
        cursor_coords: Optional[Tuple[float, float]] = None,
        cursor_label: Optional[str] = None,
    ) -> str:
        """
        Renders UI elements and cursor pointer on a 2D text canvas.

        Args:
            elements: List of UIElement instances with normalized bounding boxes.
            cursor_coords: Optional (x, y) coordinates in [0, 1].
            cursor_label: Optional label for pointer action.

        Returns:
            Multi-line string representation of the canvas.
        """
        # Initialize grid with spaces
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]

        # Draw outer borders
        for col in range(self.width):
            grid[0][col] = "-"
            grid[self.height - 1][col] = "-"
        for row in range(self.height):
            grid[row][0] = "|"
            grid[row][self.width - 1] = "|"
        grid[0][0] = "+"
        grid[0][self.width - 1] = "+"
        grid[self.height - 1][0] = "+"
        grid[self.height - 1][self.width - 1] = "+"

        # Render elements
        for el in elements:
            x_min, y_min, x_max, y_max = el.bbox
            c1 = max(1, min(self.width - 2, int(x_min * (self.width - 2)) + 1))
            c2 = max(1, min(self.width - 2, int(x_max * (self.width - 2)) + 1))
            r1 = max(1, min(self.height - 2, int(y_min * (self.height - 2)) + 1))
            r2 = max(1, min(self.height - 2, int(y_max * (self.height - 2)) + 1))

            if c2 <= c1:
                c2 = min(self.width - 2, c1 + 1)
            if r2 <= r1:
                r2 = min(self.height - 2, r1 + 1)

            # Draw box borders
            for c in range(c1, c2 + 1):
                if grid[r1][c] == " ":
                    grid[r1][c] = "-"
                if grid[r2][c] == " ":
                    grid[r2][c] = "-"
            for r in range(r1, r2 + 1):
                if grid[r][c1] == " ":
                    grid[r][c1] = "|"
                if grid[r][c2] == " ":
                    grid[r][c2] = "|"

            grid[r1][c1] = "+"
            grid[r1][c2] = "+"
            grid[r2][c1] = "+"
            grid[r2][c2] = "+"

            # Draw label inside box
            label = el.text[: (c2 - c1 - 1)]
            if not label:
                label = el.role[: (c2 - c1 - 1)]

            mid_r = (r1 + r2) // 2
            if mid_r < self.height - 1:
                for idx, ch in enumerate(label):
                    target_c = c1 + 1 + idx
                    if target_c < c2:
                        grid[mid_r][target_c] = ch

        # Render cursor pointer if present
        if cursor_coords is not None:
            cx, cy = cursor_coords
            cur_col = max(1, min(self.width - 2, int(cx * (self.width - 2)) + 1))
            cur_row = max(1, min(self.height - 2, int(cy * (self.height - 2)) + 1))

            # Draw cursor crosshair marker: [X]
            if cur_col > 1 and cur_col < self.width - 2:
                grid[cur_row][cur_col - 1] = "["
                grid[cur_row][cur_col] = "X"
                grid[cur_row][cur_col + 1] = "]"
            else:
                grid[cur_row][cur_col] = "*"

        lines = ["".join(row) for row in grid]
        return "\n".join(lines)


# ==============================================================================
# 2. Interactive CLI Execution
# ==============================================================================

def run_interactive_session(
    scenario_id: str = "ecommerce",
    custom_goal: Optional[str] = None,
    max_steps: int = 8,
    mode: str = "step",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    thought_tokens: int = 16,
    threshold: float = 0.5,
    checkpoint_path: Optional[str] = None,
    browser_type: str = "simulated",
    url: Optional[str] = None,
    headed: bool = False,
) -> None:
    """
    Runs an interactive terminal agent session with visual canvas and step inspection.
    Supports simulated WebOSEnvironment and live PlaywrightBrowserEnvironment.
    """
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"\n======================================================================")
    print(f"  JED-M2 Large Action Model (LAM) - Interactive Terminal Runtime")
    print(f"======================================================================")
    print(f"Device: {dev} | Precision: {dtype}")
    print(f"Environment: {browser_type.upper()} | Mode: {mode.upper()}")
    if browser_type == "playwright":
        print(f"Playwright Headed: {headed} | Viewport: 1280x800")
    else:
        print(f"Scenario: {scenario_id}")
    print(f"----------------------------------------------------------------------")

    # Initialize model
    config = JEDM2Config()
    model = JEDM2Model(config).to(device=dev, dtype=dtype)

    if checkpoint_path is not None:
        print(f"Loading weights from: {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=dev)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Model weights successfully loaded (missing: {len(missing)}, unexpected: {len(unexpected)}).")

    # Initialize environment and agent
    if browser_type == "playwright":
        env = PlaywrightBrowserEnvironment(
            config=config,
            headless=(not headed),
            device=dev,
            dtype=dtype,
        )
        start_url = url if url else "https://en.wikipedia.org/wiki/Main_Page"
        goal_text = custom_goal if custom_goal else "Explore page and interact with elements"
        init_res = env.reset(url=start_url, goal=goal_text)
    else:
        env = WebOSEnvironment(config=config, scenario_id=scenario_id, device=dev, dtype=dtype)
        goal_text = custom_goal if custom_goal else env.active_scenario.goal
        init_res = env.reset(scenario_id=scenario_id, custom_goal=custom_goal)

    agent = JEDM2Agent(
        model=model,
        config=config,
        device=dev,
        dtype=dtype,
        max_thought_tokens=thought_tokens,
        action_gate_threshold=threshold,
    )

    canvas = ASCIIScreenCanvas(width=72, height=18)

    print(f"Goal: {goal_text}")
    print(f"Initial Page: {env.current_state.url} ({env.current_state.title})")
    print(f"\n--- Initial Screen Canvas ---")
    print(canvas.render(env.current_state.elements))

    try:
        for step_num in range(1, max_steps + 1):
            print(f"\n======================================================================")
            print(f"Turn {step_num} / {max_steps} | URL: {env.current_state.url}")
            print(f"======================================================================")

            # Agent decision
            obs_text = env._format_observation_text()
            obs_features = env._generate_obs_features()
            screenshot = env.capture_screenshot(as_tensor=True) if hasattr(env, "capture_screenshot") else None
            agent_out = agent.act(obs_text=obs_text, obs_features=obs_features, screenshot=screenshot)

            # Print System 2 Latent Reasoning
            print(f"\n[System 2 Rationale]")
            print(f"  Thought: \"{agent_out.thought}\"")
            print(f"  Thought Tokens Generated: {agent_out.num_thought_tokens}")
            print(f"  Action Gate Probability:  {agent_out.gate_prob:.4f} (Threshold: {threshold})")

            # Print System 1 Action Decision
            cx, cy = agent_out.coordinates
            print(f"\n[System 1 Action Primitive]")
            print(f"  Action Type:  {agent_out.action_name}")
            print(f"  Coordinates:  ({cx:.3f}, {cy:.3f})")
            print(f"  Argument:     \"{agent_out.argument}\"")

            # Render screen canvas with pointer target
            print(f"\n[Visual Screen Canvas (Pointer Target: [X])]")
            print(canvas.render(env.current_state.elements, cursor_coords=agent_out.coordinates))

            # Dispatch action to environment
            step_res = env.step(
                action_type=agent_out.action_type,
                coordinates=agent_out.coordinates,
                argument=agent_out.argument,
            )

            # Print Environment Transition Feedback
            hit_name = step_res.info.get("hit_element", "None (Background)")
            print(f"\n[Environment Feedback]")
            print(f"  Hit Element:        {hit_name}")
            print(f"  Step Reward:        {step_res.reward:+.3f}")
            print(f"  Cumulative Reward:  {step_res.info.get('cumulative_reward', 0.0):+.3f}")
            print(f"  Goal Satisfied:     {step_res.info.get('goal_satisfied', False)}")
            print(f"  Episode Terminated: {step_res.done}")

            if step_res.done:
                print(f"\n======================================================================")
                if step_res.info.get("goal_satisfied", False):
                    print(f"  [SUCCESS] Task Goal Completed Successfully!")
                else:
                    print(f"  [TERMINATED] Episode terminated before goal completion.")
                print(f"  Total Turns: {step_num} | Cumulative Reward: {env.current_state.cumulative_reward:+.3f}")
                print(f"======================================================================\n")
                break

            if mode == "step" and step_num < max_steps:
                try:
                    user_in = input("\nPress [Enter] to continue to next turn (or 'q' to quit): ").strip()
                    if user_in.lower() == "q":
                        print("Session aborted by user.")
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\nSession interrupted.")
                    break

    finally:
        if hasattr(env, "close"):
            env.close()

    print("Session concluded.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="JED-M2 Interactive Agent CLI Interface")
    parser.add_argument(
        "--browser",
        type=str,
        default="simulated",
        choices=["simulated", "playwright"],
        help="Browser environment type: simulated (WebOSEnvironment sandbox) or playwright (live Chromium browser)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Initial target URL for live Playwright browser session",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch Playwright browser in headed mode (visible window)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="ecommerce",
        choices=["ecommerce", "travel", "os_admin"],
        help="Benchmark scenario domain to execute (for simulated environment)",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Optional custom task goal string",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Maximum turn steps allowed per episode",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "step"],
        help="Execution mode: auto (non-blocking) or step (pause for inspection)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--thought_tokens",
        type=int,
        default=16,
        help="Maximum reasoning tokens generated in System 2",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="ActionGate probability commitment threshold",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pt)",
    )

    args = parser.parse_args()

    run_interactive_session(
        scenario_id=args.scenario,
        custom_goal=args.goal,
        max_steps=args.steps,
        mode=args.mode,
        device=args.device,
        thought_tokens=args.thought_tokens,
        threshold=args.threshold,
        checkpoint_path=args.model_path,
        browser_type=args.browser,
        url=args.url,
        headed=args.headed,
    )


if __name__ == "__main__":
    main()
