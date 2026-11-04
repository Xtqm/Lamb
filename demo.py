"""
JED-M2 Native Large Action Model - Interactive Edge Demonstration CLI.
Runs real-time on-device inference for web browser automation.

Usage:
    python demo.py
    python demo.py --interactive
    python demo.py --preset amazon
    python demo.py --checkpoint checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt
"""

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple
import torch

from jed_m2.agent import JEDM2Agent
from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.config import JEDM2Config
from jed_m2.core.types import ActionType, UIElement
from jed_m2.model import JEDM2Model, adapt_state_dict_for_action_classes

# Realistic web interaction presets for instant evaluation
PRESETS = {
    "amazon": {
        "title": "Amazon Product Search",
        "goal": "Search for noise-cancelling headphones and select the top customer rated model",
        "obs": '<header id="navbar"><input type="text" id="twotabsearchtextbox" placeholder="Search Amazon" value=""><input type="submit" id="nav-search-submit-button" value="Go"></header>',
        "elements": [
            UIElement(element_id=1, tag="input", role="search_input", text="Search Amazon", bbox=(0.20, 0.04, 0.75, 0.08), is_input=True),
            UIElement(element_id=2, tag="input", role="submit_button", text="Go", bbox=(0.76, 0.04, 0.82, 0.08), is_clickable=True),
            UIElement(element_id=3, tag="a", role="nav_link", text="Today's Deals", bbox=(0.10, 0.10, 0.22, 0.13), is_clickable=True),
            UIElement(element_id=4, tag="a", role="nav_link", text="Customer Service", bbox=(0.24, 0.10, 0.38, 0.13), is_clickable=True),
        ],
    },
    "delta": {
        "title": "Delta Airlines Flight Booking",
        "goal": "Select departure date for flight from JFK to SFO on December 15",
        "obs": '<div id="booking-widget"><button id="from-airport">JFK (New York)</button><button id="to-airport">SFO (San Francisco)</button><input id="depart-date" placeholder="Depart Date" value=""><button id="find-flights">Search Flights</button></div>',
        "elements": [
            UIElement(element_id=1, tag="button", role="origin_button", text="JFK (New York)", bbox=(0.15, 0.25, 0.35, 0.32), is_clickable=True),
            UIElement(element_id=2, tag="button", role="destination_button", text="SFO (San Francisco)", bbox=(0.38, 0.25, 0.58, 0.32), is_clickable=True),
            UIElement(element_id=3, tag="input", role="date_input", text="Depart Date", bbox=(0.60, 0.25, 0.78, 0.32), is_input=True),
            UIElement(element_id=4, tag="button", role="submit_button", text="Search Flights", bbox=(0.80, 0.25, 0.95, 0.32), is_clickable=True),
        ],
    },
    "boardgamegeek": {
        "title": "BoardGameGeek Search & Filter",
        "goal": "Filter games by top ranked strategy games with rating above 8.0",
        "obs": '<nav class="bgg-nav"><a href="/browse">Browse Games</a><a href="/rankings">Top Rankings</a><div class="filter-dropdown">Strategy Games</div></nav>',
        "elements": [
            UIElement(element_id=1, tag="a", role="nav_link", text="Browse Games", bbox=(0.12, 0.08, 0.28, 0.12), is_clickable=True),
            UIElement(element_id=2, tag="a", role="nav_link", text="Top Rankings", bbox=(0.30, 0.08, 0.45, 0.12), is_clickable=True),
            UIElement(element_id=3, tag="div", role="dropdown", text="Strategy Games", bbox=(0.48, 0.08, 0.65, 0.12), is_clickable=True),
        ],
    },
}


def load_agent(
    checkpoint_path: Optional[str] = None,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> JEDM2Agent:
    """Loads JEDM2Model and initializes the runtime agent."""
    device = torch.device(device_str)
    dtype = torch.bfloat16 if "cuda" in device_str and torch.cuda.is_available() else torch.float32

    config = JEDM2Config()
    model = JEDM2Model(config)

    # Checkpoint resolution
    chosen_path = None
    candidates = [
        checkpoint_path,
        "checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt",
        "checkpoints_rl/jed_m2_nano_grpo_final.pt",
        "checkpoints/jed_m2_sft_base.pt",
    ]
    for c in candidates:
        if c and os.path.exists(c):
            chosen_path = c
            break

    if chosen_path:
        print(f"[*] Loading model weights from: {chosen_path}")
        state = torch.load(chosen_path, map_location="cpu", weights_only=False)
        state_dict = state.get("model_state_dict", state)
        state_dict = adapt_state_dict_for_action_classes(state_dict, target_classes=7)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[!] No checkpoint found. Initializing with raw architecture weights.")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    tokenizer = ActionBPETokenizer(vocab_size=config.vocab_size)
    agent = JEDM2Agent(
        model=model,
        config=config,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        snap_coordinates_to_candidates=True,
        max_snap_distance=0.20,
    )
    return agent


def render_step_result(
    step_num: int,
    goal: str,
    obs: str,
    output: Any,
    latency_ms: float,
    vram_mb: float,
    elements: List[UIElement],
):
    """Prints a beautiful 7-bit ASCII decision breakdown."""
    border = "=" * 80
    sub_border = "-" * 80
    print("\n" + border)
    print(f" JED-M2 STEP {step_num} DECISION SUMMARY ({latency_ms:.1f} ms | VRAM: {vram_mb:.1f} MB)")
    print(border)
    print(f"[*] High-Level Goal: {goal}")
    print(f"[*] Raw Observation: {obs[:70]}..." if len(obs) > 70 else f"[*] Raw Observation: {obs}")
    print(sub_border)
    print(f"  [System 2 Rationale]  : {output.thought if output.thought else '(Immediate Reflexive Act)'}")
    print(f"  [Action Primitive]   : ActionType.{output.action_type.name} (Class ID: {output.action_type.value})")
    print(f"  [Screen Coordinates] : (x={output.coordinates[0]:.4f}, y={output.coordinates[1]:.4f}) [Normalized in [0, 1]^2]")
    if output.argument:
        print(f"  [Typed Argument]     : \"{output.argument}\"")

    # Candidate snapping details
    snapped = getattr(output, "snapped_to_element", None)
    if snapped:
        print(f"  [Candidate Snapping] : Snapped to UI element '{snapped.text}' ({snapped.tag}) bbox: {snapped.bbox}")
    elif elements:
        print(f"  [Candidate Snapping] : Grounded across {len(elements)} page interactive candidates")
    print(border + "\n")


def run_preset(agent: JEDM2Agent, preset_key: str):
    """Executes inference on a realistic pre-configured scenario."""
    preset = PRESETS.get(preset_key, PRESETS["amazon"])
    print(f"\n[>>>] Running Preset: {preset['title']}")

    agent.reset()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(agent.device)

    t0 = time.perf_counter()
    output = agent.act(
        obs_text=f"Goal: {preset['goal']} | Context: {preset['obs']}",
        candidate_elements=preset["elements"],
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    vram_mb = torch.cuda.max_memory_allocated(agent.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

    render_step_result(
        step_num=1,
        goal=preset["goal"],
        obs=preset["obs"],
        output=output,
        latency_ms=latency_ms,
        vram_mb=vram_mb,
        elements=preset["elements"],
    )


def run_interactive(agent: JEDM2Agent):
    """Interactive loop allowing arbitrary user inputs."""
    print("\n" + "=" * 80)
    print(" JED-M2 INTERACTIVE REPL MODE (Type 'exit' to quit, 'reset' to clear context)")
    print("=" * 80)

    step = 1
    while True:
        try:
            goal = input("\n[Enter Goal] > ").strip()
            if not goal:
                continue
            if goal.lower() in ("exit", "quit", "q"):
                print("Exiting JED-M2 REPL.")
                break
            if goal.lower() == "reset":
                agent.reset()
                print("[*] Recurrent SSM and MLA context cleared.")
                step = 1
                continue

            obs = input("[Enter HTML / DOM Text] > ").strip()
            if not obs:
                obs = '<button id="submit">Search</button>'

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(agent.device)

            t0 = time.perf_counter()
            output = agent.act(obs_text=f"Goal: {goal} | State: {obs}")
            lat_ms = (time.perf_counter() - t0) * 1000.0
            vram_mb = torch.cuda.max_memory_allocated(agent.device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

            render_step_result(
                step_num=step,
                goal=goal,
                obs=obs,
                output=output,
                latency_ms=lat_ms,
                vram_mb=vram_mb,
                elements=[],
            )
            step += 1
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="JED-M2 Large Action Model Interactive Demo")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--preset", type=str, default="amazon", choices=list(PRESETS.keys()), help="Preset scenario to run")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive text REPL")
    args = parser.parse_args()

    print("=" * 80)
    print("       JED-M2: Sub-100M Native Browser Action Model (Edge Runtime)")
    print("=" * 80)

    agent = load_agent(checkpoint_path=args.checkpoint, device_str=args.device)

    if args.interactive:
        run_interactive(agent)
    else:
        run_preset(agent, args.preset)
        print(f"[Tip] Run with '--preset delta' or '--interactive' to test other scenarios.")


if __name__ == "__main__":
    main()
