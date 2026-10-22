"""
JED-M2 Desktop & Browser Sidecar Agent CLI & ASCII Terminal HUD.
Provides unified commands:
- daemon: Starts background FastAPI & WebSocket sidecar service.
- exec: Runs a one-shot natural language automation task with live terminal HUD.
- monitor: Attaches to a running daemon's WebSocket stream with real-time ASCII HUD.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import torch
import uvicorn

from jed_m2.agent import AgentStepOutput
from jed_m2.heads import ActionType
from jed_m2.sidecar.daemon import SidecarDaemon, SidecarSession
from jed_m2.sidecar.os_controller import WindowsOSController
from jed_m2.sidecar.safety import SafetySupervisor


# ==============================================================================
# Real-Time ASCII Terminal HUD Renderer
# ==============================================================================

def render_ascii_screen_overlay(
    coords: Tuple[float, float],
    width: int = 34,
    height: int = 10,
) -> str:
    """
    Renders an ASCII representation of the 2D desktop display
    with target crosshair overlay at normalized coordinates (x, y).
    """
    cx, cy = coords
    grid_x = min(width - 3, max(0, int(round(cx * (width - 3)))))
    grid_y = min(height - 1, max(0, int(round(cy * (height - 1)))))

    lines = []
    lines.append("  +" + "-" * width + "+")
    for r in range(height):
        if r == grid_y:
            # Place target marker
            left_pad = " " * grid_x
            marker = "(+)"
            right_pad = " " * max(0, width - grid_x - len(marker))
            row_str = left_pad + marker + right_pad
            lines.append("  |" + row_str[:width] + "|")
        else:
            lines.append("  |" + " " * width + "|")
    lines.append("  +" + "-" * width + "+")
    return "\n".join(lines)


def render_hud(
    step: int,
    max_steps: int,
    thought: str,
    action_name: str,
    coords: Tuple[float, float],
    argument: str,
    gate_prob: float,
    status: str = "RUNNING",
    elapsed_s: float = 0.0,
    model_size: str = "nano",
) -> None:
    """
    Renders the interactive ASCII Terminal HUD.
    """
    # ActionGate Confidence Bar: pure ASCII [#####-----]
    gate_pct = int(round(gate_prob * 10))
    bar = "#" * min(10, max(0, gate_pct)) + "-" * max(0, 10 - gate_pct)

    cx, cy = coords
    arg_repr = f"'{argument}'" if argument else "<none>"

    hud = []
    hud.append("\n" + "=" * 68)
    hud.append(f"  JED-M2 DESKTOP SIDECAR RUNTIME  |  MODEL: {model_size.upper()}  |  STATUS: {status.upper()}")
    hud.append(f"  Step: [{step}/{max_steps}]  |  Elapsed: {elapsed_s:.1f}s  |  Gate: [{bar}] {gate_prob:.3f}")
    hud.append("-" * 68)
    hud.append("  [SYSTEM 2 LATENT RATIONALE]")
    # Wrap thought lines neatly
    thought_wrapped = thought.strip() if thought else "Observing screen environment..."
    hud.append(f"  {thought_wrapped}")
    hud.append("-" * 68)
    hud.append("  [SYSTEM 1 ACTION COMMITMENT & COORDINATES]")
    hud.append(f"  Action Primitive : {action_name}")
    hud.append(f"  Target Coordinates: ({cx:.3f}, {cy:.3f})")
    hud.append(f"  Action Argument  : {arg_repr}")
    hud.append("-" * 68)
    hud.append("  [2D SCREEN TARGET OVERLAY]")
    hud.append(render_ascii_screen_overlay(coords, width=34, height=8))
    hud.append("=" * 68)

    print("\n".join(hud), flush=True)


# ==============================================================================
# CLI Commands
# ==============================================================================

def cmd_daemon(args: argparse.Namespace) -> None:
    """Starts the background FastAPI / WebSocket sidecar daemon."""
    print("=" * 68)
    print(f"Starting JED-M2 Sidecar Daemon on {args.host}:{args.port}")
    print(f"Model Size: {args.model_size.upper()} | Device: {args.device} | Mock Mode: {args.mock}")
    print("=" * 68)

    daemon = SidecarDaemon(
        model_size=args.model_size,
        model_path=args.model_path,
        device=args.device,
        mock_mode=args.mock,
        auto_approve_safety=args.auto_approve,
    )
    uvicorn.run(daemon.app, host=args.host, port=args.port, log_level="info")


def cmd_exec(args: argparse.Namespace) -> None:
    """Executes a one-shot natural language automation goal directly."""
    print("=" * 68)
    print(f"Executing Goal: '{args.goal}'")
    print(f"Model Size: {args.model_size.upper()} | Device: {args.device} | Max Steps: {args.max_steps}")
    print("=" * 68)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os_controller = WindowsOSController(mock_mode=args.mock)
    safety_supervisor = SafetySupervisor(auto_approve=args.auto_approve)

    from jed_m2.config import JEDM2Config, JEDM2MediumConfig
    from jed_m2.dataset import SimpleLAMTokenizer
    from jed_m2.model import JEDM2Model
    from jed_m2.agent import JEDM2Agent

    if args.model_size.lower() == "medium":
        config = JEDM2MediumConfig()
    else:
        config = JEDM2Config(model_size="nano")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = JEDM2Model(config).to(device=device, dtype=dtype)
    model.eval()

    if args.model_path and os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)

    tokenizer = SimpleLAMTokenizer(config.vocab_size)
    agent = JEDM2Agent(model=model, config=config, tokenizer=tokenizer, device=device, dtype=dtype)
    agent.reset()

    start_time = time.time()
    for step in range(1, args.max_steps + 1):
        elapsed = time.time() - start_time
        # 1. Screen Perception
        screen_tensor = os_controller.capture_screen(
            target_size=(config.image_size, config.image_size),
            dtype=torch.float32,
            device=device,
        )

        obs_text = f"Goal: {args.goal}" if step == 1 else f"Step {step} screen observation"

        # 2. Agent Decision Step
        step_out: AgentStepOutput = agent.act(obs_text=obs_text, screenshot=screen_tensor)

        # 3. Render Terminal HUD
        render_hud(
            step=step,
            max_steps=args.max_steps,
            thought=step_out.thought,
            action_name=step_out.action_name,
            coords=step_out.coordinates,
            argument=step_out.argument,
            gate_prob=step_out.gate_prob,
            status="RUNNING",
            elapsed_s=elapsed,
            model_size=args.model_size,
        )

        # 4. Safety Check
        authorized, violation = safety_supervisor.authorize_action(
            action_type=step_out.action_type,
            argument=step_out.argument,
        )
        if not authorized:
            print(f"\n[!] Execution paused by safety supervisor: {violation.risk_description}")
            break

        # 5. Dispatch Action
        os_controller.dispatch_action(
            action_type=step_out.action_type,
            coordinates=step_out.coordinates,
            argument=step_out.argument,
        )

        if step_out.action_type == ActionType.FINISH:
            print(f"\n[OK] Task finished successfully in {step} steps ({elapsed:.1f}s).")
            break

        time.sleep(0.1)


def cmd_monitor(args: argparse.Namespace) -> None:
    """Attaches to a running daemon via WebSocket and renders live telemetry HUD."""
    import websockets

    uri = f"ws://{args.host}:{args.port}/stream"
    print(f"Connecting to live sidecar stream at {uri} ... (Ctrl+C to stop)")

    async def _listen():
        async with websockets.connect(uri) as ws:
            start_time = time.time()
            async for message in ws:
                data = json.loads(message)
                event = data.get("event")
                elapsed = time.time() - start_time

                if event == "step":
                    render_hud(
                        step=data.get("step", 1),
                        max_steps=data.get("max_steps", 30),
                        thought=data.get("thought", ""),
                        action_name=data.get("action", ""),
                        coords=tuple(data.get("coordinates", (0.5, 0.5))),
                        argument=data.get("argument", ""),
                        gate_prob=data.get("gate_prob", 0.0),
                        status="RUNNING",
                        elapsed_s=elapsed,
                    )
                elif event == "safety_violation":
                    print(f"\n[!] Safety Interception Alert: {data.get('violation')}")
                elif event == "task_finished":
                    print(f"\n[OK] Task {data.get('task_id')} finished with status: {data.get('status')}")
                elif event == "aborted":
                    print(f"\n[!] Emergency abort triggered for task: {data.get('task_id')}")

    try:
        asyncio.run(_listen())
    except KeyboardInterrupt:
        print("\nDisconnected from sidecar monitor.")


# ==============================================================================
# Argument Parsing Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="JED-M2 Desktop & Browser Sidecar Runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Daemon Command
    daemon_p = subparsers.add_parser("daemon", help="Start background sidecar service")
    daemon_p.add_argument("--host", default="127.0.0.1", help="Host bind address")
    daemon_p.add_argument("--port", type=int, default=8765, help="Port to listen on")
    daemon_p.add_argument("--model_size", choices=["nano", "medium"], default="nano", help="Model scale tier")
    daemon_p.add_argument("--model_path", default=None, help="Path to checkpoint weights")
    daemon_p.add_argument("--device", default=None, help="Device (cuda or cpu)")
    daemon_p.add_argument("--mock", action="store_true", help="Run OS automation in mock mode")
    daemon_p.add_argument("--auto_approve", action="store_true", help="Auto-approve safety checks")

    # 2. Exec Command
    exec_p = subparsers.add_parser("exec", help="Run one-shot natural language automation goal")
    exec_p.add_argument("goal", type=str, help="Natural language automation task goal")
    exec_p.add_argument("--model_size", choices=["nano", "medium"], default="nano", help="Model scale tier")
    exec_p.add_argument("--model_path", default=None, help="Path to checkpoint weights")
    exec_p.add_argument("--device", default=None, help="Device (cuda or cpu)")
    exec_p.add_argument("--max_steps", type=int, default=10, help="Maximum steps")
    exec_p.add_argument("--mock", action="store_true", help="Run in mock mode")
    exec_p.add_argument("--auto_approve", action="store_true", help="Auto-approve safety checks")

    # 3. Monitor Command
    monitor_p = subparsers.add_parser("monitor", help="Attach to running daemon with ASCII HUD")
    monitor_p.add_argument("--host", default="127.0.0.1", help="Daemon host")
    monitor_p.add_argument("--port", type=int, default=8765, help="Daemon port")

    args = parser.parse_args()

    if args.command == "daemon":
        cmd_daemon(args)
    elif args.command == "exec":
        cmd_exec(args)
    elif args.command == "monitor":
        cmd_monitor(args)


if __name__ == "__main__":
    main()
