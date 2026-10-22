"""
JED-M2 Desktop & Browser Sidecar Daemon.
Provides:
- FastAPI / asyncio background server with REST and WebSocket endpoints:
  - POST /task: Submit natural language automation goal.
  - GET /status: Query active step, latest thought rationale, and execution telemetry.
  - POST /abort: Immediate emergency stop releasing peripheral control.
  - GET /stream: Real-time WebSocket streaming of agent thoughts and action primitives.
- Minimal idle footprint (< 100 MB RAM, 0% idle GPU).
- Stateful Session Manager: Maintains layer-wise Mamba SSM states, rolling causal conv buffers,
  and MLA KV-cache across multiple turns without sequence length explosion.
- Model Flexibility: Supports seamless loading of JED-M2-Nano (51M) or JED-M2-Medium (273M).
"""

import asyncio
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from jed_m2.agent import AgentStepOutput, JEDM2Agent, TurnRecord
from jed_m2.config import JEDM2Config, JEDM2MediumConfig
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model
from jed_m2.sidecar.os_controller import WindowsOSController
from jed_m2.sidecar.safety import SafetySupervisor, SafetyViolation


# ==============================================================================
# Pydantic Request Models
# ==============================================================================

class TaskRequest(BaseModel):
    """Payload for submitting an autonomous desktop automation task."""
    goal: str
    max_steps: int = 30
    require_confirmation: bool = True


# ==============================================================================
# Stateful Session Container
# ==============================================================================

@dataclass
class SidecarSession:
    """Maintains stateful recurrent states and telemetry across interaction turns."""

    task_id: str
    goal: str
    max_steps: int = 30
    status: str = "idle"  # idle, running, paused, completed, aborted, error
    step_count: int = 0
    current_thought: str = ""
    current_action: str = ""
    current_coordinates: Tuple[float, float] = (0.5, 0.5)
    current_argument: str = ""
    gate_prob: float = 0.0
    error_message: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Recurrent state buffers persisted across turns for O(1) memory
    recurrent_states: Optional[List[Tuple[Any, Any, Any]]] = None
    current_pos: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts session telemetry to serializable dictionary."""
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "status": self.status,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "current_thought": self.current_thought,
            "current_action": self.current_action,
            "current_coordinates": self.current_coordinates,
            "current_argument": self.current_argument,
            "gate_prob": self.gate_prob,
            "error_message": self.error_message,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            "history_length": len(self.history),
        }


# ==============================================================================
# Sidecar Daemon Service
# ==============================================================================

class SidecarDaemon:
    """
    Asynchronous Local Sidecar Daemon service.
    Orchestrates FastAPI endpoints, background PyTorch inference rollouts,
    per-monitor DPI-aware OS automation, and safety supervisor checks.
    """

    def __init__(
        self,
        model_size: str = "nano",
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        mock_mode: bool = False,
        auto_approve_safety: bool = False,
    ):
        """
        Initializes the sidecar daemon.

        Args:
            model_size: 'nano' (51M) or 'medium' (273M).
            model_path: Optional file path to pretrained model checkpoint.
            device: 'cuda', 'cpu', or None (auto-detect).
            dtype: Tensor computation data type (torch.bfloat16 recommended).
            mock_mode: If True, uses mock OS automation without moving hardware.
            auto_approve_safety: If True, bypasses interactive safety confirmation dialogs.
        """
        self.model_size = model_size.lower()
        self.model_path = model_path
        self.dtype = dtype
        self.mock_mode = mock_mode

        # Determine target device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize OS Controller and Safety Supervisor
        self.os_controller = WindowsOSController(mock_mode=mock_mode)
        self.safety_supervisor = SafetySupervisor(auto_approve=auto_approve_safety)

        # Initialize Model and Agent
        self.config, self.model = self._init_model()
        self.tokenizer = SimpleLAMTokenizer(vocab_size=self.config.vocab_size)
        self.agent = JEDM2Agent(
            model=self.model,
            config=self.config,
            tokenizer=self.tokenizer,
            device=self.device,
            dtype=self.dtype,
        )

        # Session state and synchronization
        self.active_session: Optional[SidecarSession] = None
        self._abort_requested = False
        self._worker_task: Optional[asyncio.Task] = None
        self._subscribers: Set[WebSocket] = set()

        # Build FastAPI application
        self.app = self._build_fastapi_app()

    def _init_model(self) -> Tuple[JEDM2Config, JEDM2Model]:
        """Initializes model architecture according to requested model_size."""
        if self.model_size == "medium":
            config = JEDM2MediumConfig()
        else:
            config = JEDM2Config(model_size="nano")

        model = JEDM2Model(config).to(device=self.device, dtype=self.dtype)
        model.eval()

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict, strict=False)

        return config, model

    def _build_fastapi_app(self) -> FastAPI:
        """Constructs and binds FastAPI endpoints."""
        app = FastAPI(title="JED-M2 Desktop Sidecar Runtime", version="1.0.0")

        @app.post("/task")
        async def submit_task(req: TaskRequest):
            if self.active_session is not None and self.active_session.status == "running":
                raise HTTPException(status_code=409, detail="A task is already actively running")

            session = SidecarSession(
                task_id=str(uuid.uuid4())[:8],
                goal=req.goal,
                max_steps=req.max_steps,
                status="running",
            )
            self.active_session = session
            self._abort_requested = False

            # Launch background worker
            self._worker_task = asyncio.create_task(self._execution_worker(session))

            await self._broadcast({
                "event": "task_started",
                "task_id": session.task_id,
                "goal": session.goal,
            })
            return {"task_id": session.task_id, "status": "started", "goal": session.goal}

        @app.get("/status")
        async def get_status():
            if self.active_session is None:
                return {
                    "status": "idle",
                    "model_size": self.model_size,
                    "device": str(self.device),
                    "active_task": None,
                }
            return {
                "status": self.active_session.status,
                "model_size": self.model_size,
                "device": str(self.device),
                "active_task": self.active_session.to_dict(),
            }

        @app.post("/abort")
        async def abort_task():
            if self.active_session is None or self.active_session.status != "running":
                return {"status": "no_active_task"}

            self._abort_requested = True
            self.active_session.status = "aborted"

            if self._worker_task and not self._worker_task.done():
                self._worker_task.cancel()

            await self._broadcast({
                "event": "aborted",
                "task_id": self.active_session.task_id,
            })
            return {"status": "aborted", "task_id": self.active_session.task_id}

        @app.websocket("/stream")
        async def websocket_stream(websocket: WebSocket):
            await websocket.accept()
            self._subscribers.add(websocket)
            try:
                # Send current initial state
                status_dict = await get_status()
                await websocket.send_json({"event": "connected", "telemetry": status_dict})

                # Keep connection alive listening for ping/pong or client messages
                while True:
                    data = await websocket.receive_text()
            except WebSocketDisconnect:
                self._subscribers.remove(websocket)
            except Exception:
                if websocket in self._subscribers:
                    self._subscribers.remove(websocket)

        return app

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        """Broadcasts event telemetry to all connected WebSocket clients."""
        dead_sockets = set()
        for ws in self._subscribers:
            try:
                await ws.send_json(payload)
            except Exception:
                dead_sockets.add(ws)
        self._subscribers -= dead_sockets

    async def _execution_worker(self, session: SidecarSession) -> None:
        """Asynchronous execution loop driving multi-turn agent interaction steps."""
        try:
            # Reinitialize agent rollout buffers for new session
            self.agent.reset()

            while session.step_count < session.max_steps:
                if self._abort_requested:
                    session.status = "aborted"
                    break

                session.step_count += 1

                # 1. High-Speed Screen Perception
                screen_tensor = await asyncio.to_thread(
                    self.os_controller.capture_screen,
                    target_size=(self.config.image_size, self.config.image_size),
                    dtype=torch.float32,
                    device=self.device,
                )

                # 2. Build observation context text
                if session.step_count == 1:
                    obs_text = f"Goal: {session.goal} | Desktop screen"
                else:
                    obs_text = f"Step {session.step_count} screen observation"

                # 3. Autoregressive System 2 thought & System 1 action decoding via agent.act()
                step_output = await asyncio.to_thread(
                    self.agent.act,
                    obs_text=obs_text,
                    screenshot=screen_tensor,
                )

                session.current_thought = step_output.thought
                session.current_action = step_output.action_name
                session.current_coordinates = step_output.coordinates
                session.current_argument = step_output.argument
                session.gate_prob = step_output.gate_prob

                # 4. Human-in-the-Loop Safety Interception
                authorized, violation = self.safety_supervisor.authorize_action(
                    action_type=step_output.action_type,
                    argument=step_output.argument,
                )

                if not authorized:
                    session.status = "paused"
                    await self._broadcast({
                        "event": "safety_violation",
                        "step": session.step_count,
                        "violation": violation.risk_description if violation else "Denied",
                    })
                    break

                # 5. Dispatch Action Primitive to Windows OS
                dispatch_res = await asyncio.to_thread(
                    self.os_controller.dispatch_action,
                    action_type=step_output.action_type,
                    coordinates=step_output.coordinates,
                    argument=step_output.argument,
                )

                # Record turn history
                turn_dict = {
                    "step": session.step_count,
                    "thought": step_output.thought,
                    "action": step_output.action_name,
                    "coordinates": step_output.coordinates,
                    "argument": step_output.argument,
                    "gate_prob": step_output.gate_prob,
                    "dispatch": dispatch_res,
                }
                session.history.append(turn_dict)

                # 6. Stream Telemetry Event to WebSocket Clients
                await self._broadcast({
                    "event": "step",
                    "step": session.step_count,
                    "thought": step_output.thought,
                    "action": step_output.action_name,
                    "coordinates": step_output.coordinates,
                    "argument": step_output.argument,
                    "gate_prob": round(step_output.gate_prob, 3),
                })

                if step_output.action_type == ActionType.FINISH:
                    session.status = "completed"
                    break

                # Inter-turn delay
                await asyncio.sleep(0.05)

            if session.status == "running":
                session.status = "completed"

        except asyncio.CancelledError:
            session.status = "aborted"
        except Exception as e:
            session.status = "error"
            session.error_message = str(e)
            await self._broadcast({"event": "error", "message": str(e)})
        finally:
            await self._broadcast({
                "event": "task_finished",
                "task_id": session.task_id,
                "status": session.status,
            })
