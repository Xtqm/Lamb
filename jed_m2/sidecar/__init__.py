"""
JED-M2 Desktop & Browser Sidecar Runtime Package.
Provides:
- WindowsOSController: Pure Python DPI-aware OS automation controller.
- SafetySupervisor: Rule-based human-in-the-loop action interceptor.
- SidecarDaemon: Asynchronous FastAPI / asyncio background daemon.
- SidecarSession: Multi-turn stateful session manager for recurrent rollouts.
"""

from jed_m2.sidecar.os_controller import WindowsOSController
from jed_m2.sidecar.safety import SafetySupervisor
from jed_m2.sidecar.daemon import SidecarDaemon, SidecarSession

__all__ = [
    "WindowsOSController",
    "SafetySupervisor",
    "SidecarDaemon",
    "SidecarSession",
]
