"""
JED-M2 Core Types and Utilities.
Provides unified foundation dataclasses and observation feature encoders.
"""

from jed_m2.core.features import compute_obs_features
from jed_m2.core.types import (
    InteractiveElement,
    InteractionTurn,
    PreferenceTurn,
    RealWebInteractionTurn,
    UIElement,
)

__all__ = [
    "UIElement",
    "InteractiveElement",
    "InteractionTurn",
    "RealWebInteractionTurn",
    "PreferenceTurn",
    "compute_obs_features",
]
