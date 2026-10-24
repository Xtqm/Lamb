"""
JED-M2 Headless Interactive Browser Sandbox & Programmatic Task Oracles (Track B).
Compatibility Façade re-exporting canonical implementations from `jed_m2.browser`.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

from jed_m2.browser import (
    PlaywrightEnvironment,
    PlaywrightStepResult,
    PlaywrightWebEnvironment,
    WebTaskOracle,
    get_default_web_oracles,
    get_fixture_path,
)

__all__ = [
    "WebTaskOracle",
    "get_default_web_oracles",
    "get_fixture_path",
    "PlaywrightWebEnvironment",
    "PlaywrightEnvironment",
    "PlaywrightStepResult",
]
