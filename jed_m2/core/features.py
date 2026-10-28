"""
JED-M2 Core Semantic Observation Feature Extractor.
Provides deterministic MD5 word hashing and spatial UI element center projection for JEPA world model conditioning.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import hashlib
from typing import Any, List, Optional, Sequence
import torch

from jed_m2.core.types import UIElement


def compute_obs_features(
    obs_text: str,
    elements: Sequence[Any],
    hidden_dim: int,
) -> torch.Tensor:
    """
    Computes a deterministic semantic observation embedding vector [hidden_dim]
    combining MD5 word hashing with normalized spatial UI element center projections.
    Used for JEPA world model state transitions and conditioning.

    Args:
        obs_text: Raw observation text string.
        elements: Sequence of UI elements providing normalized .center coordinates.
        hidden_dim: Target feature dimension matching model hidden_dim.

    Returns:
        L2-normalized float32 tensor of shape [hidden_dim].
    """
    obs_words = obs_text.split()
    obs_features = torch.zeros(hidden_dim, dtype=torch.float32)

    # Word hashing projection with position-decay weighting
    for w_idx, word in enumerate(obs_words):
        w_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
        dim_idx = w_hash % hidden_dim
        weight = 1.0 / (1.0 + (w_idx % 32) * 0.05)
        obs_features[dim_idx] += weight

    # Spatial pointer coordinates projection
    for el in elements:
        if not getattr(el, "has_bbox", True):
            continue
        center = getattr(el, "center", None)
        if center is not None:
            cx_dim = int(center[0] * (hidden_dim - 1))
            cy_dim = int(center[1] * (hidden_dim - 1))
            cx_dim = max(0, min(hidden_dim - 1, cx_dim))
            cy_dim = max(0, min(hidden_dim - 1, cy_dim))
            obs_features[cx_dim] += 0.5
            obs_features[cy_dim] += 0.5

    # L2 Normalization
    f_norm = obs_features.norm(p=2)
    if f_norm > 1e-6:
        obs_features = obs_features / f_norm

    return obs_features
