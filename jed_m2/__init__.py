"""
JED-M2: Native Large Action Model (LAM) Architecture in PyTorch.
"""

from jed_m2.config import JEDM2Config
from jed_m2.mla import MultiHeadLatentAttention, RotaryEmbedding, RMSNorm
from jed_m2.moe import SparseTop2MoE, SwiGLUFeedForward, GELUFeedForward
from jed_m2.mamba import MambaBlock
from jed_m2.jepa import LatentStateEncoder, EMATargetEncoder, LatentPredictor, JEPAWorldModel

__all__ = [
    "JEDM2Config",
    "MultiHeadLatentAttention",
    "RotaryEmbedding",
    "RMSNorm",
    "SparseTop2MoE",
    "SwiGLUFeedForward",
    "GELUFeedForward",
    "MambaBlock",
    "LatentStateEncoder",
    "EMATargetEncoder",
    "LatentPredictor",
    "JEPAWorldModel",
]

