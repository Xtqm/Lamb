"""
JED-M2: Native Large Action Model (LAM) Architecture in PyTorch.
"""

from jed_m2.config import JEDM2Config
from jed_m2.dataset import (
    AgentTrajectoryDataset,
    InteractionTurn,
    SimpleLAMTokenizer,
    SyntheticTrajectoryGenerator,
    TrajectoryBatch,
    TrajectoryCollator,
    TrajectorySample,
    UIElement,
)
from jed_m2.heads import (
    ActionGate,
    ActionOutput,
    ActionType,
    ActionTypeHead,
    ArgumentHead,
    CoordinateHead,
    NativeActionHead,
)
from jed_m2.jepa import (
    EMATargetEncoder,
    JEPAWorldModel,
    LatentPredictor,
    LatentStateEncoder,
)
from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
from jed_m2.mamba import MambaBlock
from jed_m2.mla import MultiHeadLatentAttention, RMSNorm, RotaryEmbedding
from jed_m2.model import JEDM2Block, JEDM2Model, JEDM2Output
from jed_m2.moe import GELUFeedForward, SparseTop2MoE, SwiGLUFeedForward

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
    "ActionGate",
    "ActionTypeHead",
    "CoordinateHead",
    "ArgumentHead",
    "NativeActionHead",
    "ActionType",
    "ActionOutput",
    "JEDM2Block",
    "JEDM2Model",
    "JEDM2Output",
    "UIElement",
    "InteractionTurn",
    "TrajectorySample",
    "SimpleLAMTokenizer",
    "SyntheticTrajectoryGenerator",
    "AgentTrajectoryDataset",
    "TrajectoryBatch",
    "TrajectoryCollator",
    "JEDM2LossConfig",
    "JEDM2CompositeLoss",
]
