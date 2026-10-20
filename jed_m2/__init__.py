"""
JED-M2: Native Large Action Model (LAM) Architecture in PyTorch.
"""

from jed_m2.agent import (
    AgentStepOutput,
    JEDM2Agent,
    TurnRecord,
)
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
from jed_m2.environment import (
    EnvironmentState,
    ScenarioDefinition,
    StepResult,
    WebOSEnvironment,
    get_default_scenarios,
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
from jed_m2.rl import (
    DPOConfig,
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
    calculate_dpo_loss,
    calculate_grpo_loss,
    compute_token_log_probs,
)
from jed_m2.rl_pipeline import (
    RolloutTrajectory,
    RolloutTurn,
    TrajectoryRolloutSampler,
)
from jed_m2.trainer import (
    JEDM2Trainer,
    JEDM2TrainerConfig,
    build_cosine_scheduler,
    build_optimizer,
)
from jed_m2.vision import (
    VisionPatchEncoder,
    get_2d_sincos_pos_embed,
)

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
    "VisionPatchEncoder",
    "get_2d_sincos_pos_embed",
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
    "JEDM2TrainerConfig",
    "JEDM2Trainer",
    "build_optimizer",
    "build_cosine_scheduler",
    "DPOConfig",
    "GRPOConfig",
    "DPOTrainer",
    "GRPOTrainer",
    "calculate_dpo_loss",
    "calculate_grpo_loss",
    "compute_token_log_probs",
    "WebOSEnvironment",
    "EnvironmentState",
    "StepResult",
    "ScenarioDefinition",
    "get_default_scenarios",
    "JEDM2Agent",
    "AgentStepOutput",
    "TurnRecord",
    "TrajectoryRolloutSampler",
    "RolloutTrajectory",
    "RolloutTurn",
]
