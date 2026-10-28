"""
JED-M2 Global Configuration Dataclass.
Calibrated for an ~45M parameter budget Large Action Model (LAM).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class JEDM2Config:
    """Configuration for the JED-M2 Large Action Model architecture."""

    # Transformer backbone dimensions
    hidden_dim: int = 512
    num_heads: int = 8
    max_seq_len: int = 2048
    vocab_size: int = 8192  # Compact discrete action/text vocabulary
    layer_norm_eps: float = 1e-5

    # Multi-Head Latent Attention (MLA) parameters
    q_lora_rank: int = 256  # Query compression rank (c_Q)
    kv_lora_rank: int = 128  # Key-Value compression rank (c_KV)
    qk_rope_head_dim: int = 32  # Decoupled RoPE dimension (d_h^R)
    qk_nope_head_dim: int = 32  # Non-positional dimension (d_h^{NOPE})
    v_head_dim: int = 64  # Value head dimension (d_h^V)
    rope_theta: float = 10000.0  # Base frequency for RoPE

    # Sparse Mixture of Experts (MoE) parameters
    num_experts: int = 8
    top_k: int = 2
    moe_intermediate_dim: int = 1024
    aux_loss_coef: float = 0.01
    activation: str = "swiglu"  # "swiglu" or "gelu"

    # Mamba Selective State Space Model (SSM) parameters
    mamba_d_state: int = 16  # SSM state expansion rank (N)
    mamba_d_conv: int = 4  # 1D causal convolution kernel size
    mamba_expand: int = 2  # Internal dimension expansion factor (E)
    mamba_dt_rank: int = 32  # Rank of delta projection (dt_rank)

    # JEPA World Model parameters
    jepa_latent_dim: int = 256  # Abstract latent state dimension
    jepa_ema_decay: float = 0.99  # Momentum factor for target encoder
    jepa_loss_cos_weight: float = 0.5  # Weight of cosine distance in composite loss
    jepa_action_dim: int = 64  # Action embedding dimension for predictor conditioning

    # Backbone Stack parameters
    num_layers: int = 3  # Stacked JEDM2Block recurrent layers

    # Architectural Baseline & Ablation Toggles (Phase 10)
    attn_type: Literal["mla", "mha"] = "mla"  # "mla" (Multi-Head Latent Attention) or "mha" (Standard MHA)
    ffn_type: Literal["moe", "dense"] = "moe"  # "moe" (Sparse Top-2 MoE) or "dense" (Dense SwiGLU FFN)
    use_mamba: bool = True  # Enable/disable Mamba Selective State Space Model recurrence
    use_jepa: bool = True  # Enable/disable JEPA Latent World Model
    dense_intermediate_dim: int = 1024  # Intermediate hidden dim for Dense SwiGLU FFN
    mha_head_dim: int = 64  # Per-head dimension for Standard MHA (default: hidden_dim // num_heads = 64)
    model_size: Literal["nano", "medium"] = "nano"  # Model scale tier (Phase 11)

    # Dual-System Action Head parameters
    num_action_types: int = 7  # CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH, SELECT
    action_dropout: float = 0.1
    action_gate_threshold: float = 0.5

    # Vision Patch Encoder parameters (Phase 6D)
    image_size: int = 224  # Standard screenshot input resolution (e.g. 224x224 or 448x448)
    patch_size: int = 16  # Non-overlapping patch spatial dimension (16x16)
    num_channels: int = 3  # RGB image channels
    visual_grounding_heads: int = 4  # Number of attention heads for visual grounding in coordinate head

    @property
    def num_patches(self) -> int:
        """Number of visual patches: (image_size // patch_size) ** 2."""
        return (self.image_size // self.patch_size) ** 2

    @property
    def qk_head_dim(self) -> int:
        """Combined query/key head dimension: d_h^{NOPE} + d_h^R."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mamba_inner_dim(self) -> int:
        """Expanded inner dimension for Mamba SSM block: expand * hidden_dim."""
        return self.mamba_expand * self.hidden_dim


def JEDM2MediumConfig(max_seq_len: int = 2048, **kwargs) -> JEDM2Config:
    """
    Factory constructing JED-M2-Medium configuration.
    Calibrated for ~273M total parameters / ~75M active parameters per token
    across 6 recurrent layers, 12 heads MLA, and 16 SwiGLU experts on RTX 5060.
    """
    params = dict(
        model_size="medium",
        hidden_dim=768,
        num_heads=12,
        max_seq_len=max_seq_len,
        vocab_size=8192,
        q_lora_rank=384,
        kv_lora_rank=192,
        qk_rope_head_dim=64,
        qk_nope_head_dim=32,
        v_head_dim=64,
        rope_theta=10000.0,
        num_experts=16,
        top_k=2,
        moe_intermediate_dim=1024,
        aux_loss_coef=0.01,
        activation="swiglu",
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=48,
        jepa_latent_dim=384,
        jepa_ema_decay=0.99,
        jepa_loss_cos_weight=0.5,
        jepa_action_dim=64,
        num_layers=6,
        attn_type="mla",
        ffn_type="moe",
        use_mamba=True,
        use_jepa=True,
        dense_intermediate_dim=1024,
        mha_head_dim=64,
        num_action_types=7,
        action_dropout=0.1,
        action_gate_threshold=0.5,
        image_size=224,
        patch_size=16,
        num_channels=3,
        visual_grounding_heads=4,
    )
    params.update(kwargs)
    return JEDM2Config(**params)

