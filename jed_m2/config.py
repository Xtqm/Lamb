"""
JED-M2 Global Configuration Dataclass.
Calibrated for an ~45M parameter budget Large Action Model (LAM).
"""

from dataclasses import dataclass


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

    # Dual-System Action Head parameters
    num_action_types: int = 6  # CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH
    action_dropout: float = 0.1
    action_gate_threshold: float = 0.5

    @property
    def qk_head_dim(self) -> int:
        """Combined query/key head dimension: d_h^{NOPE} + d_h^R."""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mamba_inner_dim(self) -> int:
        """Expanded inner dimension for Mamba SSM block: expand * hidden_dim."""
        return self.mamba_expand * self.hidden_dim
