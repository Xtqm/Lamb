"""
Architectural Baselines for JED-M2 Ablation Study (Phase 10).
Provides standard architectural reference components:
1. StandardMHA: Standard Multi-Head Attention with rotary position embeddings (RoPE),
   scaled dot-product attention, and uncompressed per-head recurrent KV-caching.
2. DenseSwiGLUFFN: Dense gated SwiGLU feed-forward network.
3. Parameter-matched factory functions calibrated to ~50.0M parameters:
   - Baseline A (Pure Transformer): Standard MHA + Dense SwiGLU FFN (4 layers, ~50.5M params)
   - Baseline B (Hybrid Attention-SSM): Mamba SSM + MLA + Dense SwiGLU FFN (3 layers, ~50.2M params)
   - Baseline C (JED-M2 without JEPA): Mamba SSM + MLA + Sparse Top-2 MoE (3 layers, ~50.9M params)
   - Baseline D (Full JED-M2 with Connected JEPA): Mamba SSM + MLA + Sparse MoE + Connected JEPA (3 layers, ~53.0M params)
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.mla import RotaryEmbedding, RMSNorm


class DenseSwiGLUFFN(nn.Module):
    """
    Dense SwiGLU Multi-Layer Perceptron (Linear -> SiLU(Gate) * Up -> Linear).
    Standard feed-forward network used in modern dense LLMs (e.g. LLaMA).
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        # Gate and Up projections
        self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        # Down projection
        self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Dense SwiGLU FFN.
        Supports inputs of shape [B, S, D] or [B, D].
        """
        # [B, S, D] -> [B, S, intermediate_dim]
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        # [B, S, intermediate_dim] -> [B, S, D]
        return self.w_down(gate * up)


class StandardMHA(nn.Module):
    """
    Standard Multi-Head Attention (MHA) with Rotary Position Embeddings (RoPE).
    Maintains standard uncompressed Key-Value caches during recurrent generation.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = getattr(config, "mha_head_dim", self.hidden_dim // self.num_heads)
        self.all_head_dim = self.num_heads * self.head_dim

        # Standard Q, K, V linear projections
        self.q_proj = nn.Linear(self.hidden_dim, self.all_head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.all_head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.all_head_dim, bias=False)
        self.out_proj = nn.Linear(self.all_head_dim, self.hidden_dim, bias=False)

        # Decoupled RoPE applied to full head dimension
        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sequence forward pass for Standard Multi-Head Attention.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim].
            attention_mask: Optional attention mask [batch_size, 1, seq_len, seq_len] or [seq_len, seq_len].

        Returns:
            Output tensor [batch_size, seq_len, hidden_dim].
        """
        B, S, _ = x.shape

        # [B, S, D] -> [B, S, H, D_h]
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim)

        # Apply RoPE positional embeddings
        q = self.rope(q, seq_len=S, start_pos=0)
        k = self.rope(k, seq_len=S, start_pos=0)

        # Transpose for scaled dot-product attention: [B, H, S, D_h]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        is_causal = attention_mask is None and S > 1
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=is_causal
        )

        # [B, H, S, D_h] -> [B, S, H, D_h] -> [B, S, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.all_head_dim)
        return self.out_proj(attn_out)

    def step(
        self,
        x_t: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental single-step recurrent rollout with uncompressed KV-caching.

        Args:
            x_t: Input token tensor [B, D] or [B, 1, D].
            kv_cache: Optional tuple (k_cache, v_cache) from previous steps,
                      each of shape [B, pos, H, D_h].
            pos: Current sequence position integer (0-indexed).

        Returns:
            Tuple of:
            - out_t: Output representation at timestep t (matches x_t shape).
            - next_kv_cache: Updated tuple (next_k_cache, next_v_cache) of shape [B, pos + 1, H, D_h].
        """
        is_2d = x_t.ndim == 2
        if is_2d:
            x_t = x_t.unsqueeze(1)
        B, S, _ = x_t.shape  # S == 1

        # Project Q, K, V for current single token: [B, 1, H, D_h]
        q_t = self.q_proj(x_t).view(B, 1, self.num_heads, self.head_dim)
        k_t = self.k_proj(x_t).view(B, 1, self.num_heads, self.head_dim)
        v_t = self.v_proj(x_t).view(B, 1, self.num_heads, self.head_dim)

        # Apply RoPE at current position
        q_t = self.rope(q_t, seq_len=1, start_pos=pos)
        k_t = self.rope(k_t, seq_len=1, start_pos=pos)

        # Accumulate uncompressed per-head KV cache (linear O(N) memory growth)
        if kv_cache is None:
            next_k_cache = k_t
            next_v_cache = v_t
        else:
            prev_k, prev_v = kv_cache
            next_k_cache = torch.cat([prev_k, k_t], dim=1)
            next_v_cache = torch.cat([prev_v, v_t], dim=1)

        # Prepare for scaled dot-product attention
        # Query: [B, H, 1, D_h]
        q_step = q_t.transpose(1, 2)
        # Key & Value: [B, H, pos + 1, D_h]
        k_step = next_k_cache.transpose(1, 2)
        v_step = next_v_cache.transpose(1, 2)

        # Attention against all cached tokens up to pos
        attn_out = F.scaled_dot_product_attention(q_step, k_step, v_step, is_causal=False)

        # [B, H, 1, D_h] -> [B, 1, H * D_h] -> [B, 1, D]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.all_head_dim)
        out_t = self.out_proj(attn_out)

        if is_2d:
            out_t = out_t.squeeze(1)

        return out_t, (next_k_cache, next_v_cache)


def get_baseline_config(baseline_name: str) -> JEDM2Config:
    """
    Returns calibrated JEDM2Config for the requested baseline architecture.

    Baselines:
    - 'baseline_a' / 'transformer': Pure Transformer (4 layers, MHA + Dense SwiGLU FFN, ~50.5M params)
    - 'baseline_b' / 'hybrid': Hybrid Attention-SSM (3 layers, Mamba + MLA + Dense SwiGLU FFN, ~50.2M params)
    - 'baseline_c' / 'jed_m2_no_jepa': JED-M2 without JEPA (3 layers, Mamba + MLA + Sparse MoE, ~50.9M params)
    - 'baseline_d' / 'full_jed_m2': Full JED-M2 (3 layers, Mamba + MLA + Sparse MoE + JEPA, ~53.0M params)
    """
    name = baseline_name.lower()
    if name in ("baseline_a", "transformer", "pure_transformer"):
        return JEDM2Config(
            num_layers=4,
            attn_type="mha",
            ffn_type="dense",
            use_mamba=False,
            use_jepa=False,
            dense_intermediate_dim=6592,  # Calibrated for ~50.5M parameters
        )
    elif name in ("baseline_b", "hybrid", "hybrid_ssm_attn"):
        return JEDM2Config(
            num_layers=3,
            attn_type="mla",
            ffn_type="dense",
            use_mamba=True,
            use_jepa=False,
            dense_intermediate_dim=8064,  # Calibrated for ~50.2M parameters
        )
    elif name in ("baseline_c", "jed_m2_no_jepa", "no_jepa"):
        return JEDM2Config(
            num_layers=3,
            attn_type="mla",
            ffn_type="moe",
            use_mamba=True,
            use_jepa=False,
        )
    elif name in ("baseline_d", "full_jed_m2", "jed_m2"):
        return JEDM2Config(
            num_layers=3,
            attn_type="mla",
            ffn_type="moe",
            use_mamba=True,
            use_jepa=True,
        )
    else:
        raise ValueError(f"Unknown baseline name: {baseline_name}. Choose from: baseline_a, baseline_b, baseline_c, baseline_d.")


def create_baseline_a() -> Any:
    """Instantiate Baseline A (Pure Transformer: Standard MHA + Dense SwiGLU FFN, 4 layers, ~50.5M params)."""
    from jed_m2.model import JEDM2Model
    return JEDM2Model(get_baseline_config("baseline_a"))


def create_baseline_b() -> Any:
    """Instantiate Baseline B (Hybrid Attention-SSM: Mamba SSM + MLA + Dense SwiGLU FFN, 3 layers, ~50.2M params)."""
    from jed_m2.model import JEDM2Model
    return JEDM2Model(get_baseline_config("baseline_b"))


def create_baseline_c() -> Any:
    """Instantiate Baseline C (JED-M2 without JEPA: Mamba SSM + MLA + Sparse Top-2 MoE, 3 layers, ~50.9M params)."""
    from jed_m2.model import JEDM2Model
    return JEDM2Model(get_baseline_config("baseline_c"))


def create_baseline_d() -> Any:
    """Instantiate Baseline D (Full JED-M2 with Connected JEPA: 3 layers, ~53.0M params)."""
    from jed_m2.model import JEDM2Model
    return JEDM2Model(get_baseline_config("baseline_d"))
