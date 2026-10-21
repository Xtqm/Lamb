"""
Multi-Head Latent Attention (MLA) with Low-Rank Key-Value & Query Compression.
Implements DeepSeek-style decoupled Rotary Position Embedding (RoPE).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Decoupled Rotary Position Embedding (RoPE).
    Operates on the positional sub-vector of dimension qk_rope_head_dim.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # inv_freq: [dim // 2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Precompute cosine and sine frequency tables."""
        # t: [seq_len]
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # freqs: [seq_len, dim // 2]
        freqs = torch.outer(t, self.inv_freq)
        # emb: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        # Split last dimension in half: [-x2, x1]
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> torch.Tensor:
        """
        Apply RoPE to input tensor x.
        x shape: [batch_size, seq_len, num_heads or 1, dim]
        """
        end_pos = start_pos + seq_len
        if end_pos > self.cos_cached.shape[0]:
            self._build_cache(end_pos)

        # cos, sin: [1, seq_len, 1, dim]
        cos = (
            self.cos_cached[start_pos:end_pos]
            .to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        sin = (
            self.sin_cached[start_pos:end_pos]
            .to(dtype=x.dtype, device=x.device)
            .unsqueeze(0)
            .unsqueeze(2)
        )

        # Standard RoPE application
        return (x * cos) + (self._rotate_half(x) * sin)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) with Low-Rank KV and Query Compression.

    Architectural details:
    1. Query Low-Rank Compression:
       x -> c_Q = W_DQ(x) -> RMSNorm(c_Q)
       Q_nope = W_UQ(c_Q), Q_rope = RoPE(W_QR(c_Q))
       Q = [Q_nope, Q_rope]
    2. Key-Value Low-Rank Compression:
       x -> c_KV = W_DKV(x) -> RMSNorm(c_KV)
       K_nope = W_UK(c_KV), K_rope = RoPE(W_KR(x))  (Shared RoPE key broadcasted across heads)
       K = [K_nope, K_rope]
       V = W_UV(c_KV)
    3. Scaled Dot-Product Attention with FlashAttention causal execution.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_head_dim  # qk_nope_head_dim + qk_rope_head_dim = 64

        # Query Compression and Decompression projections
        # W_DQ: [hidden_dim -> q_lora_rank]
        self.w_dq = nn.Linear(self.hidden_dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.layer_norm_eps)

        # W_UQ: [q_lora_rank -> num_heads * qk_nope_head_dim]
        self.w_uq = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        # W_QR: [q_lora_rank -> num_heads * qk_rope_head_dim]
        self.w_qr = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_rope_head_dim, bias=False)

        # Key-Value Compression and Decompression projections
        # W_DKV: [hidden_dim -> kv_lora_rank]
        self.w_dkv = nn.Linear(self.hidden_dim, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.layer_norm_eps)

        # W_UK: [kv_lora_rank -> num_heads * qk_nope_head_dim]
        self.w_uk = nn.Linear(self.kv_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        # W_KR: [hidden_dim -> qk_rope_head_dim] (Shared across heads per DeepSeek MLA)
        self.w_kr = nn.Linear(self.hidden_dim, self.qk_rope_head_dim, bias=False)
        # W_UV: [kv_lora_rank -> num_heads * v_head_dim]
        self.w_uv = nn.Linear(self.kv_lora_rank, self.num_heads * self.v_head_dim, bias=False)

        # Output projection
        # W_O: [num_heads * v_head_dim -> hidden_dim]
        self.w_o = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_dim, bias=False)

        # Rotary Positional Embedding for positional head dim
        self.rotary_emb = RotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Multi-Head Latent Attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask: Optional attention mask (defaults to causal)

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        b_sz, seq_len, _ = x.shape

        # ==========================================
        # 1. Query Projection & Compression
        # ==========================================
        # x: [B, S, hidden_dim]
        # c_q: [B, S, q_lora_rank] (256)
        c_q = self.w_dq(x)
        c_q = self.q_norm(c_q)

        # Q_nope: [B, S, num_heads * qk_nope_head_dim] -> [B, S, num_heads, qk_nope_head_dim]
        q_nope = self.w_uq(c_q).view(b_sz, seq_len, self.num_heads, self.qk_nope_head_dim)

        # Q_rope: [B, S, num_heads * qk_rope_head_dim] -> [B, S, num_heads, qk_rope_head_dim]
        q_rope = self.w_qr(c_q).view(b_sz, seq_len, self.num_heads, self.qk_rope_head_dim)
        # Apply RoPE to query positional sub-vectors
        q_rope = self.rotary_emb(q_rope, seq_len)

        # Combined Q: [B, S, num_heads, qk_nope_head_dim + qk_rope_head_dim] = [B, S, 8, 64]
        q = torch.cat([q_nope, q_rope], dim=-1)

        # ==========================================
        # 2. Key-Value Projection & Compression
        # ==========================================
        # c_kv: [B, S, kv_lora_rank] (128)
        c_kv = self.w_dkv(x)
        c_kv = self.kv_norm(c_kv)

        # K_nope: [B, S, num_heads * qk_nope_head_dim] -> [B, S, num_heads, qk_nope_head_dim]
        k_nope = self.w_uk(c_kv).view(b_sz, seq_len, self.num_heads, self.qk_nope_head_dim)

        # K_rope: [B, S, qk_rope_head_dim] -> [B, S, 1, qk_rope_head_dim]
        # Shared RoPE key across all heads, saving memory in KV cache
        k_rope = self.w_kr(x).view(b_sz, seq_len, 1, self.qk_rope_head_dim)
        k_rope = self.rotary_emb(k_rope, seq_len)
        # Broadcast across heads: [B, S, num_heads, qk_rope_head_dim]
        k_rope = k_rope.expand(-1, -1, self.num_heads, -1)

        # Combined K: [B, S, num_heads, qk_nope_head_dim + qk_rope_head_dim] = [B, S, 8, 64]
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Values: [B, S, num_heads * v_head_dim] -> [B, S, num_heads, v_head_dim] = [B, S, 8, 64]
        v = self.w_uv(c_kv).view(b_sz, seq_len, self.num_heads, self.v_head_dim)

        # ==========================================
        # 3. Scaled Dot-Product Attention
        # ==========================================
        # Transpose to [B, num_heads, S, head_dim] for PyTorch SDPA
        # q: [B, 8, S, 64]
        # k: [B, 8, S, 64]
        # v: [B, 8, S, 64]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use PyTorch native scaled_dot_product_attention (FlashAttention / MemoryEfficient backend)
        is_causal = attention_mask is None
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
        )
        # attn_out: [B, num_heads, S, v_head_dim] -> [B, S, num_heads, v_head_dim]
        attn_out = attn_out.transpose(1, 2).contiguous()
        # Flatten heads: [B, S, num_heads * v_head_dim] = [B, S, 512]
        attn_out = attn_out.view(b_sz, seq_len, self.num_heads * self.v_head_dim)

        # ==========================================
        # 4. Output Projection
        # ==========================================
        # out: [B, S, hidden_dim] = [B, S, 512]
        out = self.w_o(attn_out)
        return out

    def step(
        self,
        x_t: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Recurrent single-token step for Multi-Head Latent Attention with True Latent Caching
        and Matrix Absorption.
        
        DeepSeek MLA Optimization:
        - Stores only compressed latent state c_KV [B, S, d_c] and shared RoPE key K_rope [B, S, 1, d_R]
          in the KV cache (160 floats/token vs 1024 floats/token in standard multi-head caching).
        - Absorb W_UK into W_Q: Q_absorbed_h = Q_nope_h @ W_UK_h, enabling direct inner product
          with the compressed latent vector c_KV without expanding multi-head keys into memory.
        - Computes values via associative projection: (Attn @ c_KV) @ W_UV^T.

        Args:
            x_t: Input tensor for current timestep, shape [B, D] or [B, 1, D].
            kv_cache: Optional tuple of (cached_c_kv, cached_k_rope).
                      cached_c_kv shape: [B, past_len, kv_lora_rank]
                      cached_k_rope shape: [B, past_len, 1, qk_rope_head_dim]
            pos: Current sequence position integer (0-indexed).

        Returns:
            Tuple of:
            - out_t: Output tensor for current timestep, shape matches x_t [B, D] or [B, 1, D].
            - next_kv_cache: Updated (c_kv_all, k_rope_all) latent cache tuple.
        """
        squeeze_needed = False
        if x_t.ndim == 2:
            x_t = x_t.unsqueeze(1)  # [B, 1, D]
            squeeze_needed = True

        b_sz, seq_len, _ = x_t.shape

        # 1. Query projection & compression: [B, 1, q_lora_rank]
        c_q = self.w_dq(x_t)
        c_q = self.q_norm(c_q)

        # Q_nope: [B, num_heads, 1, qk_nope_head_dim]
        q_nope = (
            self.w_uq(c_q)
            .view(b_sz, seq_len, self.num_heads, self.qk_nope_head_dim)
            .transpose(1, 2)
        )
        # Q_rope: [B, num_heads, 1, qk_rope_head_dim]
        q_rope = (
            self.w_qr(c_q)
            .view(b_sz, seq_len, self.num_heads, self.qk_rope_head_dim)
        )
        q_rope = self.rotary_emb(q_rope, seq_len=1, start_pos=pos).transpose(1, 2)

        # Absorb W_UK into Q_nope:
        # W_UK: [num_heads * qk_nope_head_dim, kv_lora_rank] -> [num_heads, qk_nope_head_dim, kv_lora_rank]
        w_uk_heads = self.w_uk.weight.view(
            self.num_heads, self.qk_nope_head_dim, self.kv_lora_rank
        )
        # Q_absorbed: [B, num_heads, 1, kv_lora_rank]
        q_absorbed = torch.matmul(q_nope, w_uk_heads.unsqueeze(0))

        # 2. Key-Value latent projection & RoPE key
        # c_kv: [B, 1, kv_lora_rank]
        c_kv = self.w_dkv(x_t)
        c_kv = self.kv_norm(c_kv)

        # Shared RoPE key: [B, 1, 1, qk_rope_head_dim]
        k_rope = self.w_kr(x_t).view(b_sz, seq_len, 1, self.qk_rope_head_dim)
        k_rope = self.rotary_emb(k_rope, seq_len=1, start_pos=pos)

        # 3. Update True Latent KV-cache
        if kv_cache is None:
            c_kv_all = c_kv
            k_rope_all = k_rope
        else:
            c_kv_prev, k_rope_prev = kv_cache
            c_kv_all = torch.cat([c_kv_prev, c_kv], dim=1)
            k_rope_all = torch.cat([k_rope_prev, k_rope], dim=1)

        next_kv_cache = (c_kv_all, k_rope_all)

        # 4. Attention scores directly over compressed latent cache:
        # q_absorbed: [B, num_heads, 1, kv_lora_rank]
        # c_kv_all.unsqueeze(1): [B, 1, S, kv_lora_rank]
        # scores_nope: [B, num_heads, 1, S]
        scores_nope = torch.matmul(q_absorbed, c_kv_all.unsqueeze(1).transpose(-1, -2))

        # q_rope: [B, num_heads, 1, qk_rope_head_dim]
        # k_rope_all.squeeze(2).unsqueeze(1): [B, 1, S, qk_rope_head_dim]
        # scores_rope: [B, num_heads, 1, S]
        scores_rope = torch.matmul(
            q_rope, k_rope_all.squeeze(2).unsqueeze(1).transpose(-1, -2)
        )

        scores = (scores_nope + scores_rope) / (self.qk_head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # [B, num_heads, 1, S]

        # 5. Value aggregation & projection:
        # u: [B, num_heads, 1, kv_lora_rank]
        u = torch.matmul(attn_weights, c_kv_all.unsqueeze(1))

        # Project aggregated latent vectors with W_UV:
        # W_UV: [num_heads * v_head_dim, kv_lora_rank] -> [num_heads, v_head_dim, kv_lora_rank]
        w_uv_heads = self.w_uv.weight.view(
            self.num_heads, self.v_head_dim, self.kv_lora_rank
        )
        # v_out: [B, num_heads, 1, v_head_dim]
        v_out = torch.matmul(u, w_uv_heads.unsqueeze(0).transpose(-1, -2))

        # Combine heads: [B, 1, num_heads * v_head_dim]
        attn_out = (
            v_out.transpose(1, 2)
            .contiguous()
            .view(b_sz, seq_len, self.num_heads * self.v_head_dim)
        )

        # 6. Output projection
        out_t = self.w_o(attn_out)

        if squeeze_needed:
            out_t = out_t.squeeze(1)

        return out_t, next_kv_cache
