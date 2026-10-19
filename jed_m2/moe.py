"""
Sparse Top-2 Mixture of Experts (MoE) block with auxiliary load-balancing loss.
Supports SwiGLU and GELU expert feed-forward networks.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config


class SwiGLUFeedForward(nn.Module):
    """SwiGLU Multi-Layer Perceptron (Linear -> SiLU(Gate) * Up -> Linear)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        # Gate and Up projections
        self.w_gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        # Down projection
        self.w_down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [num_tokens, hidden_dim]
        # gate: [num_tokens, intermediate_dim]
        # up: [num_tokens, intermediate_dim]
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        # down: [num_tokens, hidden_dim]
        return self.w_down(gate * up)


class GELUFeedForward(nn.Module):
    """Standard GELU Multi-Layer Perceptron (Linear -> GELU -> Linear)."""

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))


class SparseTop2MoE(nn.Module):
    """
    Sparse Top-2 Mixture of Experts (MoE) Layer.

    Features:
    - Softmax gating router selecting Top-2 experts per token.
    - Renormalized Top-2 routing weights summing to 1.0.
    - Switch Transformer / DeepSeek style load-balancing auxiliary loss:
        L_aux = aux_loss_coef * N * sum_{i=1}^N (f_i * P_i)
      where f_i is fraction of tokens routed to expert i, and P_i is average
      routing probability for expert i.
    - Modular expert networks (SwiGLU default, GELU configurable).
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_experts = config.num_experts
        self.top_k = config.top_k  # 2
        self.aux_loss_coef = config.aux_loss_coef

        # Router: Linear projection from hidden_dim to num_experts
        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # Expert MLPs
        expert_cls = SwiGLUFeedForward if config.activation == "swiglu" else GELUFeedForward
        self.experts = nn.ModuleList([
            expert_cls(self.hidden_dim, config.moe_intermediate_dim)
            for _ in range(self.num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Sparse Top-2 MoE.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]

        Returns:
            Tuple of:
                output: [batch_size, seq_len, hidden_dim]
                aux_loss: scalar tensor representing load-balancing auxiliary loss
        """
        b_sz, seq_len, hidden_dim = x.shape
        num_tokens = b_sz * seq_len

        # Flatten batch and sequence dimensions for token-level routing
        # x_flat: [num_tokens, hidden_dim]
        x_flat = x.view(num_tokens, hidden_dim)

        # ==========================================
        # 1. Router Gating & Probabilities
        # ==========================================
        # router_logits: [num_tokens, num_experts]
        router_logits = self.router(x_flat)

        # Compute routing probabilities in fp32 for numerical stability
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # ==========================================
        # 2. Top-2 Selection & Renormalization
        # ==========================================
        # top2_probs: [num_tokens, 2], top2_indices: [num_tokens, 2]
        top2_probs, top2_indices = torch.topk(router_probs, k=self.top_k, dim=-1)

        # Renormalize Top-2 weights so each token's expert weights sum to 1.0
        top2_weights = top2_probs / (top2_probs.sum(dim=-1, keepdim=True) + 1e-20)
        # Cast weights back to x's dtype
        top2_weights = top2_weights.to(dtype=x.dtype)

        # ==========================================
        # 3. Auxiliary Load-Balancing Loss
        # ==========================================
        # P_i: Average probability allocated to expert i across all tokens
        # P_i retains gradients back to router parameters
        # p_i: [num_experts]
        p_i = router_probs.mean(dim=0)

        # f_i: Fraction of tokens routed to expert i in Top-2 selection
        # One-hot encoding of selected experts: [num_tokens, 2, num_experts] -> [num_tokens, num_experts]
        expert_mask = F.one_hot(top2_indices, num_classes=self.num_experts).sum(dim=1)
        # Detach f_i as token assignment is a discrete routing decision
        f_i = expert_mask.float().mean(dim=0).detach()

        # L_aux = aux_loss_coef * N * sum_{i=1}^N (f_i * P_i)
        aux_loss = self.aux_loss_coef * (self.num_experts * torch.sum(f_i * p_i))

        # ==========================================
        # 4. Expert Dispatch & Weighted Output
        # ==========================================
        # Initialize output accumulator
        final_output = torch.zeros_like(x_flat)

        # Dispatch tokens to their selected experts
        for expert_idx, expert in enumerate(self.experts):
            # Find all tokens and their slot (0 or 1) that routed to expert_idx
            token_indices, slot_indices = torch.where(top2_indices == expert_idx)
            if token_indices.numel() == 0:
                continue

            # Gather tokens routed to this expert
            tokens_for_expert = x_flat[token_indices]

            # Execute expert MLP
            expert_out = expert(tokens_for_expert)

            # Get corresponding normalized routing weights: [num_assigned, 1]
            weights = top2_weights[token_indices, slot_indices].unsqueeze(-1)

            # Accumulate weighted expert output back into final tensor
            final_output = final_output.index_add(0, token_indices, weights * expert_out)

        # Reshape back to [batch_size, seq_len, hidden_dim]
        output = final_output.view(b_sz, seq_len, hidden_dim)
        return output, aux_loss
