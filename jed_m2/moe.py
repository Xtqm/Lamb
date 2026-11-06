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
        self.last_routing_stats = {}

        # Expert MLPs
        expert_cls = SwiGLUFeedForward if config.activation == "swiglu" else GELUFeedForward
        self.experts = nn.ModuleList([
            expert_cls(self.hidden_dim, config.moe_intermediate_dim)
            for _ in range(self.num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Sparse Top-2 MoE with optional Per-Source Load Balancing (Guardrail 4A).

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            source_mask: Optional tensor indicating sample source: 1 for REAL, 0 for SYNTH.
                         Shape [batch_size] or [batch_size, seq_len].

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
        # 3. Auxiliary Load-Balancing Loss (Guardrail 4A)
        # ==========================================
        # One-hot encoding of selected experts: [num_tokens, 2, num_experts] -> [num_tokens, num_experts]
        expert_mask = F.one_hot(top2_indices, num_classes=self.num_experts).sum(dim=1)

        if source_mask is not None:
            if source_mask.dim() == 1:
                source_mask_expanded = source_mask.unsqueeze(1).expand(b_sz, seq_len)
            else:
                source_mask_expanded = source_mask
            source_mask_flat = source_mask_expanded.reshape(-1)

            real_indices = (source_mask_flat > 0.5).nonzero(as_tuple=True)[0]
            synth_indices = (source_mask_flat <= 0.5).nonzero(as_tuple=True)[0]

            has_real = (real_indices.numel() > 0)
            has_synth = (synth_indices.numel() > 0)

            if has_real and has_synth:
                p_real = router_probs[real_indices].mean(dim=0)
                f_real = expert_mask[real_indices].float().mean(dim=0).detach()
                l_balance_real = self.num_experts * torch.sum(f_real * p_real)

                p_synth = router_probs[synth_indices].mean(dim=0)
                f_synth = expert_mask[synth_indices].float().mean(dim=0).detach()
                l_balance_synth = self.num_experts * torch.sum(f_synth * p_synth)

                aux_loss = self.aux_loss_coef * (l_balance_real + l_balance_synth)

                h_real = -(p_real * torch.log(p_real + 1e-12)).sum().item()
                h_synth = -(p_synth * torch.log(p_synth + 1e-12)).sum().item()
                self.last_routing_stats = {
                    "entropy_real": h_real,
                    "entropy_synth": h_synth,
                    "loss_real": l_balance_real.item(),
                    "loss_synth": l_balance_synth.item(),
                }
            else:
                p_i = router_probs.mean(dim=0)
                f_i = expert_mask.float().mean(dim=0).detach()
                aux_loss = self.aux_loss_coef * (self.num_experts * torch.sum(f_i * p_i))
                h_all = -(p_i * torch.log(p_i + 1e-12)).sum().item()
                self.last_routing_stats = {
                    "entropy_real": h_all if has_real else 0.0,
                    "entropy_synth": h_all if has_synth else 0.0,
                    "loss_real": (self.num_experts * torch.sum(f_i * p_i)).item() if has_real else 0.0,
                    "loss_synth": (self.num_experts * torch.sum(f_i * p_i)).item() if has_synth else 0.0,
                }
        else:
            # P_i: Average probability allocated to expert i across all tokens
            p_i = router_probs.mean(dim=0)
            # f_i: Fraction of tokens routed to expert i in Top-2 selection
            f_i = expert_mask.float().mean(dim=0).detach()
            # L_aux = aux_loss_coef * N * sum_{i=1}^N (f_i * P_i)
            aux_loss = self.aux_loss_coef * (self.num_experts * torch.sum(f_i * p_i))
            h_all = -(p_i * torch.log(p_i + 1e-12)).sum().item()
            self.last_routing_stats = {"entropy_real": h_all, "entropy_synth": h_all}

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
