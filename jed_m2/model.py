"""
JED-M2 Unified Large Action Model (LAM) Architecture.
Assembles the complete backbone:
- JEDM2Block: Composite layer interleaving RMSNorm -> MambaBlock -> RMSNorm -> MultiHeadLatentAttention -> RMSNorm -> SparseTop2MoE.
- JEDM2Model: Full LAM backbone stacking 3 JEDM2Block layers over observation embeddings and JEPAWorldModel,
  feeding NativeActionHead and autoregressive LM reasoning head.
- Supports both sequence-parallel forward pass and multi-step recurrent rollout step(...)
  preserving KV-cache and Mamba rolling state buffers.

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
Target scale: ~45M total parameters / ~14M active parameters per token.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.heads import ActionOutput, NativeActionHead
from jed_m2.jepa import JEPAWorldModel
from jed_m2.mamba import MambaBlock
from jed_m2.mla import MultiHeadLatentAttention, RMSNorm
from jed_m2.moe import SparseTop2MoE


@dataclass
class JEDM2Output:
    """Structured container for full JEDM2Model forward output."""

    lm_logits: torch.Tensor
    action_output: ActionOutput
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    jepa_loss: Optional[torch.Tensor] = None
    jepa_metrics: Optional[Dict[str, torch.Tensor]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to standard dictionary."""
        return {
            "lm_logits": self.lm_logits,
            "action_output": self.action_output,
            "hidden_states": self.hidden_states,
            "aux_loss": self.aux_loss,
            "jepa_loss": self.jepa_loss,
            "jepa_metrics": self.jepa_metrics,
        }

    def __getitem__(self, item: str) -> Any:
        """Allow subscripting for backward compatibility."""
        return getattr(self, item)


class JEDM2Block(nn.Module):
    """
    Composite recurrent layer for JED-M2 Large Action Model.

    Execution Pipeline (Pre-RMSNorm with residual connections):
    1. Mamba SSM Recurrence: x = x + MambaBlock(RMSNorm_1(x))
    2. Multi-Head Latent Attention: x = x + MLA(RMSNorm_2(x))
    3. Sparse Top-2 MoE: x = x + MoE(RMSNorm_3(x))

    Supports both sequence-parallel forward and O(1) step modes.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        eps = config.layer_norm_eps

        # 1. Sequential recurrence (Mamba SSM)
        self.norm_mamba = RMSNorm(self.hidden_dim, eps=eps)
        self.mamba = MambaBlock(config)

        # 2. Compressed Latent Attention (MLA)
        self.norm_mla = RMSNorm(self.hidden_dim, eps=eps)
        self.mla = MultiHeadLatentAttention(config)

        # 3. Sparse routing feed-forward (Top-2 MoE)
        self.norm_moe = RMSNorm(self.hidden_dim, eps=eps)
        self.moe = SparseTop2MoE(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequence forward pass for JEDM2Block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            attention_mask: Optional attention mask for MLA.

        Returns:
            Tuple of:
            - Output tensor of shape [batch_size, seq_len, hidden_dim].
            - Auxiliary load-balancing loss from SparseTop2MoE.
        """
        # 1. Mamba SSM with pre-RMSNorm & residual
        # [B, S, D] -> [B, S, D]
        x = x + self.mamba(self.norm_mamba(x))

        # 2. Multi-Head Latent Attention with pre-RMSNorm & residual
        # [B, S, D] -> [B, S, D]
        x = x + self.mla(self.norm_mla(x), attention_mask=attention_mask)

        # 3. Sparse Top-2 MoE with pre-RMSNorm & residual
        # [B, S, D] -> [B, S, D], aux_loss: scalar
        moe_out, aux_loss = self.moe(self.norm_moe(x))
        x = x + moe_out

        return x, aux_loss

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[Any, Any, Any]] = None,
        pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[Any, Any, Any]]:
        """
        Recurrent single-step update preserving Mamba states and MLA KV cache.

        Args:
            x_t: Input token representation at timestep t, shape [B, D] or [B, 1, D].
            state: Optional tuple (ssm_state, conv_state, kv_cache) from previous timestep.
            pos: Current sequence position integer (0-indexed).

        Returns:
            Tuple of:
            - out_t: Updated representation at timestep t, shape matches x_t.
            - next_state: Updated tuple (next_ssm_state, next_conv_state, next_kv_cache).
        """
        ssm_state, conv_state, kv_cache = (None, None, None) if state is None else state

        # 1. Mamba single-step
        norm_mamba_in = self.norm_mamba(x_t)
        mamba_out, next_ssm_state, next_conv_state = self.mamba.step(
            norm_mamba_in, ssm_state=ssm_state, conv_state=conv_state
        )
        x_t = x_t + mamba_out

        # 2. MLA single-step with KV-caching
        norm_mla_in = self.norm_mla(x_t)
        mla_out, next_kv_cache = self.mla.step(
            norm_mla_in, kv_cache=kv_cache, pos=pos
        )
        x_t = x_t + mla_out

        # 3. Sparse Top-2 MoE single-step (inference mode, aux loss ignored)
        norm_moe_in = self.norm_moe(x_t)
        if norm_moe_in.ndim == 2:
            moe_out, _ = self.moe(norm_moe_in.unsqueeze(1))
            moe_out = moe_out.squeeze(1)
        else:
            moe_out, _ = self.moe(norm_moe_in)
        x_t = x_t + moe_out

        next_state = (next_ssm_state, next_conv_state, next_kv_cache)
        return x_t, next_state


class JEDM2Model(nn.Module):
    """
    Unified JED-M2 Large Action Model (LAM).

    Stack:
    - Token & Observation Embeddings: [vocab_size -> hidden_dim]
    - 3 Stacked JEDM2Block layers (Mamba SSM + MLA + SparseTop2MoE)
    - Final RMSNorm layer
    - Dual-System Decoupled Heads:
      - System 2 (Rationale): Autoregressive LM head [hidden_dim -> vocab_size]
      - System 1 (Action): NativeActionHead (ActionGate, ActionType, Coords, Argument)
    - JEPA Latent World Model: Latent state representation & transition dynamics

    Target parameter budget: ~45M total / ~14M active parameters.
    """

    def __init__(self, config: Optional[JEDM2Config] = None, tie_weights: bool = True):
        super().__init__()
        self.config = config if config is not None else JEDM2Config()
        self.hidden_dim = self.config.hidden_dim
        self.vocab_size = self.config.vocab_size
        self.num_layers = getattr(self.config, "num_layers", 3)

        # 1. Token Embeddings: [vocab_size -> hidden_dim]
        self.tok_embeddings = nn.Embedding(self.vocab_size, self.hidden_dim)

        # 2. Stacked JEDM2Block Recurrent Layers (x3)
        self.blocks = nn.ModuleList(
            [JEDM2Block(self.config) for _ in range(self.num_layers)]
        )

        # 3. Final Root-Mean-Square Normalization
        self.norm_f = RMSNorm(self.hidden_dim, eps=self.config.layer_norm_eps)

        # 4. System 2: Autoregressive LM Head (Latent Reasoning & Rationale)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        # 5. System 1: Native Action Decoding Heads
        self.action_head = NativeActionHead(self.config)

        # 6. Latent World Model (JEPA)
        self.jepa = JEPAWorldModel(self.config)

        # Optional weight tying between token embedding, LM head, and argument head
        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight
            self.action_head.argument_head.proj.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        obs_t: Optional[torch.Tensor] = None,
        action_t: Optional[torch.Tensor] = None,
        obs_next: Optional[torch.Tensor] = None,
    ) -> JEDM2Output:
        """
        Sequence-parallel forward pass for training and verification.

        Args:
            input_ids: Optional input token IDs of shape [batch_size, seq_len].
            inputs_embeds: Optional direct embeddings of shape [batch_size, seq_len, hidden_dim].
            attention_mask: Optional attention mask for MLA layers.
            obs_t: Optional JEPA observation state at time t [batch_size, hidden_dim].
            action_t: Optional JEPA action conditioning at time t [batch_size, action_dim].
            obs_next: Optional JEPA target observation state at time t+1 [batch_size, hidden_dim].

        Returns:
            JEDM2Output container with LM logits, action predictions, hidden states,
            MoE auxiliary loss, and optional JEPA loss.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds.")
            # [B, S] -> [B, S, hidden_dim]
            x = self.tok_embeddings(input_ids)
        else:
            x = inputs_embeds

        # Propagate through stacked JEDM2Block layers and accumulate MoE aux loss
        total_aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x, aux_loss = block(x, attention_mask=attention_mask)
            total_aux_loss = total_aux_loss + aux_loss

        # Normalize final hidden states: [B, S, hidden_dim]
        h_f = self.norm_f(x)

        # System 2: Autoregressive LM rationale logits: [B, S, vocab_size]
        lm_logits = self.lm_head(h_f)

        # System 1: Dual-system action heads
        action_output = self.action_head(h_f)

        # JEPA World Model evaluation (if trajectory transition inputs provided)
        jepa_loss, jepa_metrics = None, None
        if obs_t is not None and action_t is not None and obs_next is not None:
            _, _, jepa_loss, jepa_metrics = self.jepa(obs_t, action_t, obs_next)

        return JEDM2Output(
            lm_logits=lm_logits,
            action_output=action_output,
            hidden_states=h_f,
            aux_loss=total_aux_loss,
            jepa_loss=jepa_loss,
            jepa_metrics=jepa_metrics,
        )

    def step(
        self,
        token_id_or_embed: torch.Tensor,
        states: Optional[List[Tuple[Any, Any, Any]]] = None,
        pos: int = 0,
    ) -> Tuple[torch.Tensor, ActionOutput, List[Tuple[Any, Any, Any]]]:
        """
        Autoregressive multi-step recurrent rollout step.
        Propagates a single token through all stacked layers while preserving
        layer-wise Mamba rolling state buffers and MLA dynamic KV caches.

        Args:
            token_id_or_embed: Single token ID tensor of shape [B] or [B, 1],
                               or token embedding tensor [B, D] or [B, 1, D].
            states: Optional list of per-layer state tuples:
                    [(ssm_state_l, conv_state_l, kv_cache_l), ...] for l in 0..num_layers-1.
            pos: Current sequence position (0-indexed).

        Returns:
            Tuple of:
            - lm_logits_t: LM vocabulary logits for current timestep [B, vocab_size].
            - action_output_t: ActionOutput container for current timestep.
            - next_states: Updated list of per-layer state tuples.
        """
        # Resolve token embedding
        if token_id_or_embed.dtype in (torch.int32, torch.int64):
            if token_id_or_embed.ndim == 2:
                token_id_or_embed = token_id_or_embed.squeeze(1)
            # [B] -> [B, hidden_dim]
            x_t = self.tok_embeddings(token_id_or_embed)
        else:
            if token_id_or_embed.ndim == 3:
                x_t = token_id_or_embed.squeeze(1)
            else:
                x_t = token_id_or_embed

        # Initialize empty states if first step
        if states is None:
            states = [None] * len(self.blocks)

        next_states = []
        for l, block in enumerate(self.blocks):
            layer_state = states[l] if l < len(states) else None
            x_t, next_layer_state = block.step(x_t, state=layer_state, pos=pos)
            next_states.append(next_layer_state)

        # Final RMSNorm
        h_t = self.norm_f(x_t)

        # System 2: LM Logits at timestep t: [B, vocab_size]
        lm_logits_t = self.lm_head(h_t)

        # System 1: Action outputs at timestep t
        action_output_t = self.action_head(h_t)

        return lm_logits_t, action_output_t, next_states

    def count_parameters(self) -> Dict[str, int]:
        """
        Computes total, trainable, and active parameters per token.
        In SparseTop2MoE, only 2 of 8 experts are active per token,
        rendering 6 experts inactive.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Compute inactive MoE parameters across all blocks
        inactive_moe_params = 0
        for block in self.blocks:
            num_experts = block.moe.num_experts
            top_k = block.moe.top_k
            expert_params = sum(p.numel() for p in block.moe.experts.parameters())
            single_expert_params = expert_params // num_experts
            inactive_moe_params += (num_experts - top_k) * single_expert_params

        # JEPA target encoder is non-trainable EMA parameters
        target_encoder_params = sum(
            p.numel() for p in self.jepa.target_encoder.parameters()
        )

        active_params = total_params - inactive_moe_params - target_encoder_params

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "active_params": active_params,
            "inactive_moe_params": inactive_moe_params,
        }
