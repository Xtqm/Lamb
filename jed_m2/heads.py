"""
JED-M2 Dual-System Action Decoding Heads.
Implements System 1 action emission heads:
- ActionGate: Binary/scalar projection (hidden_dim -> 1) predicting when to emit an action vs continue latent reasoning.
- ActionTypeHead: Discrete primitive classifier (hidden_dim -> hidden_dim // 2 -> GELU -> Dropout -> num_action_types)
  covering primitives (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH, SELECT).
- CoordinateHead: Spatial pointer regressor (hidden_dim -> hidden_dim // 4 -> GELU -> 2 -> Sigmoid)
  predicting normalized continuous coordinates (x, y) in [0, 1].
- ArgumentHead: Linear projection (hidden_dim -> vocab_size) for tokenized arguments.
- NativeActionHead: Composite module coordinating all four heads.
"""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.mla import RMSNorm


class ActionType(IntEnum):
    """Discrete UI and digital environment interaction action primitives.

    SELECT is appended at index 6 so six-class legacy checkpoints retain their
    original output shape and class indices.
    """

    CLICK = 0
    TYPE = 1
    SCROLL = 2
    NAVIGATE = 3
    API_CALL = 4
    FINISH = 5
    SELECT = 6


@dataclass
class ActionOutput:
    """Structured container for dual-system action outputs."""

    gate_logits: torch.Tensor
    action_type_logits: torch.Tensor
    coordinates: torch.Tensor
    argument_logits: torch.Tensor
    pointer_probs: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    argument_probs: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to standard dictionary."""
        d = {
            "gate_logits": self.gate_logits,
            "action_type_logits": self.action_type_logits,
            "coordinates": self.coordinates,
            "argument_logits": self.argument_logits,
        }
        if self.pointer_probs is not None:
            d["pointer_probs"] = self.pointer_probs
        if self.alpha is not None:
            d["alpha"] = self.alpha
        if self.argument_probs is not None:
            d["argument_probs"] = self.argument_probs
        return d

    def __getitem__(self, item: str) -> torch.Tensor:
        """Allow dict-style key subscripting for backward compatibility."""
        return getattr(self, item)


class ActionGate(nn.Module):
    """
    Binary / scalar projection predicting action emission vs latent rationale continuation.
    Projects hidden state to scalar logit: hidden_dim -> 1.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        # Projection: [hidden_dim -> 1]
        self.proj = nn.Linear(self.hidden_dim, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for action emission gate.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim].

        Returns:
            Gate logits of shape [batch_size, seq_len, 1] or [batch_size, 1].
        """
        # [B, S, D] -> [B, S, 1] or [B, D] -> [B, 1]
        gate_logits = self.proj(x)
        return gate_logits


class ActionTypeHead(nn.Module):
    """
    Discrete primitive classifier predicting categorical distribution over action types.
    Architecture: hidden_dim -> hidden_dim // 2 -> GELU -> Dropout -> num_action_types.
    """

    def __init__(
        self,
        config: JEDM2Config,
        num_action_types: Optional[int] = None,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_action_types = (
            num_action_types
            if num_action_types is not None
            else getattr(config, "num_action_types", 6)
        )
        self.dropout_p = (
            dropout if dropout is not None else getattr(config, "action_dropout", 0.1)
        )

        mid_dim = self.hidden_dim // 2

        # 1. In-projection: [hidden_dim -> hidden_dim // 2]
        self.fc1 = nn.Linear(self.hidden_dim, mid_dim, bias=True)
        # 2. Activation & regularization
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.dropout_p)
        # 3. Out-projection: [hidden_dim // 2 -> num_action_types]
        self.fc2 = nn.Linear(mid_dim, self.num_action_types, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for discrete primitive classifier.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim].

        Returns:
            Class logits of shape [batch_size, seq_len, num_action_types] or [batch_size, num_action_types].
        """
        # [..., D] -> [..., D // 2]
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        # [..., D // 2] -> [..., num_action_types]
        logits = self.fc2(h)
        return logits


class CoordinateHead(nn.Module):
    """
    Spatial pointer regressor predicting normalized continuous screen coordinates (x, y) in [0, 1].
    Supports cross-attention visual grounding over visual patch tokens.
    Architecture:
    - Optional Visual Cross-Attention: RMSNorm(x + CrossAttn(RMSNorm(x), RMSNorm(v)))
    - MLP: hidden_dim -> hidden_dim // 4 -> GELU -> 2 -> Sigmoid.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        num_heads = getattr(config, "visual_grounding_heads", 4)
        eps = config.layer_norm_eps

        # Visual cross-attention grounding layers
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm_query = RMSNorm(self.hidden_dim, eps=eps)
        self.norm_visual = RMSNorm(self.hidden_dim, eps=eps)
        self.norm_grounded = RMSNorm(self.hidden_dim, eps=eps)

        mid_dim = self.hidden_dim // 4

        # 1. In-projection: [hidden_dim -> hidden_dim // 4]
        self.fc1 = nn.Linear(self.hidden_dim, mid_dim, bias=True)
        # 2. Non-linear activation
        self.act = nn.GELU()
        # 3. Coordinate projection: [hidden_dim // 4 -> 2]
        self.fc2 = nn.Linear(mid_dim, 2, bias=True)
        # 4. Bounded normalization activation: maps R -> [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for continuous coordinate regressor.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim].
            visual_tokens: Optional visual patch tokens [batch_size, N_v, hidden_dim].

        Returns:
            Normalized coordinates in [0, 1] of shape [batch_size, seq_len, 2] or [batch_size, 2].
        """
        if visual_tokens is not None:
            # Query from normalized hidden state, Key/Value from visual patch tokens
            if x.ndim == 2:
                q = self.norm_query(x).unsqueeze(1)  # [B, 1, D]
                kv = self.norm_visual(visual_tokens)  # [B, N_v, D]
                attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)  # [B, 1, D]
                h_grounded = self.norm_grounded(q.squeeze(1) + attn_out.squeeze(1))  # [B, D]
            else:
                q = self.norm_query(x)  # [B, S, D]
                kv = self.norm_visual(visual_tokens)  # [B, N_v, D]
                attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)  # [B, S, D]
                h_grounded = self.norm_grounded(q + attn_out)  # [B, S, D]
        else:
            h_grounded = x

        # [..., D] -> [..., D // 4]
        h = self.fc1(h_grounded)
        h = self.act(h)
        # [..., D // 4] -> [..., 2]
        coords_raw = self.fc2(h)
        # [..., 2] -> [..., 2] bounded in [0, 1]
        coords = self.sigmoid(coords_raw)
        return coords


class ArgumentHead(nn.Module):
    """
    Pointer-generator action argument head.
    Combines vocabulary generation logits with copy attention distributions over input prompt tokens:
        P(word) = alpha * P_vocab(word) + (1 - alpha) * sum_{i : x_i = word} P_pointer(i)
    Falls back gracefully to standard vocabulary projection when prompt inputs are absent.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.pointer_dim = getattr(config, "pointer_dim", max(64, self.hidden_dim // 4))

        # Standard vocabulary linear projection: [hidden_dim -> vocab_size]
        self.vocab_proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        # Backward-compatibility alias (e.g. for weight tying in JEDM2Model)
        self.proj = self.vocab_proj

        # Pointer cross-attention query & key projections
        self.query_proj = nn.Linear(self.hidden_dim, self.pointer_dim, bias=False)
        self.key_proj = nn.Linear(self.hidden_dim, self.pointer_dim, bias=False)

        # Generation probability gate: alpha = sigmoid(W_alpha [h; context] + b_alpha)
        self.gate_fc = nn.Linear(self.hidden_dim * 2, 1, bias=True)

    def forward_components(
        self,
        x: torch.Tensor,
        prompt_hidden: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute argument outputs returning all components: (arg_logits_or_log_p, p_pointer, alpha, p_final).
        """
        if prompt_hidden is None or prompt_ids is None:
            raw_logits = self.vocab_proj(x)
            return raw_logits, None, None, None

        if x.ndim == 2:
            # x: [B, D]
            # prompt_hidden: [B, S_p, D]
            # prompt_ids: [B, S_p]
            q = self.query_proj(x)  # [B, pointer_dim]
            k = self.key_proj(prompt_hidden)  # [B, S_p, pointer_dim]
            scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.pointer_dim)  # [B, S_p]
            num_special = getattr(self.config, "num_special_tokens", 11)
            valid_mask = (prompt_ids >= num_special) & (prompt_ids < self.vocab_size)
            if prompt_mask is not None:
                valid_mask = valid_mask & prompt_mask
            has_valid = valid_mask.any(dim=-1, keepdim=True)
            effective_mask = torch.where(has_valid, valid_mask, (prompt_ids >= 0) & (prompt_ids < self.vocab_size))
            scores = scores.masked_fill(~effective_mask, -1e4)

            p_pointer = F.softmax(scores, dim=-1)  # [B, S_p]
            context = torch.bmm(p_pointer.unsqueeze(1), prompt_hidden).squeeze(1)  # [B, D]
            alpha = torch.sigmoid(self.gate_fc(torch.cat([x, context], dim=-1)))  # [B, 1]
            vocab_logits = self.vocab_proj(x)  # [B, vocab_size]
            p_vocab = F.softmax(vocab_logits, dim=-1)  # [B, vocab_size]

            p_final = alpha * p_vocab  # [B, vocab_size]
            safe_ids = prompt_ids.clamp(0, self.vocab_size - 1)
            copy_src = ((1.0 - alpha) * p_pointer).to(p_final.dtype)
            p_final = p_final.scatter_add(1, safe_ids, copy_src)

            log_p = torch.log(p_final.clamp(min=1e-12))
            return log_p, p_pointer, alpha, p_final
        else:
            # x: [B, S, D]
            B, S, D = x.shape
            _, S_p, _ = prompt_hidden.shape
            q = self.query_proj(x)  # [B, S, pointer_dim]
            k = self.key_proj(prompt_hidden)  # [B, S_p, pointer_dim]
            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.pointer_dim)  # [B, S, S_p]

            num_special = getattr(self.config, "num_special_tokens", 11)
            valid_mask = (prompt_ids >= num_special) & (prompt_ids < self.vocab_size)
            if prompt_mask is not None:
                valid_mask = valid_mask & prompt_mask
            has_valid = valid_mask.any(dim=-1, keepdim=True)
            effective_mask = torch.where(has_valid, valid_mask, (prompt_ids >= 0) & (prompt_ids < self.vocab_size))
            scores = scores.masked_fill(~effective_mask.unsqueeze(1), -1e4)

            p_pointer = F.softmax(scores, dim=-1)  # [B, S, S_p]
            context = torch.bmm(p_pointer, prompt_hidden)  # [B, S, D]
            alpha = torch.sigmoid(self.gate_fc(torch.cat([x, context], dim=-1)))  # [B, S, 1]
            vocab_logits = self.vocab_proj(x)  # [B, S, vocab_size]
            p_vocab = F.softmax(vocab_logits, dim=-1)  # [B, S, vocab_size]

            p_final = alpha * p_vocab  # [B, S, vocab_size]
            safe_ids = prompt_ids.clamp(0, self.vocab_size - 1).unsqueeze(1).expand(-1, S, -1)
            copy_src = ((1.0 - alpha) * p_pointer).to(p_final.dtype)
            p_final = p_final.scatter_add(2, safe_ids, copy_src)

            log_p = torch.log(p_final.clamp(min=1e-12))
            return log_p, p_pointer, alpha, p_final

    def forward(
        self,
        x: torch.Tensor,
        prompt_hidden: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass returning argument logits or log-probabilities.
        """
        arg_logits, _, _, _ = self.forward_components(
            x, prompt_hidden=prompt_hidden, prompt_ids=prompt_ids, prompt_mask=prompt_mask
        )
        return arg_logits


class NativeActionHead(nn.Module):
    """
    Composite module coordinating all four dual-system action heads:
    1. ActionGate: Action emission gating (emit action vs continue thinking).
    2. ActionTypeHead: Discrete UI primitive distribution.
    3. CoordinateHead: Normalized spatial coordinates in [0, 1].
    4. ArgumentHead: Tokenized argument distributions with pointer-generator copy network.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.gate = ActionGate(config)
        self.action_type_head = ActionTypeHead(
            config,
            num_action_types=getattr(config, "num_action_types", 6),
            dropout=getattr(config, "action_dropout", 0.1),
        )
        self.coordinate_head = CoordinateHead(config)
        self.argument_head = ArgumentHead(config)

    def forward(
        self,
        x: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
        prompt_hidden: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> ActionOutput:
        """
        Forward pass executing all four action heads simultaneously.

        Args:
            x: Hidden representations of shape [batch_size, seq_len, hidden_dim]
               or [batch_size, hidden_dim].
            visual_tokens: Optional visual patch tokens [batch_size, N_v, hidden_dim].
            prompt_hidden: Optional prompt sequence hidden representations [batch_size, S_p, hidden_dim].
            prompt_ids: Optional prompt token IDs [batch_size, S_p].
            prompt_mask: Optional prompt boolean valid mask [batch_size, S_p].

        Returns:
            ActionOutput containing gate logits, action type logits, coordinates, argument logits,
            and optional pointer-generator components.
        """
        # 1. Gate logit: [..., 1]
        gate_logits = self.gate(x)
        # 2. Action type logits: [..., num_action_types]
        action_type_logits = self.action_type_head(x)
        # 3. Coordinates: [..., 2] (with optional visual grounding)
        coordinates = self.coordinate_head(x, visual_tokens=visual_tokens)
        # 4. Argument logits: [..., vocab_size] with optional pointer copy
        argument_logits, p_pointer, alpha, p_final = self.argument_head.forward_components(
            x, prompt_hidden=prompt_hidden, prompt_ids=prompt_ids, prompt_mask=prompt_mask
        )

        return ActionOutput(
            gate_logits=gate_logits,
            action_type_logits=action_type_logits,
            coordinates=coordinates,
            argument_logits=argument_logits,
            pointer_probs=p_pointer,
            alpha=alpha,
            argument_probs=p_final,
        )

    def predict_action(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        visual_tokens: Optional[torch.Tensor] = None,
        prompt_hidden: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Inference-time helper to decode discrete action predictions from hidden state.

        Args:
            x: Single-step or batch hidden representations [batch_size, hidden_dim] or [1, hidden_dim].
            threshold: Probability threshold for action gate commitment (defaults to config).
            visual_tokens: Optional visual patch tokens [batch_size, N_v, hidden_dim].
            prompt_hidden: Optional prompt sequence hidden representations [batch_size, S_p, hidden_dim].
            prompt_ids: Optional prompt token IDs [batch_size, S_p].
            prompt_mask: Optional prompt boolean valid mask [batch_size, S_p].

        Returns:
            Dictionary with parsed action predictions:
            - emit_action: bool tensor indicating whether action gate fired
            - gate_prob: scalar sigmoid probability of action emission
            - action_type_id: argmax integer ID of action type
            - action_type_name: human-readable name of predicted primitive
            - coordinates: (x, y) tensor in [0, 1]
            - argument_id: argmax token ID for argument head
        """
        if threshold is None:
            threshold = getattr(self.config, "action_gate_threshold", 0.5)

        out = self.forward(
            x,
            visual_tokens=visual_tokens,
            prompt_hidden=prompt_hidden,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
        )

        # Gate sigmoid probability
        gate_prob = torch.sigmoid(out.gate_logits)
        emit_action = gate_prob >= threshold

        # Argmax action primitive
        action_type_id = torch.argmax(out.action_type_logits, dim=-1)

        # Coordinates
        coordinates = out.coordinates

        # Argmax argument token
        argument_id = torch.argmax(out.argument_logits, dim=-1)

        # Decode string names
        flat_ids = action_type_id.view(-1).tolist()
        names = [ActionType(idx).name for idx in flat_ids]

        return {
            "emit_action": emit_action,
            "gate_prob": gate_prob,
            "action_type_id": action_type_id,
            "action_type_name": names if len(names) > 1 else names[0],
            "coordinates": coordinates,
            "argument_id": argument_id,
        }
