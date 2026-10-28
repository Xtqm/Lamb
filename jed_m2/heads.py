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

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert container to standard dictionary."""
        return {
            "gate_logits": self.gate_logits,
            "action_type_logits": self.action_type_logits,
            "coordinates": self.coordinates,
            "argument_logits": self.argument_logits,
        }

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
    Linear projection for tokenized arguments (e.g., typing keystrokes, query strings).
    Architecture: hidden_dim -> vocab_size.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size

        # Linear projection: [hidden_dim -> vocab_size]
        self.proj = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for argument token projection.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim].

        Returns:
            Argument logits of shape [batch_size, seq_len, vocab_size] or [batch_size, vocab_size].
        """
        # [..., D] -> [..., vocab_size]
        arg_logits = self.proj(x)
        return arg_logits


class NativeActionHead(nn.Module):
    """
    Composite module coordinating all four dual-system action heads:
    1. ActionGate: Action emission gating (emit action vs continue thinking).
    2. ActionTypeHead: Discrete UI primitive distribution.
    3. CoordinateHead: Normalized spatial coordinates in [0, 1].
    4. ArgumentHead: Tokenized argument distributions.
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
    ) -> ActionOutput:
        """
        Forward pass executing all four action heads simultaneously.

        Args:
            x: Hidden representations of shape [batch_size, seq_len, hidden_dim]
               or [batch_size, hidden_dim].
            visual_tokens: Optional visual patch tokens [batch_size, N_v, hidden_dim].

        Returns:
            ActionOutput containing gate logits, action type logits, coordinates, and argument logits.
        """
        # 1. Gate logit: [..., 1]
        gate_logits = self.gate(x)
        # 2. Action type logits: [..., num_action_types]
        action_type_logits = self.action_type_head(x)
        # 3. Coordinates: [..., 2] (with optional visual grounding)
        coordinates = self.coordinate_head(x, visual_tokens=visual_tokens)
        # 4. Argument logits: [..., vocab_size]
        argument_logits = self.argument_head(x)

        return ActionOutput(
            gate_logits=gate_logits,
            action_type_logits=action_type_logits,
            coordinates=coordinates,
            argument_logits=argument_logits,
        )

    def predict_action(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Inference-time helper to decode discrete action predictions from hidden state.

        Args:
            x: Single-step or batch hidden representations [batch_size, hidden_dim] or [1, hidden_dim].
            threshold: Probability threshold for action gate commitment (defaults to config).
            visual_tokens: Optional visual patch tokens [batch_size, N_v, hidden_dim].

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

        out = self.forward(x, visual_tokens=visual_tokens)

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
