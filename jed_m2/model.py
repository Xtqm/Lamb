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
from jed_m2.baselines import DenseSwiGLUFFN, StandardMHA
from jed_m2.heads import ActionOutput, NativeActionHead
from jed_m2.jepa import JEPAWorldModel
from jed_m2.mamba import MambaBlock
from jed_m2.mla import MultiHeadLatentAttention, RMSNorm
from jed_m2.moe import SparseTop2MoE
from jed_m2.vision import VisionPatchEncoder


@dataclass
class JEDM2Output:
    """Structured container for full JEDM2Model forward output."""

    lm_logits: torch.Tensor
    action_output: ActionOutput
    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    jepa_loss: Optional[torch.Tensor] = None
    jepa_metrics: Optional[Dict[str, torch.Tensor]] = None
    visual_tokens: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert container to standard dictionary."""
        return {
            "lm_logits": self.lm_logits,
            "action_output": self.action_output,
            "hidden_states": self.hidden_states,
            "aux_loss": self.aux_loss,
            "jepa_loss": self.jepa_loss,
            "jepa_metrics": self.jepa_metrics,
            "visual_tokens": self.visual_tokens,
        }

    def __getitem__(self, item: str) -> Any:
        """Allow subscripting for backward compatibility."""
        return getattr(self, item)


class JEDM2Block(nn.Module):
    """
    Composite recurrent layer for JED-M2 Large Action Model.

    Execution Pipeline (Pre-RMSNorm with residual connections):
    1. Mamba SSM Recurrence: x = x + MambaBlock(RMSNorm_1(x)) (if use_mamba)
    2. Attention: x = x + Attention(RMSNorm_2(x)) (MLA or StandardMHA)
    3. Feed-Forward: x = x + FFN(RMSNorm_3(x)) (Sparse Top-2 MoE or Dense SwiGLU)

    Supports both sequence-parallel forward and O(1) step modes.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        eps = config.layer_norm_eps

        # Configuration toggles
        self.use_mamba = getattr(config, "use_mamba", True)
        self.attn_type = getattr(config, "attn_type", "mla")
        self.ffn_type = getattr(config, "ffn_type", "moe")

        # 1. Sequential recurrence (Mamba SSM)
        if self.use_mamba:
            self.norm_mamba = RMSNorm(self.hidden_dim, eps=eps)
            self.mamba = MambaBlock(config)
        else:
            self.norm_mamba = None
            self.mamba = None

        # 2. Attention (MLA or Standard MHA)
        self.norm_attn = RMSNorm(self.hidden_dim, eps=eps)
        if self.attn_type == "mha":
            self.attn = StandardMHA(config)
        else:
            self.attn = MultiHeadLatentAttention(config)

        # Backward compatibility aliases
        self.norm_mla = self.norm_attn
        self.mla = self.attn

        # 3. Feed-Forward Network (Sparse Top-2 MoE or Dense SwiGLU)
        self.norm_ffn = RMSNorm(self.hidden_dim, eps=eps)
        if self.ffn_type == "dense":
            dense_dim = getattr(config, "dense_intermediate_dim", 1024)
            self.ffn = DenseSwiGLUFFN(self.hidden_dim, dense_dim)
        else:
            self.ffn = SparseTop2MoE(config)

        # Backward compatibility aliases
        self.norm_moe = self.norm_ffn
        self.moe = self.ffn

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sequence forward pass for JEDM2Block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            attention_mask: Optional attention mask for Attention.

        Returns:
            Tuple of:
            - Output tensor of shape [batch_size, seq_len, hidden_dim].
            - Auxiliary load-balancing loss (scalar, 0.0 for dense FFN).
        """
        # 1. Mamba SSM with pre-RMSNorm & residual (if enabled)
        if self.use_mamba and self.mamba is not None:
            # [B, S, D] -> [B, S, D]
            x = x + self.mamba(self.norm_mamba(x))

        # 2. Attention with pre-RMSNorm & residual
        # [B, S, D] -> [B, S, D]
        x = x + self.attn(self.norm_attn(x), attention_mask=attention_mask)

        # 3. Feed-Forward with pre-RMSNorm & residual
        # [B, S, D] -> [B, S, D]
        if self.ffn_type == "dense":
            ffn_out = self.ffn(self.norm_ffn(x))
            aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        else:
            ffn_out, aux_loss = self.ffn(self.norm_ffn(x))
        x = x + ffn_out

        return x, aux_loss

    def step(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[Any, Any, Any]] = None,
        pos: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[Any, Any, Any]]:
        """
        Recurrent single-step update preserving Mamba states and KV cache.

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

        # 1. Mamba single-step (if enabled)
        if self.use_mamba and self.mamba is not None:
            norm_mamba_in = self.norm_mamba(x_t)
            mamba_out, next_ssm_state, next_conv_state = self.mamba.step(
                norm_mamba_in, ssm_state=ssm_state, conv_state=conv_state
            )
            x_t = x_t + mamba_out
        else:
            next_ssm_state, next_conv_state = None, None

        # 2. Attention single-step with KV-caching
        norm_attn_in = self.norm_attn(x_t)
        attn_out, next_kv_cache = self.attn.step(
            norm_attn_in, kv_cache=kv_cache, pos=pos
        )
        x_t = x_t + attn_out

        # 3. Feed-Forward single-step
        norm_ffn_in = self.norm_ffn(x_t)
        if self.ffn_type == "dense":
            ffn_out = self.ffn(norm_ffn_in)
        else:
            if norm_ffn_in.ndim == 2:
                ffn_out, _ = self.ffn(norm_ffn_in.unsqueeze(1))
                ffn_out = ffn_out.squeeze(1)
            else:
                ffn_out, _ = self.ffn(norm_ffn_in)
        x_t = x_t + ffn_out

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

        # 6. Latent World Model (JEPA) (if enabled)
        self.use_jepa = getattr(self.config, "use_jepa", True)
        if self.use_jepa:
            self.jepa = JEPAWorldModel(self.config)
        else:
            self.jepa = None

        # 7. Vision Patch Encoder (Phase 6D)
        self.vision_encoder = VisionPatchEncoder(self.config)

        self.gradient_checkpointing: bool = False

        # Optional weight tying between token embedding, LM head, and argument head
        if tie_weights:
            self.lm_head.weight = self.tok_embeddings.weight
            self.action_head.argument_head.proj.weight = self.tok_embeddings.weight

    def gradient_checkpointing_enable(self) -> None:
        """Enables selective gradient checkpointing across recurrent blocks during training."""
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Disables gradient checkpointing."""
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,
        obs_t: Optional[torch.Tensor] = None,
        action_t: Optional[torch.Tensor] = None,
        obs_next: Optional[torch.Tensor] = None,
        visual_tokens_t: Optional[torch.Tensor] = None,
        visual_tokens_next: Optional[torch.Tensor] = None,
        return_all_tokens: bool = False,
    ) -> JEDM2Output:
        """
        Sequence-parallel forward pass for training and verification.
        Supports dual multimodal observations (DOM/text tokens + visual screenshot patches).

        Args:
            input_ids: Optional input token IDs of shape [batch_size, seq_len].
            inputs_embeds: Optional direct embeddings of shape [batch_size, seq_len, hidden_dim].
            attention_mask: Optional attention mask for MLA layers.
            images: Optional screenshot image tensor [batch_size, 3, H, W].
            visual_tokens: Optional pre-extracted visual patch tokens [batch_size, N_v, hidden_dim].
            obs_t: Optional JEPA observation state at time t [batch_size, hidden_dim].
            action_t: Optional JEPA action conditioning at time t [batch_size, action_dim].
            obs_next: Optional JEPA target observation state at time t+1 [batch_size, hidden_dim].
            visual_tokens_t: Optional JEPA visual patch tokens at time t [batch_size, N_v, hidden_dim].
            visual_tokens_next: Optional JEPA visual patch tokens at time t+1 [batch_size, N_v, hidden_dim].
            return_all_tokens: If True, returns lm_logits/hidden_states over entire concatenated sequence.

        Returns:
            JEDM2Output container with LM logits, action predictions, hidden states,
            MoE auxiliary loss, optional JEPA loss, and visual tokens.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds.")
            # [B, S_t] -> [B, S_t, hidden_dim]
            x_text = self.tok_embeddings(input_ids)
        else:
            x_text = inputs_embeds

        # 1. Process visual inputs if provided
        if visual_tokens is None and images is not None:
            # [B, 3, H, W] -> [B, N_v, hidden_dim]
            visual_tokens = self.vision_encoder(images)

        num_visual_tokens = 0
        if visual_tokens is not None:
            num_visual_tokens = visual_tokens.shape[1]
            # Early fusion: prefix visual tokens before text/DOM tokens
            # [B, N_v, D] + [B, S_t, D] -> [B, N_v + S_t, D]
            x = torch.cat([visual_tokens, x_text], dim=1)

            # Update or construct multimodal attention mask if mask provided
            if attention_mask is not None:
                B, S_total = x.shape[0], x.shape[1]
                if attention_mask.ndim == 4 and attention_mask.shape[-1] != S_total:
                    # Input mask was defined for text tokens: [B, 1, S_t, S_t]
                    N_v = num_visual_tokens
                    full_mask = torch.ones(
                        (B, 1, S_total, S_total),
                        dtype=torch.bool,
                        device=x.device,
                    )
                    # Visual tokens do not attend to subsequent text tokens
                    full_mask[:, :, :N_v, N_v:] = False
                    # Text tokens attend to text tokens via provided mask
                    full_mask[:, :, N_v:, N_v:] = attention_mask
                    # Text tokens attend to visual tokens (already True)
                    # Visual tokens attend to visual tokens (already True)
                    attention_mask = full_mask
        else:
            x = x_text

        # Propagate through stacked JEDM2Block layers and accumulate MoE aux loss
        total_aux_loss = torch.zeros((), device=x.device, dtype=x.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Selective gradient checkpointing across JEDM2Block layers to constrain activation memory
                x, aux_loss = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                x, aux_loss = block(x, attention_mask=attention_mask)
            total_aux_loss = total_aux_loss + aux_loss

        # Normalize final hidden states: [B, S_total, hidden_dim]
        h_f = self.norm_f(x)

        # Slice text hidden states if visual tokens were prepended
        if num_visual_tokens > 0 and not return_all_tokens:
            h_text = h_f[:, num_visual_tokens:, :]
        else:
            h_text = h_f

        # System 2: Autoregressive LM rationale logits: [B, S_t, vocab_size]
        lm_logits = self.lm_head(h_text)

        # System 1: Dual-system action heads with visual grounding
        action_output = self.action_head(h_text, visual_tokens=visual_tokens)

        # JEPA World Model evaluation: connect directly to contextual backbone representations
        jepa_loss, jepa_metrics = None, None
        if self.jepa is not None and action_t is not None and obs_next is not None:
            if obs_t is not None:
                h_jepa_in = obs_t
            else:
                # Derive contextual backbone representation directly from recurrent stack
                if h_text.ndim == 3:
                    h_jepa_in = h_text.mean(dim=1)
                else:
                    h_jepa_in = h_text

            vt_t = visual_tokens_t if visual_tokens_t is not None else visual_tokens
            vt_next = visual_tokens_next
            _, _, jepa_loss, jepa_metrics = self.jepa(
                h_jepa_in,
                action_t,
                obs_next,
                visual_tokens_t=vt_t,
                visual_tokens_next=vt_next,
            )

        return JEDM2Output(
            lm_logits=lm_logits,
            action_output=action_output,
            hidden_states=h_text,
            aux_loss=total_aux_loss,
            jepa_loss=jepa_loss,
            jepa_metrics=jepa_metrics,
            visual_tokens=visual_tokens,
        )

    def step(
        self,
        token_id_or_embed: torch.Tensor,
        states: Optional[List[Tuple[Any, Any, Any]]] = None,
        pos: int = 0,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ActionOutput, List[Tuple[Any, Any, Any]]]:
        """
        Autoregressive multi-step recurrent rollout step.
        Propagates a single token through all stacked layers while preserving
        layer-wise Mamba rolling state buffers and MLA dynamic KV caches.
        Supports visual grounding on visual patch tokens.

        Args:
            token_id_or_embed: Single token ID tensor of shape [B] or [B, 1],
                               or token embedding tensor [B, D] or [B, 1, D].
            states: Optional list of per-layer state tuples:
                    [(ssm_state_l, conv_state_l, kv_cache_l), ...] for l in 0..num_layers-1.
            pos: Current sequence position (0-indexed).
            visual_tokens: Optional visual patch tokens [B, N_v, hidden_dim] for coordinate grounding.

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

        # System 1: Action outputs at timestep t with optional visual grounding
        action_output_t = self.action_head(h_t, visual_tokens=visual_tokens)

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
            if getattr(block, "ffn_type", "moe") == "moe" and hasattr(block, "moe") and hasattr(block.moe, "experts"):
                num_experts = block.moe.num_experts
                top_k = block.moe.top_k
                expert_params = sum(p.numel() for p in block.moe.experts.parameters())
                single_expert_params = expert_params // num_experts
                inactive_moe_params += (num_experts - top_k) * single_expert_params

        # JEPA target encoder is non-trainable EMA parameters
        target_encoder_params = 0
        if self.jepa is not None and hasattr(self.jepa, "target_encoder"):
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
