"""
JED-M2 Joint-Embedding Predictive Architecture (JEPA) Latent World Model.
Implements the Latent State Encoder, EMA Momentum Target Encoder,
Action-Conditioned Latent Predictor, and Composite Representation Loss.

STRICT BUILD RULE: Pure PyTorch without external compiled C++/CUDA packages.
"""

import copy
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.mla import RMSNorm


class LatentStateEncoder(nn.Module):
    """
    Online Latent State Encoder for JED-M2 JEPA.
    Maps high-dimensional observation features to an abstract, normalized latent space z_t.
    """

    def __init__(self, config: JEDM2Config, obs_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.obs_dim: int = obs_dim or config.hidden_dim
        self.latent_dim: int = config.jepa_latent_dim
        intermediate_dim: int = config.hidden_dim

        # Input normalization: [..., obs_dim]
        self.norm_in = RMSNorm(self.obs_dim, eps=config.layer_norm_eps)

        # Non-linear projection: [..., obs_dim] -> [..., intermediate_dim]
        self.fc1 = nn.Linear(self.obs_dim, intermediate_dim, bias=False)
        self.act = nn.SiLU()

        # Latent projection: [..., intermediate_dim] -> [..., latent_dim]
        self.fc2 = nn.Linear(intermediate_dim, self.latent_dim, bias=False)

        # Output latent normalization: [..., latent_dim]
        self.norm_out = RMSNorm(self.latent_dim, eps=config.layer_norm_eps)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode raw observation features into abstract latent representation.

        Args:
            obs: Observation tensor of shape [B, D_obs] or [B, L, D_obs].

        Returns:
            Latent state tensor of shape [B, D_latent] or [B, L, D_latent].
        """
        # [B, ..., D_obs] -> [B, ..., D_obs]
        h = self.norm_in(obs)
        # [B, ..., D_obs] -> [B, ..., intermediate_dim]
        h = self.act(self.fc1(h))
        # [B, ..., intermediate_dim] -> [B, ..., latent_dim]
        z = self.fc2(h)
        # [B, ..., latent_dim] -> [B, ..., latent_dim]
        z = self.norm_out(z)
        return z


class EMATargetEncoder(nn.Module):
    """
    Exponential Moving Average (EMA) Target Encoder for JED-M2 JEPA.
    Maintains a momentum-averaged copy of the online state encoder.
    All parameters are strictly frozen (requires_grad=False) and detached.
    """

    def __init__(self, online_encoder: LatentStateEncoder):
        super().__init__()
        # Create a deep copy of the online encoder architecture and weights
        self.encoder = copy.deepcopy(online_encoder)

        # Freeze all parameters to ensure no gradients propagate to target network
        for param in self.encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode target observation with stop-gradient.

        Args:
            obs: Target observation tensor of shape [B, D_obs] or [B, L, D_obs].

        Returns:
            Detached target latent representation z_{t+1}.
        """
        self.encoder.eval()
        z_target = self.encoder(obs)
        return z_target.detach()

    @torch.no_grad()
    def update(self, online_encoder: LatentStateEncoder, decay: float) -> None:
        """
        Update target encoder parameters using Polyak momentum averaging:
        θ_target <- decay * θ_target + (1 - decay) * θ_online

        Args:
            online_encoder: Current online LatentStateEncoder.
            decay: Momentum coefficient tau in [0, 1).
        """
        for target_param, online_param in zip(
            self.encoder.parameters(), online_encoder.parameters()
        ):
            target_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)


class LatentPredictor(nn.Module):
    """
    Action-Conditioned Latent Transition Predictor for JED-M2 JEPA.
    Predicts the expected next abstract state z_hat_{t+1} from current state z_t and action a_t.
    """

    def __init__(self, config: JEDM2Config):
        super().__init__()
        self.config = config
        self.latent_dim: int = config.jepa_latent_dim
        self.action_dim: int = config.jepa_action_dim
        intermediate_dim: int = config.hidden_dim

        # Action projection / embedding layer supporting both dense features and discrete IDs
        self.action_proj = nn.Linear(self.action_dim, self.action_dim, bias=False)

        # Discrete action embedding table for discrete token actions
        self.action_embed = nn.Embedding(config.vocab_size, self.action_dim)

        # Predictor MLP with residual transition dynamics
        # [B, latent_dim + action_dim] -> [B, intermediate_dim]
        self.fc1 = nn.Linear(self.latent_dim + self.action_dim, intermediate_dim, bias=False)
        self.act = nn.SiLU()
        # [B, intermediate_dim] -> [B, latent_dim]
        self.fc2 = nn.Linear(intermediate_dim, self.latent_dim, bias=False)

        # Residual normalization
        self.norm_res = RMSNorm(self.latent_dim, eps=config.layer_norm_eps)

    def _prepare_action(self, action: torch.Tensor) -> torch.Tensor:
        """Converts discrete action IDs or dense action vectors to action_dim representation."""
        if action.dtype in (torch.int32, torch.int64):
            # Discrete action ID: [B] or [B, L] -> [B, action_dim] or [B, L, action_dim]
            return self.action_embed(action)
        elif action.shape[-1] == self.action_dim:
            # Already matches action_dim: [..., action_dim]
            return self.action_proj(action)
        else:
            raise ValueError(
                f"Action tensor last dim {action.shape[-1]} does not match action_dim {self.action_dim}"
            )

    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent representation conditioned on current state and action.

        Args:
            z_t: Current latent state tensor of shape [B, D_latent] or [B, L, D_latent].
            action: Action tensor (discrete IDs or dense features [B, D_action] or [B, L, D_action]).

        Returns:
            Predicted next latent state tensor z_hat_{t+1} of shape [B, D_latent] or [B, L, D_latent].
        """
        # Embed/project action: [..., action_dim]
        a = self._prepare_action(action)

        # Concatenate latent state and action: [..., latent_dim + action_dim]
        za = torch.cat([z_t, a], dim=-1)

        # Compute state transition delta Δz: [..., latent_dim]
        delta_z = self.fc2(self.act(self.fc1(za)))

        # Residual update: z_hat_{t+1} = RMSNorm(z_t + Δz)
        z_next_pred = self.norm_res(z_t + delta_z)
        return z_next_pred


class JEPAWorldModel(nn.Module):
    """
    Unified JEPA World Model encapsulating:
    1. Online LatentStateEncoder (z_t = Encoder(o_t))
    2. EMA TargetEncoder (z_{t+1} = TargetEncoder(o_{t+1}) with stop_gradient)
    3. LatentPredictor (z_hat_{t+1} = Predictor(z_t, a_t))
    4. Composite Representation Loss (Smooth L1 + Cosine Distance)
    """

    def __init__(self, config: JEDM2Config, obs_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.obs_dim: int = obs_dim or config.hidden_dim
        self.latent_dim: int = config.jepa_latent_dim
        self.ema_decay: float = config.jepa_ema_decay
        self.cos_weight: float = config.jepa_loss_cos_weight

        # 1. Online state encoder
        self.encoder = LatentStateEncoder(config, obs_dim=self.obs_dim)

        # 2. EMA target encoder (frozen)
        self.target_encoder = EMATargetEncoder(self.encoder)

        # 3. Action-conditioned transition predictor
        self.predictor = LatentPredictor(config)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encodes observation using the online encoder."""
        return self.encoder(obs)

    def encode_target(self, obs_next: torch.Tensor) -> torch.Tensor:
        """Encodes target observation using the detached EMA target encoder."""
        return self.target_encoder(obs_next)

    def predict(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predicts expected next latent state given current state and action."""
        return self.predictor(z_t, action)

    def compute_loss(
        self, z_pred: torch.Tensor, z_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes composite JEPA loss:
        L = SmoothL1(z_pred, sg(z_target)) + lambda_cos * (1 - cosine_similarity(z_pred, sg(z_target)))

        Args:
            z_pred: Predicted latent states [..., latent_dim].
            z_target: Target latent states [..., latent_dim] (stop-gradient).

        Returns:
            Tuple of:
            - total_loss: Scalar composite loss tensor.
            - metrics: Dictionary of detached sub-loss metrics.
        """
        # Smooth L1 / Huber loss
        loss_smooth_l1 = F.smooth_l1_loss(z_pred, z_target)

        # Cosine similarity distance: 1 - cos(z_pred, z_target)
        cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean()
        loss_cos = 1.0 - cos_sim

        # Total composite loss
        total_loss = loss_smooth_l1 + self.cos_weight * loss_cos

        metrics = {
            "loss_total": total_loss.detach(),
            "loss_smooth_l1": loss_smooth_l1.detach(),
            "loss_cos_dist": loss_cos.detach(),
            "cos_sim": cos_sim.detach(),
        }
        return total_loss, metrics

    def forward(
        self,
        obs_t: torch.Tensor,
        action_t: torch.Tensor,
        obs_next: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass for self-supervised JEPA transition modeling.

        Args:
            obs_t: Current observation tensor [B, D_obs].
            action_t: Executed action tensor [B, D_action] or discrete IDs [B].
            obs_next: Next observed state tensor [B, D_obs].

        Returns:
            Tuple of:
            - z_pred: Predicted next latent state z_hat_{t+1}.
            - z_target: Stop-gradient target latent state z_{t+1}.
            - loss: Composite representation consistency loss.
            - metrics: Detailed loss metrics dictionary.
        """
        # 1. Online encode current observation: [B, D_latent]
        z_t = self.encode(obs_t)

        # 2. Predict next latent state: [B, D_latent]
        z_pred = self.predict(z_t, action_t)

        # 3. Target encode next observation with detached gradients: [B, D_latent]
        z_target = self.encode_target(obs_next)

        # 4. Compute composite consistency loss
        loss, metrics = self.compute_loss(z_pred, z_target)

        return z_pred, z_target, loss, metrics

    def update_target_network(self) -> None:
        """Applies momentum update to target encoder parameters."""
        self.target_encoder.update(self.encoder, self.ema_decay)
