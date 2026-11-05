"""
JED-M2 Joint-Embedding Predictive Architecture (JEPA) Latent World Model.
Implements the Latent State Encoder, EMA Momentum Target Encoder,
Action-Conditioned Latent Predictor, and Composite Representation Loss.

STRICT BUILD RULE: Pure PyTorch without external compiled C++/CUDA packages.
"""

import copy
from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.mla import RMSNorm


class LatentStateEncoder(nn.Module):
    """
    Online Latent State Encoder for JED-M2 JEPA.
    Maps high-dimensional observation features (and optional visual patch tokens)
    to an abstract, normalized latent space z_t.
    """

    def __init__(self, config: JEDM2Config, obs_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.obs_dim: int = obs_dim or config.hidden_dim
        self.latent_dim: int = config.jepa_latent_dim
        intermediate_dim: int = config.hidden_dim

        # Input normalization for dense observation: [..., obs_dim]
        self.norm_in = RMSNorm(self.obs_dim, eps=config.layer_norm_eps)

        # Dense observation projection: [..., obs_dim] -> [..., intermediate_dim]
        self.fc1 = nn.Linear(self.obs_dim, intermediate_dim, bias=False)
        self.act = nn.SiLU()

        # Multimodal visual patch projection & normalization
        self.norm_vis = RMSNorm(self.obs_dim, eps=config.layer_norm_eps)
        self.fc_vis = nn.Linear(self.obs_dim, intermediate_dim, bias=False)

        # Latent projection: [..., intermediate_dim] -> [..., latent_dim]
        self.fc2 = nn.Linear(intermediate_dim, self.latent_dim, bias=False)

        # Output latent normalization: [..., latent_dim]
        self.norm_out = RMSNorm(self.latent_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        obs: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode observation features into abstract latent representation.

        Args:
            obs: Observation tensor of shape [B, D_obs] or [B, L, D_obs].
            visual_tokens: Optional visual patch tokens [B, N_v, hidden_dim].

        Returns:
            Latent state tensor of shape [B, D_latent] or [B, L, D_latent].
        """
        # [B, ..., D_obs] -> [B, ..., D_obs]
        h_obs = self.norm_in(obs)
        # [B, ..., D_obs] -> [B, ..., intermediate_dim]
        h = self.fc1(h_obs)

        # Multimodal fusion if visual tokens provided
        if visual_tokens is not None:
            # Pool visual tokens spatially across patches: [B, N_v, D] -> [B, D]
            if visual_tokens.ndim == 3:
                vis_pooled = visual_tokens.mean(dim=1)
            else:
                vis_pooled = visual_tokens
            h_vis = self.fc_vis(self.norm_vis(vis_pooled))
            if h.ndim == 3 and h_vis.ndim == 2:
                h_vis = h_vis.unsqueeze(1)
            h = h + h_vis

        h = self.act(h)
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
    def forward(
        self,
        obs: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode target observation with stop-gradient.

        Args:
            obs: Target observation tensor of shape [B, D_obs] or [B, L, D_obs].
            visual_tokens: Optional target visual patch tokens [B, N_v, hidden_dim].

        Returns:
            Detached target latent representation z_{t+1}.
        """
        self.encoder.eval()
        z_target = self.encoder(obs, visual_tokens=visual_tokens)
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

        # Spatial coordinate projection for grounded action conditioning (x, y) in [0, 1]
        self.coord_proj = nn.Linear(2, self.action_dim, bias=False)

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

    def forward(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict next latent representation conditioned on current state, action, and optional coordinates.

        Args:
            z_t: Current latent state tensor of shape [B, D_latent] or [B, L, D_latent].
            action: Action tensor (discrete IDs or dense features [B, D_action] or [B, L, D_action]).
            coords: Optional continuous normalized coordinates [B, 2] or [B, L, 2].

        Returns:
            Predicted next latent state tensor z_hat_{t+1} of shape [B, D_latent] or [B, L, D_latent].
        """
        # Embed/project action: [..., action_dim]
        a = self._prepare_action(action)

        if coords is not None:
            c = coords.to(device=a.device, dtype=a.dtype)
            if c.ndim == 1 and a.ndim == 2:
                c = c.unsqueeze(0)
            a = a + self.coord_proj(c)

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

    def encode(
        self,
        obs: torch.Tensor,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes observation using the online encoder."""
        return self.encoder(obs, visual_tokens=visual_tokens)

    def encode_target(
        self,
        obs_next: torch.Tensor,
        visual_tokens_next: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes target observation using the detached EMA target encoder."""
        return self.target_encoder(obs_next, visual_tokens=visual_tokens_next)

    def predict(
        self,
        z_t: torch.Tensor,
        action: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predicts expected next latent state given current state, action, and optional coordinates."""
        return self.predictor(z_t, action, coords=coords)

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
        visual_tokens_t: Optional[torch.Tensor] = None,
        visual_tokens_next: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Full forward pass for self-supervised JEPA transition modeling.

        Args:
            obs_t: Current observation tensor [B, D_obs].
            action_t: Executed action tensor [B, D_action] or discrete IDs [B].
            obs_next: Next observed state tensor [B, D_obs].
            visual_tokens_t: Optional current visual patch tokens [B, N_v, hidden_dim].
            visual_tokens_next: Optional next visual patch tokens [B, N_v, hidden_dim].

        Returns:
            Tuple of:
            - z_pred: Predicted next latent state z_hat_{t+1}.
            - z_target: Stop-gradient target latent state z_{t+1}.
            - loss: Composite representation consistency loss.
            - metrics: Detailed loss metrics dictionary.
        """
        # 1. Online encode current observation: [B, D_latent]
        z_t = self.encode(obs_t, visual_tokens=visual_tokens_t)

        # 2. Predict next latent state: [B, D_latent]
        z_pred = self.predict(z_t, action_t)

        # 3. Target encode next observation with detached gradients: [B, D_latent]
        z_target = self.encode_target(obs_next, visual_tokens_next=visual_tokens_next)

        # 4. Compute composite consistency loss
        loss, metrics = self.compute_loss(z_pred, z_target)

        return z_pred, z_target, loss, metrics

    def update_target_network(self) -> None:
        """Applies momentum update to target encoder parameters."""
        self.target_encoder.update(self.encoder, self.ema_decay)


# ==============================================================================
# 5. Best-of-N JEPA Test-Time Search Engine
# ==============================================================================

@dataclass
class CandidateAction:
    """Hypothesis container representing a candidate action-coordinate pair."""

    action_type_id: int
    coordinates: Tuple[float, float]
    raw_score: float = 0.0
    argument: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchCandidateResult:
    """Detailed evaluation metrics for a single prospective candidate rollout."""

    action_type_id: int
    coordinates: Tuple[float, float]
    argument: str
    goal_dist: float
    loop_dist: float
    drift: float
    fused_score: float
    pruned: bool
    prune_reason: str = ""
    rdtree_reward: float = 0.0
    rdtree_diag: Optional[str] = None


@dataclass
class JEPASearchResult:
    """Structured decision output from test-time JEPA latent candidate rollout."""

    selected_action_id: int
    selected_coordinates: Tuple[float, float]
    selected_argument: str
    selected_score: float
    fallback_used: bool
    num_candidates_evaluated: int
    num_candidates_pruned: int
    latency_ms: float
    candidates: List[SearchCandidateResult] = field(default_factory=list)


class JEPASearchHarness:
    """
    Test-Time Model-Predictive Control (MPC) Search Harness powered by JEPA.
    Executes vectorized Best-of-N candidate action rollout in latent space:
    1. Batched transition prediction: z_hat_{t+1}^(k) = Predictor(z_t, a_k, x_k, y_k)
    2. Goal drift pruning: d(z_hat, z_goal) > d(z_t, z_goal) + delta_drift
    3. Historical loop elimination: max_tau sim(z_hat, z_{t-tau}) > 1 - eps_loop
    4. Fused value re-ranking with safe top-policy fallback
    Executes within < 10 ms on consumer GPU.
    """

    def __init__(
        self,
        jepa_model: JEPAWorldModel,
        drift_threshold: float = 0.15,
        loop_threshold: float = 0.05,
        plan_lambda: float = 1.0,
        spatial_beta: float = 2.0,
        rdtree_gamma: float = 1.0,
        rdtree_critic: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.jepa = jepa_model
        self.drift_threshold = drift_threshold
        self.loop_threshold = loop_threshold
        self.plan_lambda = plan_lambda
        self.spatial_beta = spatial_beta
        self.rdtree_gamma = rdtree_gamma
        self.rdtree_critic = rdtree_critic
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(jepa_model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        self.dtype = dtype

    def search(
        self,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        candidates: List[CandidateAction],
        latent_history: Optional[List[torch.Tensor]] = None,
        pointer_prior: Optional[Tuple[float, float]] = None,
        rdtree_critic: Optional[Any] = None,
        elements: Optional[List[Any]] = None,
    ) -> JEPASearchResult:
        """
        Evaluates candidate action hypotheses in a single batched JEPA latent rollout pass.

        Args:
            z_t: Current latent state tensor [1, D_latent] or [D_latent].
            z_goal: Target goal latent state tensor [1, D_latent] or [D_latent].
            candidates: List of CandidateAction instances (N <= 16).
            latent_history: Optional list of past visited latent tensors [z_{t-H}, ..., z_{t-1}].
            pointer_prior: Optional predicted pointer coordinates (x_pred, y_pred).

        Returns:
            JEPASearchResult containing the chosen action, scores, and pruning telemetry.
        """
        start_time = time.perf_counter()

        if not candidates:
            raise ValueError("Candidates list must contain at least one CandidateAction.")

        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(device=self.device, dtype=self.dtype)

        if z_goal.ndim == 1:
            z_goal = z_goal.unsqueeze(0)
        z_goal = z_goal.to(device=self.device, dtype=self.dtype)

        curr_goal_dist = float(1.0 - F.cosine_similarity(z_t, z_goal, dim=-1).clamp(-1.0, 1.0).item())

        M = len(candidates)
        z_batch = z_t.expand(M, -1)
        act_tensor = torch.tensor([c.action_type_id for c in candidates], device=self.device, dtype=torch.long)
        coords_tensor = torch.tensor([c.coordinates for c in candidates], device=self.device, dtype=self.dtype)

        # 1. Vectorized 1-step lookahead rollout in JEPA latent space
        with torch.no_grad():
            z_pred_batch = self.jepa.predict(z_batch, act_tensor, coords=coords_tensor)

        # 2. Compute goal cosine distances for all candidates in parallel
        sims = F.cosine_similarity(z_pred_batch, z_goal.expand(M, -1), dim=-1).clamp(-1.0, 1.0)
        goal_dists = (1.0 - sims).tolist()

        # 3. Compute loop distance against historical latent states
        hist_tensors = [h for h in (latent_history or [])[-8:] if h is not None]
        if hist_tensors:
            hist_stack = torch.cat([h if h.ndim == 2 else h.unsqueeze(0) for h in hist_tensors], dim=0)
            hist_stack = hist_stack.to(device=self.device, dtype=self.dtype)
            hist_sims = F.cosine_similarity(
                z_pred_batch.unsqueeze(1), hist_stack.unsqueeze(0), dim=-1
            ).clamp(-1.0, 1.0)
            min_loop_dists = (1.0 - hist_sims.max(dim=-1).values).tolist()
        else:
            min_loop_dists = [1.0] * M

        # 4. Pruning and score evaluation
        evaluated_candidates: List[SearchCandidateResult] = []
        valid_candidates: List[Tuple[float, int, Tuple[float, float], str]] = []
        all_candidates_scored: List[Tuple[float, int, Tuple[float, float], str]] = []

        rx, ry = pointer_prior if pointer_prior is not None else (0.5, 0.5)
        critic = rdtree_critic if rdtree_critic is not None else self.rdtree_critic

        for m in range(M):
            cand = candidates[m]
            cx, cy = cand.coordinates
            g_dist = float(goal_dists[m])
            loop_dist = float(min_loop_dists[m])
            drift = g_dist - curr_goal_dist

            spatial_dist = math.hypot(cx - rx, cy - ry)

            # Symbolic RDTree Process Supervision
            rdtree_reward = 0.0
            rdtree_diag: Optional[str] = None
            rdtree_pruned = False
            rdtree_prune_reason = ""

            if critic is not None:
                r_eval = critic.evaluate_action(
                    action_type=cand.action_type_id,
                    coordinates=cand.coordinates,
                    argument=cand.argument,
                    elements=elements,
                )
                rdtree_reward = float(r_eval.process_reward)
                rdtree_diag = r_eval.diagnostic.value if hasattr(r_eval.diagnostic, "value") else str(r_eval.diagnostic)
                if r_eval.is_critical_violation or rdtree_reward <= -0.8:
                    rdtree_pruned = True
                    rdtree_prune_reason = f"rdtree ({rdtree_diag}: {r_eval.message})"

            # Fused candidate utility: policy logit - spatial - goal_dist + gamma * rdtree_reward
            fused_score = (
                cand.raw_score
                - self.spatial_beta * spatial_dist
                - self.plan_lambda * g_dist
                + self.rdtree_gamma * rdtree_reward
            )

            pruned = False
            prune_reason = ""
            if rdtree_pruned:
                pruned = True
                prune_reason = rdtree_prune_reason
            elif drift > self.drift_threshold:
                pruned = True
                prune_reason = f"drift ({drift:.3f} > {self.drift_threshold:.3f})"
            elif loop_dist < self.loop_threshold:
                pruned = True
                prune_reason = f"loop ({loop_dist:.3f} < {self.loop_threshold:.3f})"

            eval_res = SearchCandidateResult(
                action_type_id=cand.action_type_id,
                coordinates=cand.coordinates,
                argument=cand.argument,
                goal_dist=g_dist,
                loop_dist=loop_dist,
                drift=drift,
                fused_score=fused_score,
                pruned=pruned,
                prune_reason=prune_reason,
                rdtree_reward=rdtree_reward,
                rdtree_diag=rdtree_diag,
            )
            evaluated_candidates.append(eval_res)
            all_candidates_scored.append((fused_score, cand.action_type_id, cand.coordinates, cand.argument))

            if not pruned:
                valid_candidates.append((fused_score, cand.action_type_id, cand.coordinates, cand.argument))

        # 5. Selection with safe fallback guarantee
        if valid_candidates:
            valid_candidates.sort(key=lambda x: x[0], reverse=True)
            sel_score, sel_act_id, sel_coords, sel_arg = valid_candidates[0]
            fallback_used = False
        else:
            all_candidates_scored.sort(key=lambda x: x[0], reverse=True)
            sel_score, sel_act_id, sel_coords, sel_arg = all_candidates_scored[0]
            fallback_used = True

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return JEPASearchResult(
            selected_action_id=sel_act_id,
            selected_coordinates=sel_coords,
            selected_argument=sel_arg,
            selected_score=float(sel_score),
            fallback_used=fallback_used,
            num_candidates_evaluated=M,
            num_candidates_pruned=M - len(valid_candidates),
            latency_ms=latency_ms,
            candidates=evaluated_candidates,
        )

    def rank_candidates(
        self,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        candidates: List[CandidateAction],
        latent_history: Optional[List[torch.Tensor]] = None,
        pointer_prior: Optional[Tuple[float, float]] = None,
        rdtree_critic: Optional[Any] = None,
        elements: Optional[List[Any]] = None,
    ) -> JEPASearchResult:
        """
        Ranks prospective candidate actions using JEPA latent rollout and RDTree supervision.
        Alias for search().
        """
        return self.search(
            z_t=z_t,
            z_goal=z_goal,
            candidates=candidates,
            latent_history=latent_history,
            pointer_prior=pointer_prior,
            rdtree_critic=rdtree_critic,
            elements=elements,
        )
