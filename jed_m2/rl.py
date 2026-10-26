"""
JED-M2 Trajectory Policy Optimization Scaffolding.
Implements:
1. Direct Policy Optimization (DPO) for offline paired trajectory preference learning.
2. Group Relative Policy Optimization (GRPO) for online task outcome reward optimization.

Calibrated for Large Action Models executing multi-turn digital agent tasks.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.model import JEDM2Model, JEDM2Output


@dataclass
class DPOConfig:
    """Hyperparameters for Direct Policy Optimization (DPO)."""

    beta: float = 0.1  # Implicit reward scaling temperature (tau = 1 / beta)
    label_smoothing: float = 0.0  # Optional label smoothing for soft preference labels
    loss_type: str = "sigmoid"  # "sigmoid" (standard DPO) or "ipo" (Identity Policy Optimization)


@dataclass
class GRPOConfig:
    """Hyperparameters for Group Relative Policy Optimization (GRPO)."""

    group_size: int = 4  # Number of sampled rollout trajectories per instruction prompt
    clip_eps: float = 0.2  # PPO-style surrogate objective clipping threshold epsilon
    beta_kl: float = 0.04  # KL divergence penalty coefficient against reference model
    temperature: float = 0.7  # Rollout generation sampling temperature
    lambda_coord: float = 0.5  # Weighting for continuous spatial coordinate policy gradient ratio
    coord_sigma: float = 0.05  # Gaussian spatial exploration std dev
    coord_kl_beta: float = 0.01  # Spatial coordinate KL regularization weight
    entropy_coeff: float = 0.01  # Token policy entropy regularization bonus weight


class CurriculumScheduler:
    """
    Curriculum Learning Scheduler for Multi-Turn Agent Workflows:
    - Level 1: 1-step atomic actions (click, type, scroll, navigate), max_turns=4
    - Level 2: 2-3 step compound workflows (search -> select -> add to cart), max_turns=6
    - Level 3: 5-8 step long-horizon workflows (full checkout, multi-step form wizard, cloud admin), max_turns=8

    Tracks moving success rate over a rolling window.
    Advances to next level when moving success rate >= success_threshold (default 0.80).
    """

    LEVEL_CONFIGS = {
        1: {"name": "Level 1: Focused (4 Turns)", "max_turns": 4, "complexity": 1},
        2: {"name": "Level 2: Compound (6 Turns)", "max_turns": 6, "complexity": 2},
        3: {"name": "Level 3: Long-Horizon (8 Turns)", "max_turns": 8, "complexity": 3},
    }

    def __init__(
        self,
        initial_level: int = 1,
        success_threshold: float = 0.80,
        window_size: int = 10,
        min_episodes_per_level: int = 8,
    ):
        self.current_level = initial_level
        self.success_threshold = success_threshold
        self.window_size = window_size
        self.min_episodes_per_level = min_episodes_per_level
        self.history: List[float] = []
        self.episodes_in_current_level = 0

    @property
    def max_turns(self) -> int:
        return self.LEVEL_CONFIGS.get(self.current_level, {}).get("max_turns", 8)

    @property
    def level_name(self) -> str:
        return self.LEVEL_CONFIGS.get(self.current_level, {}).get("name", f"Level {self.current_level}")

    @property
    def moving_success_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def record_outcomes(self, successes: List[bool]) -> bool:
        """
        Records episode success outcomes and checks for tier promotion.
        Returns True if curriculum advanced to a higher level.
        """
        for s in successes:
            self.history.append(1.0 if s else 0.0)
            if len(self.history) > self.window_size:
                self.history.pop(0)
            self.episodes_in_current_level += 1

        if (
            self.current_level < 3
            and self.episodes_in_current_level >= self.min_episodes_per_level
            and self.moving_success_rate >= self.success_threshold
        ):
            self.current_level += 1
            self.episodes_in_current_level = 0
            self.history.clear()
            return True
        return False


class ExplorationScheduler:
    """
    Manages temperature annealing and adaptive entropy bonus scheduling to prevent policy entropy collapse.
    """

    def __init__(
        self,
        initial_temp: float = 0.7,
        min_temp: float = 0.2,
        temp_decay: float = 0.99,
        initial_entropy_coeff: float = 0.01,
        min_entropy_coeff: float = 0.001,
        entropy_decay: float = 0.99,
    ):
        self.temperature = initial_temp
        self.min_temp = min_temp
        self.temp_decay = temp_decay
        self.entropy_coeff = initial_entropy_coeff
        self.min_entropy_coeff = min_entropy_coeff
        self.entropy_decay = entropy_decay

    def step(self) -> Tuple[float, float]:
        """Steps temperature and entropy coefficient decay."""
        self.temperature = max(self.min_temp, self.temperature * self.temp_decay)
        self.entropy_coeff = max(self.min_entropy_coeff, self.entropy_coeff * self.entropy_decay)
        return self.temperature, self.entropy_coeff


def compute_coordinate_log_probs(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    sigma: float = 0.05,
) -> torch.Tensor:
    """
    Computes 2D continuous Gaussian log-probabilities:
        log pi_coord(c | mu, sigma) = -0.5 * sum_{k in {x, y}} [ ((c_k - mu_k)^2 / sigma^2) + ln(2 * pi * sigma^2) ]

    Args:
        pred_coords: Mean predicted coordinates [..., 2] in [0, 1].
        target_coords: Sampled action coordinates [..., 2] in [0, 1].
        sigma: Standard deviation of Gaussian exploration policy.

    Returns:
        Tensor of coordinate log-probabilities of shape [...].
    """
    var = max(1e-6, sigma ** 2)
    diff_sq = (target_coords - pred_coords).pow(2)  # [..., 2]
    # Sum over x and y dimensions
    log_probs = -0.5 * (diff_sq.sum(dim=-1) / var + 2.0 * math.log(2.0 * math.pi * var))
    return log_probs


def sample_continuous_coordinates(
    mu_coords: torch.Tensor,
    sigma: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples continuous 2D coordinates from a Gaussian spatial policy centered at mu_coords.

    Args:
        mu_coords: Mean predicted coordinates [..., 2] in [0, 1].
        sigma: Standard deviation of exploration.

    Returns:
        Tuple of (sampled_coords in [0, 1], log_probs).
    """
    noise = torch.randn_like(mu_coords) * sigma
    sampled = (mu_coords + noise).clamp(0.0, 1.0)
    logps = compute_coordinate_log_probs(mu_coords, sampled, sigma=sigma)
    return sampled, logps


def compute_point_box_iou(
    coords: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
    point_radius: float = 0.03,
) -> float:
    """
    Computes 2D Intersection-over-Union (IoU) between a point radius buffer and target bounding box.
    """
    px, py = coords
    bx1, by1, bx2, by2 = bbox
    px1, py1 = max(0.0, px - point_radius), max(0.0, py - point_radius)
    px2, py2 = min(1.0, px + point_radius), min(1.0, py + point_radius)

    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    p_area = (px2 - px1) * (py2 - py1)
    b_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
    union = p_area + b_area - intersection
    return float(intersection / max(1e-8, union))


def compute_shaped_reward(
    goal_satisfied: bool,
    milestones_count: int = 0,
    turn_count: int = 1,
    iou_sum: float = 0.0,
    repeated_action_count: int = 0,
    completion_bonus: float = 2.0,
    milestone_reward: float = 0.5,
    iou_weight: float = 0.3,
    step_penalty: float = 0.05,
    looping_penalty: float = 0.5,
) -> float:
    """
    Dense multi-factor reward shaping for digital LAM agents:
    R = (+2.0 * completion) + (+0.5 * milestones) + (+0.3 * IoU) - (0.05 * steps) - (0.5 * loops)
    """
    r_task = completion_bonus if goal_satisfied else 0.0
    r_milestone = milestone_reward * milestones_count
    r_iou = iou_weight * max(0.0, min(1.0, iou_sum))
    r_step = -step_penalty * max(0, turn_count)
    r_loop = -looping_penalty * max(0, repeated_action_count)
    return round(r_task + r_milestone + r_iou + r_step + r_loop, 4)


def compute_token_log_probs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes token-level and sequence-aggregated log probabilities from logits and target IDs.

    Args:
        logits: Logit tensor of shape [B, S, vocab_size].
        targets: Target token IDs of shape [B, S].
        mask: Optional binary or boolean mask of shape [B, S].

    Returns:
        Tuple of:
        - token_logps: Token log-probabilities of shape [B, S].
        - seq_logps: Sequence-aggregated log-probabilities of shape [B].
    """
    # [B, S, vocab_size] -> log-probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Gather log-probabilities at target indices: [B, S]
    # Handle possible negative target indices (e.g., -100 padding) safely
    safe_targets = targets.clamp(min=0)
    token_logps = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)

    if mask is not None:
        token_logps = token_logps * mask.float()
        token_counts = mask.float().sum(dim=-1).clamp(min=1.0)
        seq_logps = token_logps.sum(dim=-1) / token_counts
    else:
        seq_logps = token_logps.mean(dim=-1)

    return token_logps, seq_logps


def compute_action_type_log_probs(
    action_logits: torch.Tensor,
    action_targets: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Computes selected discrete action log-probabilities at action positions."""
    log_probs = F.log_softmax(action_logits.float(), dim=-1)
    safe_targets = action_targets.clamp(min=0)
    selected = log_probs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    return selected * action_mask.float()


def calculate_dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    config: Optional[DPOConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Direct Policy Optimization (DPO) loss from sequence log-probabilities of
    paired winning and losing agent trajectories:

        L_DPO(theta; pi_ref) = -E_{(tau_w, tau_l)} [ log sigma( beta * log(pi_theta(tau_w)/pi_ref(tau_w))
                                                               - beta * log(pi_theta(tau_l)/pi_ref(tau_l)) ) ]

    Args:
        policy_chosen_logps: Active policy log-probabilities for winning trajectories [B].
        policy_rejected_logps: Active policy log-probabilities for losing trajectories [B].
        reference_chosen_logps: Frozen reference model log-probabilities for winning trajectories [B].
        reference_rejected_logps: Frozen reference model log-probabilities for losing trajectories [B].
        config: Optional DPO configuration.

    Returns:
        Tuple of (scalar_dpo_loss, metrics_dictionary).
    """
    cfg = config if config is not None else DPOConfig()
    beta = cfg.beta

    # Compute policy and reference log-ratios: [B]
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # Implicit reward difference logits: [B]
    logits = pi_logratios - ref_logratios

    if cfg.loss_type == "sigmoid":
        # Standard Bradley-Terry DPO objective
        losses = -F.logsigmoid(beta * logits)
    elif cfg.loss_type == "ipo":
        # Identity Policy Optimization (IPO) quadratic objective: (logits - 1 / (2 * beta))^2
        losses = (logits - 1.0 / (2.0 * beta)) ** 2
    else:
        raise ValueError(f"Unknown DPO loss_type: {cfg.loss_type}")

    loss = losses.mean()

    # Calculate detached implicit rewards and diagnostics
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

    metrics = {
        "loss_dpo": float(loss.item()),
        "reward_margin": float(reward_margin),
        "dpo_accuracy": float(accuracy),
        "chosen_reward_mean": float(chosen_rewards.mean().item()),
        "rejected_reward_mean": float(rejected_rewards.mean().item()),
    }

    return loss, metrics


def calculate_grpo_loss(
    policy_token_logps: torch.Tensor,
    old_token_logps: torch.Tensor,
    ref_token_logps: torch.Tensor,
    rewards: torch.Tensor,
    mask: torch.Tensor,
    policy_coord_logps: Optional[torch.Tensor] = None,
    old_coord_logps: Optional[torch.Tensor] = None,
    ref_coord_logps: Optional[torch.Tensor] = None,
    policy_logits: Optional[torch.Tensor] = None,
    config: Optional[GRPOConfig] = None,
    advantages: Optional[torch.Tensor] = None,
    policy_action_logps: Optional[torch.Tensor] = None,
    old_action_logps: Optional[torch.Tensor] = None,
    ref_action_logps: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Group Relative Policy Optimization (GRPO) loss with Joint Token and
    Continuous Spatial Coordinate Policy Gradient Optimization.

    Advantage estimation is group-normalized across rollouts for the same prompt:
        A_i = (r_i - mean({r})) / (std({r}) + eps)

    Joint ratio r_{i,t}:
        r_{i,t} = exp((log pi_token - log pi_old_token) + lambda_coord * (log pi_coord - log pi_old_coord))

    The surrogate objective clips probability ratios and adds reference KL divergence penalties:
        L_GRPO = - 1/G sum_{i=1}^G 1/|T_i| sum_{t} [ min(r_{i,t} A_i, clip(r_{i,t}, 1-eps, 1+eps) A_i)
                                                    - beta_kl * D_KL(pi || pi_ref) ]
    """
    cfg = config if config is not None else GRPOConfig()
    batch_size = rewards.shape[0]
    group_size = cfg.group_size

    if advantages is None:
        if batch_size % group_size != 0:
            raise ValueError(
                f"Batch size {batch_size} must be divisible by group_size {group_size}"
            )

        num_groups = batch_size // group_size

        # 1. Group advantage normalization: [num_groups, group_size]
        rewards_grouped = rewards.view(num_groups, group_size)
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        group_stds = rewards_grouped.std(dim=1, keepdim=True).clamp(min=1e-8)
        # [num_groups, group_size] -> [B]
        advantages = ((rewards_grouped - group_means) / group_stds).view(batch_size)
    elif advantages.shape != rewards.shape:
        raise ValueError(
            f"Precomputed advantages shape {tuple(advantages.shape)} must match "
            f"rewards shape {tuple(rewards.shape)}"
        )

    # 2. Joint Probability ratio r_{i, t}: [B, S]
    log_token_ratio = policy_token_logps - old_token_logps
    joint_log_ratio = log_token_ratio
    if policy_action_logps is not None and old_action_logps is not None:
        joint_log_ratio = joint_log_ratio + (policy_action_logps - old_action_logps)
    if policy_coord_logps is not None and old_coord_logps is not None:
        log_coord_ratio = policy_coord_logps - old_coord_logps
        joint_log_ratio = joint_log_ratio + cfg.lambda_coord * log_coord_ratio

    joint_log_ratio = joint_log_ratio.clamp(min=-5.0, max=5.0)
    ratio = torch.exp(joint_log_ratio)

    # 3. Clipped surrogate policy gradient objective: [B, S]
    adv_expanded = advantages.unsqueeze(1)  # [B, 1]
    surr1 = ratio * adv_expanded
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_expanded
    policy_loss = -torch.min(surr1, surr2)  # [B, S]

    # 4. Token & Coordinate KL divergence penalty against reference policy
    kl_ratio = (ref_token_logps - policy_token_logps).clamp(min=-5.0, max=5.0)
    kl_penalty = torch.exp(kl_ratio) - kl_ratio - 1.0  # [B, S]
    if policy_action_logps is not None and ref_action_logps is not None:
        action_kl_ratio = (ref_action_logps - policy_action_logps).clamp(min=-5.0, max=5.0)
        kl_penalty = kl_penalty + torch.exp(action_kl_ratio) - action_kl_ratio - 1.0

    if policy_coord_logps is not None and ref_coord_logps is not None:
        coord_kl = (policy_coord_logps - ref_coord_logps).abs().clamp(max=5.0)
        total_kl = kl_penalty + cfg.coord_kl_beta * coord_kl
    else:
        total_kl = kl_penalty

    # 5. Composite token & spatial objective: [B, S]
    total_token_loss = policy_loss + cfg.beta_kl * total_kl

    # Aggregate over masked action/thought tokens
    mask_f = mask.float()
    active_tokens = mask_f.sum().clamp(min=1.0)
    loss = (total_token_loss * mask_f).sum() / active_tokens

    # 6. Adaptive Token Policy Entropy Regularization Bonus
    mean_entropy = 0.0
    if policy_logits is not None and cfg.entropy_coeff > 0.0:
        log_probs = F.log_softmax(policy_logits, dim=-1)
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1)  # [B, S]
        mean_entropy = (token_entropy * mask_f).sum() / active_tokens
        loss = loss - cfg.entropy_coeff * mean_entropy

    with torch.no_grad():
        mean_kl = (total_kl * mask_f).sum() / active_tokens
        clip_fraction = (
            ((ratio < (1.0 - cfg.clip_eps)) | (ratio > (1.0 + cfg.clip_eps)))
            .float()
            * mask_f
        ).sum() / active_tokens

    metrics = {
        "loss_grpo": float(loss.item()),
        "mean_advantage": float(advantages.mean().item()),
        "mean_kl": float(mean_kl.item()),
        "clip_fraction": float(clip_fraction.item()),
        "mean_reward": float(rewards.mean().item()),
        "mean_entropy": float(mean_entropy if isinstance(mean_entropy, float) else mean_entropy.item()),
    }

    return loss, metrics


class DPOTrainer:
    """
    Direct Policy Optimization Trainer for JED-M2.
    Maintains active trainable model and frozen reference model.
    """

    def __init__(
        self,
        model: JEDM2Model,
        ref_model: Optional[JEDM2Model] = None,
        config: Optional[DPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config if config is not None else DPOConfig()

        # Determine device and dtype from active model parameters
        try:
            first_param = next(model.parameters())
            target_device = first_param.device
            target_dtype = first_param.dtype
        except StopIteration:
            target_device = torch.device("cpu")
            target_dtype = torch.float32

        # If reference model not provided, create a frozen clone
        if ref_model is None:
            self.ref_model = JEDM2Model(model.config).to(device=target_device, dtype=target_dtype)
            self.ref_model.load_state_dict(model.state_dict())
        else:
            self.ref_model = ref_model.to(device=target_device, dtype=target_dtype)

        # Strictly freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        self.optimizer = optimizer

    def compute_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_targets: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_targets: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes DPO loss over paired winning and losing agent sequences.

        Args:
            chosen_input_ids: Input tokens for winning trajectories [B, S_c].
            chosen_targets: Target tokens for winning trajectories [B, S_c].
            chosen_mask: Loss mask for winning trajectories [B, S_c].
            rejected_input_ids: Input tokens for losing trajectories [B, S_r].
            rejected_targets: Target tokens for losing trajectories [B, S_r].
            rejected_mask: Loss mask for losing trajectories [B, S_r].

        Returns:
            Tuple of (scalar_dpo_loss, metrics_dictionary).
        """
        # Active policy forward pass
        policy_chosen_out = self.model(input_ids=chosen_input_ids)
        policy_rejected_out = self.model(input_ids=rejected_input_ids)

        _, policy_chosen_logps = compute_token_log_probs(
            policy_chosen_out.lm_logits, chosen_targets, chosen_mask
        )
        _, policy_rejected_logps = compute_token_log_probs(
            policy_rejected_out.lm_logits, rejected_targets, rejected_mask
        )

        # Frozen reference model forward pass
        with torch.no_grad():
            ref_chosen_out = self.ref_model(input_ids=chosen_input_ids)
            ref_rejected_out = self.ref_model(input_ids=rejected_input_ids)

            _, ref_chosen_logps = compute_token_log_probs(
                ref_chosen_out.lm_logits, chosen_targets, chosen_mask
            )
            _, ref_rejected_logps = compute_token_log_probs(
                ref_rejected_out.lm_logits, rejected_targets, rejected_mask
            )

        return calculate_dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            reference_chosen_logps=ref_chosen_logps,
            reference_rejected_logps=ref_rejected_logps,
            config=self.config,
        )


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer for JED-M2.
    Computes group-relative policy updates from trajectory outcome rewards.
    """

    def __init__(
        self,
        model: JEDM2Model,
        ref_model: Optional[JEDM2Model] = None,
        config: Optional[GRPOConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.model = model
        self.config = config if config is not None else GRPOConfig()

        # Determine device and dtype from active model parameters
        try:
            first_param = next(model.parameters())
            target_device = first_param.device
            target_dtype = first_param.dtype
        except StopIteration:
            target_device = torch.device("cpu")
            target_dtype = torch.float32

        if ref_model is None:
            self.ref_model = JEDM2Model(model.config).to(device=target_device, dtype=target_dtype)
            self.ref_model.load_state_dict(model.state_dict())
        else:
            self.ref_model = ref_model.to(device=target_device, dtype=target_dtype)

        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

        self.optimizer = optimizer

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        old_token_logps: torch.Tensor,
        rewards: torch.Tensor,
        old_coord_logps: Optional[torch.Tensor] = None,
        target_coords: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        action_type_targets: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
        old_action_logps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes GRPO loss for grouped trajectory rollouts with optional joint spatial policy optimization.

        Args:
            input_ids: Trajectory tokens of shape [B, S].
            targets: Target action tokens of shape [B, S].
            mask: Active token mask of shape [B, S].
            old_token_logps: Rollout policy token log probabilities [B, S].
            rewards: Trajectory outcome rewards [B].
            old_coord_logps: Optional rollout policy coordinate log probabilities [B, S].
            target_coords: Optional target coordinate vectors [B, S, 2].
            advantages: Optional precomputed group advantages for micro-batched updates [B].
            action_type_targets: Optional discrete action IDs aligned to action-trigger tokens [B, S].
            action_mask: Optional mask identifying action-trigger positions [B, S].
            old_action_logps: Optional rollout action-type log probabilities [B, S].

        Returns:
            Tuple of (scalar_grpo_loss, metrics_dictionary).
        """
        # Active policy forward
        policy_out = self.model(input_ids=input_ids)
        policy_token_logps, _ = compute_token_log_probs(
            policy_out.lm_logits, targets, mask
        )

        policy_coord_logps = None
        ref_coord_logps = None
        if target_coords is not None and old_coord_logps is not None:
            policy_coord_logps = compute_coordinate_log_probs(
                policy_out.action_output.coordinates, target_coords, sigma=self.config.coord_sigma
            )

        policy_action_logps = None
        if action_type_targets is not None and action_mask is not None:
            policy_action_logps = compute_action_type_log_probs(
                policy_out.action_output.action_type_logits,
                action_type_targets,
                action_mask,
            )

        # Reference policy forward
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids)
            ref_token_logps, _ = compute_token_log_probs(
                ref_out.lm_logits, targets, mask
            )
            if target_coords is not None and old_coord_logps is not None:
                ref_coord_logps = compute_coordinate_log_probs(
                    ref_out.action_output.coordinates, target_coords, sigma=self.config.coord_sigma
                )
            ref_action_logps = None
            if action_type_targets is not None and action_mask is not None:
                ref_action_logps = compute_action_type_log_probs(
                    ref_out.action_output.action_type_logits,
                    action_type_targets,
                    action_mask,
                )

        return calculate_grpo_loss(
            policy_token_logps=policy_token_logps,
            old_token_logps=old_token_logps,
            ref_token_logps=ref_token_logps,
            rewards=rewards,
            mask=mask,
            policy_coord_logps=policy_coord_logps,
            old_coord_logps=old_coord_logps,
            ref_coord_logps=ref_coord_logps,
            policy_logits=policy_out.lm_logits,
            config=self.config,
            advantages=advantages,
            policy_action_logps=policy_action_logps,
            old_action_logps=old_action_logps,
            ref_action_logps=ref_action_logps,
        )
