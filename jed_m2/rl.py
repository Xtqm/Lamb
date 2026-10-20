"""
JED-M2 Trajectory Policy Optimization Scaffolding.
Implements:
1. Direct Policy Optimization (DPO) for offline paired trajectory preference learning.
2. Group Relative Policy Optimization (GRPO) for online task outcome reward optimization.

Calibrated for Large Action Models executing multi-turn digital agent tasks.
"""

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
    config: Optional[GRPOConfig] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes Group Relative Policy Optimization (GRPO) loss for groups of sampled agent rollouts.

    Advantage estimation is group-normalized across rollouts for the same prompt:
        A_i = (r_i - mean({r})) / (std({r}) + eps)

    The surrogate objective clips probability ratios and adds token-level KL penalties:
        L_GRPO = - 1/G sum_{i=1}^G 1/|T_i| sum_{t} [ min(r_{i,t} A_i, clip(r_{i,t}, 1-eps, 1+eps) A_i)
                                                    - beta_kl * D_KL(pi_theta || pi_ref) ]

    Args:
        policy_token_logps: Token log-probabilities from active policy [B, S].
        old_token_logps: Token log-probabilities from rollout policy [B, S].
        ref_token_logps: Token log-probabilities from frozen reference policy [B, S].
        rewards: Scalar trajectory outcome rewards [B].
        mask: Valid action/thought token mask [B, S].
        config: Optional GRPO configuration.

    Returns:
        Tuple of (scalar_grpo_loss, metrics_dictionary).
    """
    cfg = config if config is not None else GRPOConfig()
    batch_size = rewards.shape[0]
    group_size = cfg.group_size

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

    # 2. Probability ratio r_{i, t} = exp(log pi_theta - log pi_old): [B, S]
    # [B, S] - [B, S] -> [B, S]
    log_ratio = policy_token_logps - old_token_logps
    ratio = torch.exp(log_ratio)

    # 3. Clipped surrogate policy gradient objective: [B, S]
    adv_expanded = advantages.unsqueeze(1)  # [B, 1]
    surr1 = ratio * adv_expanded
    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_expanded
    policy_loss = -torch.min(surr1, surr2)  # [B, S]

    # 4. Token-level KL divergence penalty against reference policy (Schulman unbiased estimator):
    # D_KL(pi || pi_ref) approx exp(log_ref - log_pi) - (log_ref - log_pi) - 1
    # Clamped to [-5.0, 5.0] to prevent exponential numerical explosion under bfloat16
    kl_ratio = (ref_token_logps - policy_token_logps).clamp(min=-5.0, max=5.0)
    kl_penalty = torch.exp(kl_ratio) - kl_ratio - 1.0  # [B, S]

    # 5. Composite token objective: [B, S]
    total_token_loss = policy_loss + cfg.beta_kl * kl_penalty

    # Aggregate over masked action/thought tokens
    mask_f = mask.float()
    active_tokens = mask_f.sum().clamp(min=1.0)
    loss = (total_token_loss * mask_f).sum() / active_tokens

    with torch.no_grad():
        mean_kl = (kl_penalty * mask_f).sum() / active_tokens
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes GRPO loss for grouped trajectory rollouts.

        Args:
            input_ids: Trajectory tokens of shape [B, S].
            targets: Target action tokens of shape [B, S].
            mask: Active token mask of shape [B, S].
            old_token_logps: Rollout policy token log probabilities [B, S].
            rewards: Trajectory outcome rewards [B].

        Returns:
            Tuple of (scalar_grpo_loss, metrics_dictionary).
        """
        # Active policy forward
        policy_out = self.model(input_ids=input_ids)
        policy_token_logps, _ = compute_token_log_probs(
            policy_out.lm_logits, targets, mask
        )

        # Reference policy forward
        with torch.no_grad():
            ref_out = self.ref_model(input_ids=input_ids)
            ref_token_logps, _ = compute_token_log_probs(
                ref_out.lm_logits, targets, mask
            )

        return calculate_grpo_loss(
            policy_token_logps=policy_token_logps,
            old_token_logps=old_token_logps,
            ref_token_logps=ref_token_logps,
            rewards=rewards,
            mask=mask,
            config=self.config,
        )
