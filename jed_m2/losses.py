"""
JED-M2 Unified Composite Loss Harness.
Calibrated for multi-task optimization across the Large Action Model (LAM):
- Reasoning loss (L_reason): Autoregressive cross-entropy on thought rationale tokens.
- Action type loss (L_type): Multi-class cross-entropy on discrete tool categories.
- Coordinate loss (L_coord): Smooth L1 / Huber regression on continuous screen coordinates in [0, 1].
- Argument loss (L_arg): Masked cross-entropy on argument payloads.
- JEPA latent loss (L_jepa): Smooth L1 + cosine distance representation alignment.
- MoE load-balancing loss (L_aux): Routing uniformity penalty from SparseTop2MoE.

STRICT CONVENTION: Pure PyTorch without external compiled C++/CUDA packages or Hugging Face pipelines.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.dataset import TrajectoryBatch
from jed_m2.model import JEDM2Output


@dataclass
class JEDM2LossConfig:
    """Configuration hyperparameters and calibrated weights for JED-M2 Composite Loss."""

    # Loss component weighting coefficients calibrated for balanced gradient magnitudes
    reason_weight: float = 1.0  # Weight for System 2 rationale NLL
    gate_weight: float = 1.0  # Weight for action gate emission binary classification
    action_type_weight: float = 2.0  # Weight for discrete primitive classification
    coord_weight: float = 10.0  # Weight for continuous (x, y) spatial coordinates
    arg_weight: float = 2.0  # Weight for argument token prediction (calibrated for TYPE argument generation)
    jepa_weight: float = 1.0  # Weight for JEPA latent world model consistency
    aux_weight: float = 1.0  # Weight for MoE load-balancing auxiliary loss

    # Regression and classification thresholds
    huber_beta: float = 0.05  # Beta threshold for Smooth L1 / Huber coordinate loss
    ignore_index: int = -100  # Index to ignore in cross-entropy evaluations
    normalize_token_loss: bool = True  # Normalize token cross-entropy by log(vocab_size) to balance gradient scales


class JEDM2CompositeLoss(nn.Module):
    """
    Unified multi-task composite loss harness for JED-M2.
    Evaluates, balances, and reports all sub-losses with robust zero-mask guardrails
    to prevent NaNs or Infinities under sparse action types or variable-length batches.
    """

    def __init__(
        self,
        config: JEDM2Config,
        loss_config: Optional[JEDM2LossConfig] = None,
    ):
        super().__init__()
        self.config = config
        self.loss_config = loss_config or JEDM2LossConfig()

    def forward(
        self,
        model_output: JEDM2Output,
        batch: TrajectoryBatch,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite multi-task loss and logging metrics.

        Args:
            model_output: Forward output container from JEDM2Model.
            batch: Collated TrajectoryBatch containing labels, coordinates, and masks.

        Returns:
            Tuple of:
            - total_loss: Differentiable scalar tensor for backpropagation.
            - metrics: Comprehensive dictionary of detached scalar floats for logging.
        """
        device = model_output.lm_logits.device
        dtype = model_output.lm_logits.dtype
        cfg = self.loss_config

        # ======================================================================
        # 1. Reasoning Loss: Autoregressive NLL on Thought Tokens
        # ======================================================================
        # model_output.lm_logits: [B, S, vocab_size]
        # batch.labels: [B, S]
        # batch.reason_loss_mask: [B, S] (bool)
        lm_logits = model_output.lm_logits
        b_sz, seq_len, vocab_size = lm_logits.shape

        # Shift for next-token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()  # [B, S-1, vocab_size]
        shift_labels = batch.labels[:, 1:].contiguous()  # [B, S-1]
        shift_mask = batch.reason_loss_mask[:, 1:].contiguous()  # [B, S-1]

        if shift_mask.any():
            # Filter labels by thought mask
            masked_labels = torch.where(
                shift_mask, shift_labels, torch.full_like(shift_labels, cfg.ignore_index)
            )
            # [B * (S-1), vocab_size] vs [B * (S-1)]
            raw_loss_reason = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                masked_labels.view(-1),
                ignore_index=cfg.ignore_index,
            )
            # Scale-normalize cross-entropy by log(vocab_size) to align with Huber and discrete CE magnitudes
            if getattr(cfg, "normalize_token_loss", True):
                loss_reason = raw_loss_reason / math.log(max(2, vocab_size))
            else:
                loss_reason = raw_loss_reason
        else:
            loss_reason = 0.0 * lm_logits.sum()

        # ======================================================================
        # 2. Action Gate Loss: Binary Cross-Entropy on Action Emission vs Thinking
        # ======================================================================
        # gate_logits: [B, S, 1]
        gate_logits = model_output.action_output.gate_logits.squeeze(-1)  # [B, S]
        gate_targets = torch.zeros_like(gate_logits)
        gate_mask = torch.zeros((b_sz, seq_len), dtype=torch.bool, device=device)

        # Active on thoughts (target = 0.0)
        gate_mask = gate_mask | batch.reason_loss_mask

        # Active on action commitment positions (target = 1.0)
        for b in range(b_sz):
            valid_act_indices = batch.action_indices[b][batch.action_type_mask[b]]
            gate_targets[b, valid_act_indices] = 1.0
            gate_mask[b, valid_act_indices] = True

        if gate_mask.any():
            loss_gate = F.binary_cross_entropy_with_logits(
                gate_logits[gate_mask], gate_targets[gate_mask]
            )
        else:
            loss_gate = 0.0 * gate_logits.sum()

        # ======================================================================
        # 3. Action Type Loss: Discrete Primitive Cross-Entropy
        # ======================================================================
        # action_output.action_type_logits: [B, S, num_action_types]
        # batch.action_indices: [B, max_actions]
        # batch.action_type_targets: [B, max_actions]
        # batch.action_type_mask: [B, max_actions] (bool)
        action_type_logits = model_output.action_output.action_type_logits
        num_action_types = action_type_logits.shape[-1]

        # Gather logits at action emission sequence positions
        # [B, max_actions, num_action_types]
        action_indices_expanded = batch.action_indices.unsqueeze(-1).expand(-1, -1, num_action_types)
        gathered_type_logits = torch.gather(action_type_logits, 1, action_indices_expanded)

        acc_action_type = 0.0
        if batch.action_type_mask.any():
            valid_type_logits = gathered_type_logits[batch.action_type_mask]  # [N_acts, num_action_types]
            valid_type_targets = batch.action_type_targets[batch.action_type_mask]  # [N_acts]

            # Calibrated class weights: CLICK=0.5, TYPE=3.0, SCROLL=1.5, NAVIGATE=1.5, API_CALL=1.5, FINISH=2.0
            legacy_type_weights = [0.5, 3.0, 1.5, 1.5, 1.5, 2.0]
            type_weights = torch.tensor(
                (legacy_type_weights + [2.0] * max(0, num_action_types - len(legacy_type_weights)))[:num_action_types],
                device=valid_type_logits.device,
                dtype=valid_type_logits.dtype,
            )
            loss_type = F.cross_entropy(valid_type_logits, valid_type_targets, weight=type_weights)

            # Compute classification accuracy
            preds = torch.argmax(valid_type_logits, dim=-1)
            acc_action_type = (preds == valid_type_targets).float().mean().item()
        else:
            loss_type = 0.0 * action_type_logits.sum()

        # ======================================================================
        # 4. Coordinate Loss: Smooth L1 / Huber on Normalized (x, y) in [0, 1]
        # ======================================================================
        # action_output.coordinates: [B, S, 2]
        # batch.coord_targets: [B, max_actions, 2]
        # batch.coord_loss_mask: [B, max_actions] (bool)
        coordinates = model_output.action_output.coordinates
        coord_indices_expanded = batch.action_indices.unsqueeze(-1).expand(-1, -1, 2)
        gathered_coords = torch.gather(coordinates, 1, coord_indices_expanded)  # [B, max_actions, 2]

        coord_mae = 0.0
        if batch.coord_loss_mask.any():
            valid_pred_coords = gathered_coords[batch.coord_loss_mask]  # [N_spatial, 2]
            valid_target_coords = batch.coord_targets[batch.coord_loss_mask].to(
                device=device, dtype=valid_pred_coords.dtype
            )  # [N_spatial, 2]

            loss_coord = F.smooth_l1_loss(
                valid_pred_coords, valid_target_coords, beta=cfg.huber_beta
            )
            coord_mae = F.l1_loss(valid_pred_coords, valid_target_coords).item()
        else:
            loss_coord = 0.0 * coordinates.sum()

        # ======================================================================
        # 5. Argument Loss: Masked Cross-Entropy on Payload Tokens
        # ======================================================================
        # action_output.argument_logits: [B, S, vocab_size]
        # batch.input_ids: [B, S]
        # batch.arg_loss_mask: [B, S] (bool)
        argument_logits = model_output.action_output.argument_logits
        shift_arg_logits = argument_logits[:, :-1, :].contiguous()  # [B, S-1, vocab_size]
        shift_arg_labels = batch.input_ids[:, 1:].contiguous()  # [B, S-1]
        shift_arg_mask = batch.arg_loss_mask[:, 1:].contiguous()  # [B, S-1]

        if shift_arg_mask.any():
            valid_arg_logits = shift_arg_logits[shift_arg_mask]  # [N_args, vocab_size]
            valid_arg_targets = shift_arg_labels[shift_arg_mask]  # [N_args]
            raw_loss_arg = F.cross_entropy(
                valid_arg_logits, valid_arg_targets, ignore_index=cfg.ignore_index
            )
            # Scale-normalize cross-entropy by log(vocab_size) to align with Huber and discrete CE magnitudes
            if getattr(cfg, "normalize_token_loss", True):
                loss_arg = raw_loss_arg / math.log(max(2, vocab_size))
            else:
                loss_arg = raw_loss_arg
        else:
            loss_arg = 0.0 * argument_logits.sum()

        # ======================================================================
        # 6. JEPA World Model Latent Consistency Loss
        # ======================================================================
        if model_output.jepa_loss is not None:
            loss_jepa = model_output.jepa_loss
        else:
            loss_jepa = torch.zeros((), device=device, dtype=dtype)

        # ======================================================================
        # 7. MoE Auxiliary Load-Balancing Loss
        # ======================================================================
        loss_aux = model_output.aux_loss

        # ======================================================================
        # 8. Unified Calibrated Composite Loss
        # ======================================================================
        total_loss = (
            cfg.reason_weight * loss_reason
            + cfg.gate_weight * loss_gate
            + cfg.action_type_weight * loss_type
            + cfg.coord_weight * loss_coord
            + cfg.arg_weight * loss_arg
            + cfg.jepa_weight * loss_jepa
            + cfg.aux_weight * loss_aux
        )

        # Build comprehensive logging metrics dictionary
        metrics: Dict[str, float] = {
            "loss_total": total_loss.item(),
            "loss_reason": loss_reason.item(),
            "loss_gate": loss_gate.item(),
            "loss_action_type": loss_type.item(),
            "loss_coord": loss_coord.item(),
            "loss_arg": loss_arg.item(),
            "loss_jepa": loss_jepa.item(),
            "loss_aux": loss_aux.item(),
            "acc_action_type": acc_action_type,
            "coord_mae": coord_mae,
        }

        # Include detailed JEPA sub-metrics if available
        if model_output.jepa_metrics is not None:
            for k, v in model_output.jepa_metrics.items():
                metrics[f"jepa_{k}"] = v.item() if isinstance(v, torch.Tensor) else float(v)

        return total_loss, metrics
