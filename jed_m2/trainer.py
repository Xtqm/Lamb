"""
JED-M2 Production Training Harness.
Calibrated for NVIDIA RTX 5060 Laptop GPU (8GB GDDR7 VRAM, sm_120 Blackwell architecture).

Key Features:
1. Fused 8-bit AdamW optimizer integration with automatic native CUDA-fused AdamW fallback.
2. Selective activation checkpointing across the 3 JEDM2Block layers.
3. Native torch.bfloat16 mixed-precision training.
4. Gradient accumulation supporting effective batch sizes 16-32 with micro-batch size 2-4.
5. Linear warmup with cosine decay learning rate schedule.
6. Atomic checkpoint persistence and resumption for model, optimizer, scheduler, and metrics.
"""

from dataclasses import asdict, dataclass
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from jed_m2.config import JEDM2Config
from jed_m2.dataset import TrajectoryBatch
from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
from jed_m2.model import JEDM2Model, JEDM2Output


@dataclass
class JEDM2TrainerConfig:
    """Configuration hyperparameters for JED-M2 production trainer."""

    # Learning rate schedule
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 1000
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Batch and accumulation
    micro_batch_size: int = 2
    grad_accum_steps: int = 8  # Effective batch size = 2 * 8 = 16
    grad_clip_norm: float = 1.0

    # Memory and precision
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "bfloat16"  # "bfloat16" or "float32"

    # Persistence & Logging
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 250
    log_interval: int = 10
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size after gradient accumulation."""
        return self.micro_batch_size * self.grad_accum_steps


def build_optimizer(
    model: nn.Module,
    config: JEDM2TrainerConfig,
    device: torch.device,
) -> Tuple[torch.optim.Optimizer, str]:
    """
    Constructs an AdamW optimizer separating weight decay for matrices vs biases/norms.
    Tries fused 8-bit AdamW (bitsandbytes) first if configured, falling back to
    PyTorch native CUDA-fused AdamW (torch.optim.AdamW(..., fused=True)).

    Args:
        model: JEDM2Model instance.
        config: Trainer configuration dataclass.
        device: Active target torch device.

    Returns:
        Tuple of (optimizer, optimizer_type_description_string).
    """
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Biases, 1D norm scales (RMSNorm), and 1D parameters do not undergo weight decay
        if param.ndim >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    opt_type = "unknown"

    # Attempt 8-bit AdamW if requested
    if config.use_8bit_adam and device.type == "cuda":
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
            )
            opt_type = "bitsandbytes.optim.AdamW8bit"
            return optimizer, opt_type
        except (ImportError, Exception):
            # Fall through to native PyTorch fused AdamW
            pass

    # Native PyTorch AdamW with hardware-accelerated fused kernel on CUDA
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config.lr,
                betas=config.betas,
                eps=config.eps,
                fused=True,
            )
            opt_type = "torch.optim.AdamW(fused=True)"
            return optimizer, opt_type
        except Exception:
            pass

    # Standard PyTorch AdamW
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
    )
    opt_type = "torch.optim.AdamW(standard)"
    return optimizer, opt_type


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    config: JEDM2TrainerConfig,
) -> LambdaLR:
    """
    Constructs a learning rate scheduler with linear warmup followed by cosine annealing.

    Args:
        optimizer: Optimizer instance.
        config: Trainer configuration dataclass.

    Returns:
        LambdaLR scheduler instance.
    """
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps
    min_lr_ratio = config.min_lr / max(config.lr, 1e-8)

    def lr_lambda(current_step: int) -> float:
        # 1. Linear Warmup
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        # 2. Post max-step clamp
        if current_step >= max_steps:
            return min_lr_ratio
        # 3. Cosine Annealing decay
        progress = float(current_step - warmup_steps) / float(
            max(1, max_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + cosine_decay * (1.0 - min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)


class JEDM2Trainer:
    """
    Production Training Harness for JED-M2 Large Action Model.
    Orchestrates mixed-precision forward/backward, gradient accumulation,
    clipping, LR scheduling, live logging, and checkpoint recovery.
    """

    def __init__(
        self,
        model: JEDM2Model,
        config: Optional[JEDM2TrainerConfig] = None,
        loss_config: Optional[JEDM2LossConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.config = config if config is not None else JEDM2TrainerConfig()
        self.loss_config = loss_config if loss_config is not None else JEDM2LossConfig()

        # Determine target device
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine execution dtype
        if self.device.type == "cuda" and self.config.mixed_precision == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        self.model = model.to(self.device, dtype=self.dtype)
        self.loss_fn = JEDM2CompositeLoss(model.config, self.loss_config)

        # Configure activation checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        else:
            self.model.gradient_checkpointing_disable()

        # Initialize optimizer and scheduler
        self.optimizer, self.opt_type = build_optimizer(
            self.model, self.config, self.device
        )
        self.scheduler = build_cosine_scheduler(self.optimizer, self.config)

        # State tracking
        self.global_step: int = 0
        self.accum_step: int = 0
        self.epoch: int = 0
        self.latest_loss: float = 0.0
        self.latest_metrics: Dict[str, float] = {}

    def get_current_lr(self) -> float:
        """Returns current learning rate from optimizer param groups."""
        return self.optimizer.param_groups[0]["lr"]

    def training_step(
        self,
        batch: TrajectoryBatch,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Executes a single micro-batch training step with mixed-precision and gradient accumulation.

        Args:
            batch: Dynamically collated TrajectoryBatch on target device.

        Returns:
            Tuple of (unscaled_total_loss_float, detached_metrics_dict).
        """
        self.model.train()

        # Mixed-precision context
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=(self.device.type == "cuda"),
        ):
            # Sequence-parallel forward pass
            output: JEDM2Output = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                obs_t=batch.jepa_obs_t,
                action_t=batch.jepa_action_t,
                obs_next=batch.jepa_obs_next,
            )

            # Evaluate composite multi-task loss
            total_loss, metrics = self.loss_fn(output, batch)

        # Scale loss by accumulation factor for backward pass
        scaled_loss = total_loss / self.config.grad_accum_steps
        scaled_loss.backward()

        self.accum_step += 1
        loss_val = total_loss.item()
        self.latest_loss = loss_val
        self.latest_metrics = metrics

        # When accumulation completes, apply gradient clip and optimizer step
        if self.accum_step % self.config.grad_accum_steps == 0:
            # Gradient clipping to prevent exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip_norm
            )
            metrics["grad_norm"] = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

            self.optimizer.step()
            self.scheduler.step()

            # Apply Polyak momentum update to JEPA target encoder (EMA rule)
            if hasattr(self.model, "jepa") and self.model.jepa is not None:
                self.model.jepa.update_target_network()

            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1

        return loss_val, metrics

    @torch.no_grad()
    def evaluation_step(
        self,
        batch: TrajectoryBatch,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Executes a single validation/evaluation step on a micro-batch without gradient calculation.

        Args:
            batch: Dynamically collated TrajectoryBatch on target device.

        Returns:
            Tuple of (unscaled_total_loss_float, detached_metrics_dict).
        """
        self.model.eval()

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=(self.device.type == "cuda"),
        ):
            # Sequence-parallel forward pass without parameter gradients
            # input_ids: [B, S], attention_mask: [B, 1, S, S]
            output: JEDM2Output = self.model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                obs_t=batch.jepa_obs_t,
                action_t=batch.jepa_action_t,
                obs_next=batch.jepa_obs_next,
            )

            # Evaluate composite multi-task loss and diagnostic metrics
            total_loss, metrics = self.loss_fn(output, batch)

        loss_val = total_loss.item()
        return loss_val, metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataset: Any,
        collator: Any,
        max_eval_steps: int = 20,
        micro_batch_size: Optional[int] = None,
        start_idx: int = 0,
    ) -> Dict[str, float]:
        """
        Evaluates the model over a held-out evaluation dataset, aggregating multi-head losses,
        coordinate MAE, and primitive accuracy across multiple evaluation batches.

        Args:
            dataset: Evaluation dataset instance (e.g. AgentTrajectoryDataset).
            collator: Batch collator instance (e.g. TrajectoryCollator).
            max_eval_steps: Maximum number of micro-batches to evaluate.
            micro_batch_size: Micro-batch size per step (defaults to config.micro_batch_size).
            start_idx: Starting index in dataset.

        Returns:
            Dictionary of mean validation metrics across evaluated batches prefixed with 'val_'.
        """
        m_bs = micro_batch_size or self.config.micro_batch_size
        metric_sums: Dict[str, float] = {}
        eval_batches = 0
        current_idx = start_idx

        for _ in range(max_eval_steps):
            samples = []
            for _ in range(m_bs):
                samples.append(dataset[current_idx % len(dataset)])
                current_idx += 1

            batch = collator(samples).to(self.device, dtype=self.dtype)
            step_loss, step_metrics = self.evaluation_step(batch)

            for k, v in step_metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
            eval_batches += 1

        if eval_batches == 0:
            return {}

        avg_metrics: Dict[str, float] = {
            f"val_{k}": v / eval_batches for k, v in metric_sums.items()
        }
        return avg_metrics

    def save_checkpoint(
        self,
        filepath: str,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Atomically saves training checkpoint to disk.

        Args:
            filepath: Target checkpoint path (.pt or .pth).
            additional_info: Optional extra metadata dictionary.

        Returns:
            Resolved absolute file path of saved checkpoint.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        tmp_filepath = f"{filepath}.tmp"

        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "model_config": asdict(self.model.config),
            "trainer_config": asdict(self.config),
            "loss_config": asdict(self.loss_config),
            "global_step": self.global_step,
            "accum_step": self.accum_step,
            "epoch": self.epoch,
            "latest_loss": self.latest_loss,
            "latest_metrics": self.latest_metrics,
            "saved_at": time.time(),
        }
        if additional_info:
            payload["additional_info"] = additional_info

        torch.save(payload, tmp_filepath)
        # Atomic rename to prevent corrupt files on abrupt interruption
        if os.path.exists(filepath):
            os.remove(filepath)
        os.rename(tmp_filepath, filepath)
        return os.path.abspath(filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """
        Loads training state from a saved checkpoint.

        Args:
            filepath: Path to checkpoint file.

        Returns:
            Full loaded checkpoint dictionary.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.accum_step = checkpoint.get("accum_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.latest_loss = checkpoint.get("latest_loss", 0.0)
        self.latest_metrics = checkpoint.get("latest_metrics", {})
        return checkpoint
