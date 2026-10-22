"""
JED-M2 Real-World Web Trajectory Scaled Pretraining Engine.
Calibrated for NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7 VRAM).

Usage:
    python train_pretrain.py --model_size nano --max_steps 5000 --batch_size 16 --micro_batch_size 2
    python train_pretrain.py --model_size medium --max_steps 10000 --batch_size 16 --micro_batch_size 2
"""

import argparse
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jed_m2.config import JEDM2Config, JEDM2MediumConfig
from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
from jed_m2.model import JEDM2Model, JEDM2Output, create_jed_m2_medium, create_jed_m2_nano
from jed_m2.real_dataset import RealTrajectoryCollator, RealWebTrajectoryDataset
from jed_m2.trainer import JEDM2TrainerConfig, build_cosine_scheduler, build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JED-M2 Scaled Real Web Trajectory Pretraining Engine"
    )
    # Model Scale
    parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "medium"],
        help="Model scale tier: nano (~53M params) or medium (~273M params)",
    )

    # Optimization & Steps
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum pretraining steps")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate (default: 3e-4 nano, 1.5e-4 medium)")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")

    # Batch and Sequence Length
    parser.add_argument("--batch_size", type=int, default=16, help="Target effective batch size (accumulated)")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro-batch size per forward pass")
    parser.add_argument("--seq_len", type=int, default=2048, help="Maximum sequence length")

    # Dataset & Cache
    parser.add_argument("--dataset_size", type=int, default=5000, help="Number of real trajectory samples in train split")
    parser.add_argument("--eval_dataset_size", type=int, default=500, help="Number of samples in val split")
    parser.add_argument("--cache_dir", type=str, default="data/real_trajectories_cache", help="Directory for cached tensors")
    parser.add_argument("--force_rebuild_dataset", action="store_true", help="Force rebuild real trajectory cache")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Checkpoint & Telemetry
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/pretrain", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=500, help="Step interval to save checkpoints")
    parser.add_argument("--eval_interval", type=int, default=250, help="Step interval to evaluate validation split")
    parser.add_argument("--eval_steps", type=int, default=20, help="Number of micro-batches per evaluation pass")
    parser.add_argument("--log_interval", type=int, default=10, help="Step interval to log metrics")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--export_history", type=str, default=None, help="Path to export JSON training history")
    parser.add_argument("--no_checkpointing", action="store_true", help="Disable gradient activation checkpointing")

    return parser.parse_args()


def train_pretrain(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    if args is None:
        args = parse_args()

    # Hardware & Precision Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"
        dtype = torch.float32

    # Model Configuration
    if args.model_size == "medium":
        config = JEDM2MediumConfig(max_seq_len=args.seq_len)
        default_lr = 1.5e-4
    else:
        config = JEDM2Config(max_seq_len=args.seq_len)
        default_lr = 3e-4

    peak_lr = args.lr if args.lr is not None else default_lr

    # Compute Gradient Accumulation Steps
    grad_accum_steps = max(1, args.batch_size // args.micro_batch_size)
    effective_batch_size = args.micro_batch_size * grad_accum_steps

    print("=" * 80)
    print("      JED-M2 Real Web Trajectory Scaled Pretraining Engine")
    print("=" * 80)
    print(f"Model Architecture    : JED-M2-{args.model_size.upper()}")
    print(f"Hardware Device       : {gpu_name} ({device})")
    print(f"Target Precision      : {dtype}")
    print(f"Max Sequence Length   : {args.seq_len}")
    print(f"Micro-Batch Size      : {args.micro_batch_size}")
    print(f"Gradient Accumulation : {grad_accum_steps} steps (Effective Batch Size: {effective_batch_size})")
    print(f"Learning Rate Schedule: {peak_lr:.2e} -> {args.min_lr:.2e} (Warmup: {args.warmup_steps} steps)")
    print(f"Max Pretraining Steps : {args.max_steps}")
    print(f"Activation Checkpoint : {'DISABLED' if args.no_checkpointing else 'ENABLED'}")
    print(f"Cache Directory       : {args.cache_dir}")
    print(f"Checkpoint Directory  : {args.checkpoint_dir}")
    print("=" * 80)

    # Initialize Model
    if args.model_size == "medium":
        model = create_jed_m2_medium(config)
    else:
        model = create_jed_m2_nano(config)

    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()

    model = model.to(device=device, dtype=dtype)

    param_counts = model.count_parameters()
    total_m = param_counts['total_params'] / 1e6
    active_m = param_counts['active_params'] / 1e6
    print(f"Total Parameters      : {total_m:.2f}M")
    print(f"Active Parameters     : {active_m:.2f}M per token")

    # Dataset & Collator Initialization
    print("\nLoading Real-World Web Trajectory Datasets...")
    train_dataset = RealWebTrajectoryDataset(
        config=config,
        split="train",
        dataset_size=args.dataset_size,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_dataset,
        seed=args.seed,
    )
    val_dataset = RealWebTrajectoryDataset(
        config=config,
        split="val",
        dataset_size=args.eval_dataset_size,
        cache_dir=args.cache_dir,
        force_rebuild=args.force_rebuild_dataset,
        seed=args.seed + 10000,
    )
    print(f"Train Dataset Size    : {len(train_dataset)} trajectories")
    print(f"Validation Dataset    : {len(val_dataset)} trajectories")

    collator = RealTrajectoryCollator(config=config, max_seq_len=args.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=collator,
        drop_last=True,
    )

    # Optimizer & Scheduler
    trainer_cfg = JEDM2TrainerConfig(
        lr=peak_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=args.grad_clip,
    )
    optimizer, opt_type = build_optimizer(model, trainer_cfg, device)
    scheduler = build_cosine_scheduler(optimizer, trainer_cfg)
    print(f"Optimizer Configured  : {opt_type}")

    # Loss Function
    loss_fn = JEDM2CompositeLoss(config, loss_config=JEDM2LossConfig())

    # Checkpoint Directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Resume if requested
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming checkpoint from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed at step {start_step}")

    # Training State Tracking
    global_step = start_step
    accum_count = 0
    running_loss = 0.0
    running_metrics: Dict[str, float] = {}
    history: List[Dict[str, Any]] = []

    train_iter = iter(train_loader)
    total_tokens_processed = 0
    start_time = time.time()
    last_log_time = time.time()
    last_log_tokens = 0

    print("\nStarting Pretraining Loop...")
    model.train()
    optimizer.zero_grad(set_to_none=True)

    while global_step < args.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device=device, dtype=dtype)
        batch_tokens = batch.input_ids.numel()
        total_tokens_processed += batch_tokens

        # Forward pass with mixed precision
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=(device.type == "cuda"),
        ):
            output: JEDM2Output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                obs_t=batch.jepa_obs_t,
                action_t=batch.jepa_action_t,
                obs_next=batch.jepa_obs_next,
            )
            loss, metrics = loss_fn(output, batch)

        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        accum_count += 1
        loss_val = loss.item()
        running_loss += loss_val
        for k, v in metrics.items():
            running_metrics[k] = running_metrics.get(k, 0.0) + v

        # Gradient accumulation step
        if accum_count % grad_accum_steps == 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm_val = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Telemetry Logging
            if global_step % args.log_interval == 0 or global_step == 1:
                cur_time = time.time()
                elapsed = cur_time - last_log_time
                tok_delta = total_tokens_processed - last_log_tokens
                tok_per_sec = tok_delta / max(elapsed, 1e-4)

                avg_loss = running_loss / args.log_interval
                avg_metrics = {k: v / args.log_interval for k, v in running_metrics.items()}
                cur_lr = scheduler.get_last_lr()[0]

                vram_gb = 0.0
                peak_vram_gb = 0.0
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.memory_allocated() / (1024**3)
                    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

                print(
                    f"Step {global_step:5d}/{args.max_steps:5d} | "
                    f"Loss: {avg_loss:6.3f} | "
                    f"Reason: {avg_metrics.get('loss_reason', 0.0):5.2f} | "
                    f"ActAcc: {avg_metrics.get('acc_action_type', 0.0)*100:4.1f}% | "
                    f"CoordMAE: {avg_metrics.get('coord_mae', 0.0):.4f} | "
                    f"JEPASim: {avg_metrics.get('jepa_cos_sim', 0.0):.3f} | "
                    f"GradNorm: {grad_norm_val:5.2f} | "
                    f"LR: {cur_lr:.2e} | "
                    f"Tok/s: {tok_per_sec:6.0f} | "
                    f"VRAM: {vram_gb:.2f}GB (Peak: {peak_vram_gb:.2f}GB)"
                )

                log_entry = {
                    "step": global_step,
                    "loss": avg_loss,
                    "metrics": avg_metrics,
                    "grad_norm": grad_norm_val,
                    "lr": cur_lr,
                    "throughput_tok_per_sec": tok_per_sec,
                    "vram_gb": vram_gb,
                    "peak_vram_gb": peak_vram_gb,
                    "timestamp": datetime.now().isoformat(),
                }
                history.append(log_entry)

                running_loss = 0.0
                running_metrics.clear()
                last_log_time = cur_time
                last_log_tokens = total_tokens_processed

            # Periodic Validation Evaluation Pass
            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                print(f"\n--- Running Validation Pass at Step {global_step} ({args.eval_steps} micro-batches) ---")
                val_metrics = evaluate_validation(model, val_loader, loss_fn, device, dtype, args.eval_steps)
                print(
                    f"Validation Result: Loss: {val_metrics['val_loss']:.3f} | "
                    f"ActAcc: {val_metrics['val_acc_action_type']*100:.2f}% (Target: >70%) | "
                    f"CoordMAE: {val_metrics['val_coord_mae']:.4f} (Target: <0.05) | "
                    f"JEPASim: {val_metrics['val_jepa_cos_sim']:.4f} (Target: >0.85)"
                )
                if history:
                    history[-1]["val_metrics"] = val_metrics
                model.train()

            # Periodic Checkpoint Persistence
            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"jed_m2_{args.model_size}_step_{global_step}.pt")
                torch.save(
                    {
                        "step": global_step,
                        "model_size": args.model_size,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "config": asdict(config),
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved to: {ckpt_path}\n")

    # Final Evaluation & Convergence Check
    print("\n" + "=" * 80)
    print("Pretraining Complete. Running Final Convergence Evaluation...")
    print("=" * 80)
    final_val_metrics = evaluate_validation(model, val_loader, loss_fn, device, dtype, args.eval_steps * 2)

    act_acc = final_val_metrics["val_acc_action_type"]
    coord_mae = final_val_metrics["val_coord_mae"]
    jepa_sim = final_val_metrics["val_jepa_cos_sim"]

    print(f"Final Validation Loss        : {final_val_metrics['val_loss']:.4f}")
    print(f"Action Primitive Accuracy    : {act_acc*100:.2f}% (Target: >70.0%)")
    print(f"Continuous Coordinate MAE    : {coord_mae:.4f} (Target: <0.0500)")
    print(f"JEPA Latent Cosine Sim       : {jepa_sim:.4f} (Target: >0.8500)")

    # Save Final Production Checkpoint
    final_ckpt_path = os.path.join(args.checkpoint_dir, f"jed_m2_{args.model_size}_pretrain_final.pt")
    torch.save(
        {
            "step": global_step,
            "model_size": args.model_size,
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "final_metrics": final_val_metrics,
        },
        final_ckpt_path,
    )
    print(f"Final model checkpoint saved to: {final_ckpt_path}")

    # Export structured history if requested
    if args.export_history:
        with open(args.export_history, "w") as f:
            json.dump(
                {
                    "model_size": args.model_size,
                    "max_steps": args.max_steps,
                    "final_metrics": final_val_metrics,
                    "history": history,
                },
                f,
                indent=2,
            )
        print(f"Training telemetry history exported to: {args.export_history}")

    return {
        "final_step": global_step,
        "final_val_metrics": final_val_metrics,
        "history": history,
        "final_checkpoint": final_ckpt_path,
    }


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: JEDM2CompositeLoss,
    device: torch.device,
    dtype: torch.dtype,
    max_steps: int = 20,
) -> Dict[str, float]:
    """Evaluates validation loss and convergence metrics across held-out real web trajectories."""
    model.eval()
    total_val_loss = 0.0
    accumulated_metrics: Dict[str, float] = {}
    steps = 0

    val_iter = iter(val_loader)
    for _ in range(max_steps):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        batch = batch.to(device=device, dtype=dtype)
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=(device.type == "cuda"),
        ):
            output: JEDM2Output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                obs_t=batch.jepa_obs_t,
                action_t=batch.jepa_action_t,
                obs_next=batch.jepa_obs_next,
            )
            loss, metrics = loss_fn(output, batch)

        total_val_loss += loss.item()
        for k, v in metrics.items():
            accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v
        steps += 1

    steps = max(1, steps)
    results = {f"val_{k}": v / steps for k, v in accumulated_metrics.items()}
    results["val_loss"] = total_val_loss / steps
    return results


if __name__ == "__main__":
    train_pretrain()
