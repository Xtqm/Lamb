"""
JED-M2 Production Training CLI Script.
Calibrated for NVIDIA RTX 5060 Laptop GPU (8GB VRAM).

Usage:
    python train.py --epochs 3 --batch_size 16 --micro_batch_size 2 --lr 3e-4 --seq_len 2048
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from jed_m2.config import JEDM2Config
from jed_m2.dataset import AgentTrajectoryDataset, TrajectoryCollator
from jed_m2.losses import JEDM2LossConfig
from jed_m2.model import JEDM2Model
from jed_m2.trainer import JEDM2Trainer, JEDM2TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JED-M2 Native Large Action Model (LAM) Training Harness"
    )
    # Optimization & Steps
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clip norm")

    # Batch and Sequence Length
    parser.add_argument("--batch_size", type=int, default=16, help="Target effective batch size")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro-batch size per forward pass")
    parser.add_argument("--seq_len", type=int, default=2048, help="Maximum sequence length")

    # Checkpoint & Persistence
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for saved checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume from")
    parser.add_argument("--save_interval", type=int, default=250, help="Step interval to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="Step interval to log metrics")

    # Dataset & Seed
    parser.add_argument("--dataset_size", type=int, default=10000, help="Virtual dataset size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no_checkpointing", action="store_true", help="Disable activation checkpointing")

    # Validation & Evaluation
    parser.add_argument("--eval_interval", type=int, default=0, help="Step interval to run validation evaluation (0 = disabled)")
    parser.add_argument("--eval_steps", type=int, default=10, help="Number of micro-batches per evaluation pass")
    parser.add_argument("--eval_dataset_size", type=int, default=1000, help="Held-out evaluation dataset size")
    parser.add_argument("--export_history", type=str, default=None, help="File path to save JSON training and validation history")
    parser.add_argument("--baseline_checkpoint", type=str, default=None, help="Explicit target path to save baseline production checkpoint")
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    if args is None:
        args = parse_args()

    # Device & Precision setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"

    # Derive gradient accumulation steps
    grad_accum_steps = max(1, args.batch_size // args.micro_batch_size)
    effective_batch_size = args.micro_batch_size * grad_accum_steps

    print("=" * 80)
    print("      JED-M2 Native Large Action Model - Training Harness")
    print("=" * 80)
    print(f"Hardware Device       : {gpu_name} ({device})")
    print(f"Max Sequence Length   : {args.seq_len}")
    print(f"Micro-Batch Size      : {args.micro_batch_size}")
    print(f"Gradient Accumulation : {grad_accum_steps} steps (Effective Batch Size: {effective_batch_size})")
    print(f"Learning Rate         : {args.lr} -> {args.min_lr} (Warmup: {args.warmup_steps} steps)")
    print(f"Gradient Checkpointing: {'DISABLED' if args.no_checkpointing else 'ENABLED'}")
    print(f"Checkpoint Directory  : {args.checkpoint_dir}")
    if args.eval_interval > 0:
        print(f"Validation Interval   : Every {args.eval_interval} steps ({args.eval_steps} micro-batches)")
    if args.baseline_checkpoint:
        print(f"Baseline Checkpoint   : {args.baseline_checkpoint}")
    if args.export_history:
        print(f"History Export Path   : {args.export_history}")
    print("=" * 80 + "\n")

    # Instantiate model and loss configurations
    model_config = JEDM2Config(max_seq_len=args.seq_len)
    loss_config = JEDM2LossConfig()

    trainer_config = JEDM2TrainerConfig(
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=args.grad_clip,
        gradient_checkpointing=not args.no_checkpointing,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    # Initialize model
    model = JEDM2Model(model_config)

    # Initialize trainer
    trainer = JEDM2Trainer(
        model=model,
        config=trainer_config,
        loss_config=loss_config,
        device=device,
    )
    print(f"[Init] Optimizer instantiated: {trainer.opt_type}")

    # Resume from checkpoint if specified
    if args.resume:
        print(f"[Init] Resuming state from: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"[Init] Resumed at global step {trainer.global_step}, epoch {trainer.epoch}")

    # Dataset and Collator
    dataset = AgentTrajectoryDataset(
        config=model_config,
        virtual_size=args.dataset_size,
        seed=args.seed,
    )
    collator = TrajectoryCollator(config=model_config, max_seq_len=args.seq_len)

    # Held-out evaluation dataset
    val_dataset: Optional[AgentTrajectoryDataset] = None
    if args.eval_interval > 0:
        val_dataset = AgentTrajectoryDataset(
            config=model_config,
            virtual_size=args.eval_dataset_size,
            seed=args.seed + 1000,
        )
        print(f"[Init] Held-out validation dataset instantiated (size={args.eval_dataset_size}, seed={args.seed + 1000})")

    # Telemetry and history tracking
    history: Dict[str, Any] = {
        "config": {
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "seed": args.seed,
            "device": str(device),
            "gpu_name": gpu_name,
        },
        "train": [],
        "eval": [],
        "summary": {},
    }

    # Training Loop
    total_tokens_processed = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_tokens = 0
    last_eval_step = -1

    print("\n[Training] Starting training iterations...")
    data_idx = 0

    try:
        while trainer.global_step < args.max_steps:
            # Assemble micro-batch samples
            samples = []
            for _ in range(args.micro_batch_size):
                samples.append(dataset[data_idx % len(dataset)])
                data_idx += 1

            # Dynamic collation and transfer to device
            batch = collator(samples).to(device, dtype=trainer.dtype)
            tokens_in_batch = batch.input_ids.numel()
            total_tokens_processed += tokens_in_batch

            # Micro-step execution
            step_loss, step_metrics = trainer.training_step(batch)

            # Live logging
            if (
                trainer.accum_step % (args.log_interval * grad_accum_steps) == 0
                or trainer.global_step >= args.max_steps
            ):
                now = time.time()
                elapsed = now - last_log_time
                step_tokens = total_tokens_processed - last_log_tokens
                tokens_per_sec = step_tokens / max(elapsed, 1e-6)

                last_log_time = now
                last_log_tokens = total_tokens_processed

                peak_vram_mb = 0.0
                if device.type == "cuda":
                    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

                current_lr = trainer.get_current_lr()

                record = {
                    "step": trainer.global_step,
                    "accum_step": trainer.accum_step,
                    "loss_total": step_loss,
                    **step_metrics,
                    "lr": current_lr,
                    "tok_per_sec": tokens_per_sec,
                    "peak_vram_mb": peak_vram_mb,
                    "elapsed_sec": now - start_time,
                }
                history["train"].append(record)

                print(
                    f"Step {trainer.global_step:5d}/{args.max_steps} | "
                    f"Loss: {step_loss:7.4f} | "
                    f"Reason: {step_metrics.get('loss_reason', 0.0):6.4f} | "
                    f"Type: {step_metrics.get('loss_action_type', 0.0):6.4f} | "
                    f"Coord: {step_metrics.get('loss_coord', 0.0):6.4f} | "
                    f"JEPA: {step_metrics.get('loss_jepa', 0.0):6.4f} | "
                    f"LR: {current_lr:8.2e} | "
                    f"Tok/s: {tokens_per_sec:7.1f} | "
                    f"VRAM: {peak_vram_mb:6.1f} MB"
                )

            # Periodic validation evaluation
            if (
                val_dataset is not None
                and trainer.global_step > 0
                and trainer.global_step % args.eval_interval == 0
                and trainer.accum_step % grad_accum_steps == 0
                and trainer.global_step != last_eval_step
            ):
                last_eval_step = trainer.global_step
                print(f"  --> [Validation] Running evaluation at step {trainer.global_step} ({args.eval_steps} micro-batches)...")
                val_metrics = trainer.evaluate(
                    val_dataset,
                    collator,
                    max_eval_steps=args.eval_steps,
                    micro_batch_size=args.micro_batch_size,
                )
                eval_record = {
                    "step": trainer.global_step,
                    **val_metrics,
                }
                history["eval"].append(eval_record)
                print(
                    f"  --> [Validation] Step {trainer.global_step:5d} | "
                    f"Val Loss: {val_metrics.get('val_loss_total', 0.0):7.4f} | "
                    f"Reason: {val_metrics.get('val_loss_reason', 0.0):6.4f} | "
                    f"Type: {val_metrics.get('val_loss_action_type', 0.0):6.4f} | "
                    f"Coord: {val_metrics.get('val_loss_coord', 0.0):6.4f} | "
                    f"JEPA: {val_metrics.get('val_loss_jepa', 0.0):6.4f} | "
                    f"Type Acc: {val_metrics.get('val_acc_action_type', 0.0)*100:5.1f}% | "
                    f"Coord MAE: {val_metrics.get('val_coord_mae', 0.0):6.4f}"
                )

            # Periodic checkpointing
            if (
                args.save_interval > 0
                and trainer.global_step > 0
                and trainer.global_step % args.save_interval == 0
                and trainer.accum_step % grad_accum_steps == 0
            ):
                ckpt_file = os.path.join(
                    args.checkpoint_dir, f"checkpoint_step_{trainer.global_step}.pt"
                )
                saved_path = trainer.save_checkpoint(ckpt_file)
                print(f"  [Checkpoint] Saved step checkpoint: {saved_path}")

        # Final evaluation pass if validation enabled
        if val_dataset is not None and last_eval_step != trainer.global_step:
            print(f"  --> [Validation] Running final evaluation at step {trainer.global_step}...")
            val_metrics = trainer.evaluate(
                val_dataset,
                collator,
                max_eval_steps=args.eval_steps,
                micro_batch_size=args.micro_batch_size,
            )
            history["eval"].append({"step": trainer.global_step, **val_metrics})

        # Final checkpoint save
        final_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
        saved_final = trainer.save_checkpoint(final_ckpt)
        total_time = time.time() - start_time

        # Baseline checkpoint save if specified
        saved_baseline = None
        if args.baseline_checkpoint:
            saved_baseline = trainer.save_checkpoint(
                args.baseline_checkpoint,
                additional_info={
                    "is_baseline": True,
                    "baseline_name": "jed_m2_sft_base",
                    "step": trainer.global_step,
                    "final_loss": step_loss,
                },
            )

        history["summary"] = {
            "total_steps": trainer.global_step,
            "total_tokens": total_tokens_processed,
            "total_time_sec": total_time,
            "mean_tokens_per_sec": total_tokens_processed / max(total_time, 1e-6),
            "final_train_loss": step_loss,
            "final_metrics": step_metrics,
            "final_checkpoint": saved_final,
            "baseline_checkpoint": saved_baseline,
        }

        # Export metrics history JSON if configured
        if args.export_history:
            export_dir = os.path.dirname(os.path.abspath(args.export_history))
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
            with open(args.export_history, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            print(f"Metrics History Saved : {os.path.abspath(args.export_history)}")

        print("\n" + "=" * 80)
        print("      Training Complete!")
        print(f"Total Steps Completed : {trainer.global_step}")
        print(f"Total Tokens Trained  : {total_tokens_processed:,}")
        print(f"Total Wall Time       : {total_time:.2f}s")
        print(f"Final Checkpoint      : {saved_final}")
        if saved_baseline:
            print(f"Baseline Checkpoint   : {saved_baseline}")
        print("=" * 80 + "\n")

        return history

    except KeyboardInterrupt:
        print("\n[Notice] Training interrupted by user. Saving emergency checkpoint...")
        emergency_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_interrupted.pt")
        trainer.save_checkpoint(emergency_ckpt)
        print(f"[Notice] Emergency checkpoint saved to: {emergency_ckpt}")
        if args.export_history:
            export_dir = os.path.dirname(os.path.abspath(args.export_history))
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
            with open(args.export_history, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
        sys.exit(0)


if __name__ == "__main__":
    main()
