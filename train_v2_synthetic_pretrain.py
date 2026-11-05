"""
JED-M2 v2.0 Procedural Synthetic Gym Pretraining Harness.
Trains all LAM components (SSM + MLA backbone, action heads, pointer-generator, JEPA world model)
on diverse multi-domain procedural trajectories (Booking, CRM, E-commerce, Cloud Admin, Web Browser).
Calibrated for NVIDIA GeForce RTX 5060 Laptop GPU (< 4.5 GB VRAM).

Usage:
    python train_v2_synthetic_pretrain.py --steps 500 --batch_size 16 --micro_batch_size 4 --device cuda
"""

import argparse
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.config import JEDM2Config
from jed_m2.dataset import AgentTrajectoryDataset, TrajectoryCollator
from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
from jed_m2.model import JEDM2Model, create_jed_m2_nano
from jed_m2.trainer import JEDM2Trainer, JEDM2TrainerConfig, build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JED-M2 v2.0 Synthetic Gym Pretraining Engine")
    parser.add_argument("--steps", type=int, default=500, help="Total pretraining optimizer steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum annealed learning rate")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Linear warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--batch_size", type=int, default=16, help="Effective batch size")
    parser.add_argument("--micro_batch_size", type=int, default=4, help="Micro-batch size per forward pass")
    parser.add_argument("--seq_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/pretrain/jed_m2_nano_pretrain_final.pt", help="Base checkpoint to resume from")
    parser.add_argument("--output_checkpoint", type=str, default="checkpoints/v2_pretrain/jed_m2_v2_synthetic_pretrain.pt", help="Output checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log_interval", type=int, default=25, help="Steps between log outputs")
    return parser.parse_args()


def get_cosine_lr(step: int, total_steps: int, peak_lr: float, min_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return min_lr + (peak_lr - min_lr) * (step / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if "cuda" in args.device and torch.cuda.is_available() else torch.float32

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    print("=" * 80)
    print("      JED-M2 v2.0 Procedural Synthetic Gym Pretraining Engine")
    print("=" * 80)
    print(f"Device                : {device} ({dtype})")
    print(f"Target Steps          : {args.steps}")
    print(f"Effective Batch Size  : {args.batch_size} (Micro-batch: {args.micro_batch_size})")
    print(f"Peak Learning Rate    : {args.lr}")
    print(f"Base Checkpoint       : {args.base_checkpoint}")
    print(f"Output Checkpoint     : {args.output_checkpoint}")
    print("=" * 80 + "\n")

    config = JEDM2Config(max_seq_len=args.seq_len)
    tokenizer = ActionBPETokenizer(vocab_size=config.vocab_size)

    # Initialize model
    model = create_jed_m2_nano(config).to(device=device, dtype=dtype)
    if os.path.exists(args.base_checkpoint):
        print(f"[Init] Loading base weights from {args.base_checkpoint}...")
        ckpt = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Init] Loaded base model. Missing keys (new components): {len(missing)}, unexpected: {len(unexpected)}")
    else:
        print(f"[Init] Warning: Base checkpoint {args.base_checkpoint} not found. Training from scratch.")

    model.train()
    model.gradient_checkpointing_enable()

    # Datasets and DataLoader
    print("[Data] Initializing streaming multi-domain procedural trajectory datasets...")
    train_dataset = AgentTrajectoryDataset(
        config=config,
        virtual_size=20000,
        seed=args.seed,
        tokenizer=tokenizer,
    )
    collator = TrajectoryCollator(config=config, tokenizer=tokenizer, max_seq_len=args.seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    # Loss and Optimizer
    loss_config = JEDM2LossConfig(
        reason_weight=1.0,
        gate_weight=1.0,
        action_type_weight=2.0,
        coord_weight=12.0,
        arg_weight=8.0,
        jepa_weight=1.0,
        aux_weight=1.0,
        finish_action_weight=2.5,
    )
    loss_fn = JEDM2CompositeLoss(config=config, loss_config=loss_config)

    grad_accum_steps = max(1, args.batch_size // args.micro_batch_size)
    trainer_cfg = JEDM2TrainerConfig(
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.steps,
        weight_decay=args.weight_decay,
        grad_accum_steps=grad_accum_steps,
        grad_clip_norm=args.grad_clip,
    )
    optimizer, opt_type = build_optimizer(model, config=trainer_cfg, device=device)

    print(f"[Train] Optimizer: {opt_type} | Grad Accum Steps: {grad_accum_steps}")
    print("[Header] Step | Total Loss | Action CE | Coord MAE | Arg Loss | JEPA Loss | LR | Tok/s | Peak VRAM")
    print("-" * 95)

    data_iter = iter(train_loader)
    step = 0
    accum_loss = 0.0
    accum_metrics: Dict[str, float] = {}
    tokens_processed = 0
    t0 = time.time()
    t_start = t0

    optimizer.zero_grad()

    while step < args.steps:
        # Accumulate gradients over micro-batches
        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = batch.to(device=device, dtype=dtype)
            tok_count = int(batch.input_ids.numel())
            tokens_processed += tok_count

            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                obs_next=batch.jepa_obs_next,
                action_t=batch.jepa_action_t,
                images=batch.images,
                prompt_ids=batch.prompt_token_ids,
                prompt_mask=batch.prompt_mask,
            )

            loss, metrics = loss_fn(output, batch)
            loss_scaled = loss / grad_accum_steps
            loss_scaled.backward()

            accum_loss += loss.item() / grad_accum_steps
            for k, v in metrics.items():
                accum_metrics[k] = accum_metrics.get(k, 0.0) + (v / grad_accum_steps)

        # Gradient clipping & optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        # Learning rate schedule update
        cur_lr = get_cosine_lr(step, args.steps, args.lr, args.min_lr, args.warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.step()
        optimizer.zero_grad()

        step += 1

        if step % args.log_interval == 0 or step == args.steps:
            elapsed = time.time() - t0
            tok_rate = tokens_processed / max(0.001, elapsed)
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            loss_total = accum_loss / args.log_interval
            act_ce = accum_metrics.get("loss_action_type", 0.0) / args.log_interval
            coord_mae = accum_metrics.get("coord_mae", 0.0) / args.log_interval
            arg_loss = accum_metrics.get("loss_arg", 0.0) / args.log_interval
            jepa_loss = accum_metrics.get("loss_jepa", 0.0) / args.log_interval

            print(
                f"[{step:04d}/{args.steps:04d}]  "
                f"Loss: {loss_total:6.4f} | "
                f"Act: {act_ce:5.3f} | "
                f"MAE: {coord_mae:6.4f} | "
                f"Arg: {arg_loss:5.3f} | "
                f"JEPA: {jepa_loss:5.3f} | "
                f"LR: {cur_lr:.2e} | "
                f"{tok_rate:5.0f} t/s | "
                f"{peak_vram:6.1f} MB"
            )

            accum_loss = 0.0
            accum_metrics.clear()
            tokens_processed = 0
            t0 = time.time()

    # Save pretrain checkpoint
    out_dir = os.path.dirname(args.output_checkpoint)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    save_payload = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "step": args.steps,
        "additional_info": {
            "phase": "v2_synthetic_pretrain",
            "base_checkpoint": args.base_checkpoint,
            "domains": "booking,crm,ecommerce,cloud_admin,web_browser,travel,os_admin",
            "steps": args.steps,
            "final_lr": cur_lr,
        },
    }
    torch.save(save_payload, args.output_checkpoint)
    total_time = time.time() - t_start
    print(f"\n[Done] Successfully saved pretrained checkpoint to: {args.output_checkpoint} (Total time: {total_time:.1f}s)")


if __name__ == "__main__":
    main()
