"""
JED-M2 Reinforcement Learning CLI Training Script.
Supports:
1. Online Group Relative Policy Optimization (GRPO) on live interactive rollouts.
2. Offline Direct Policy Optimization (DPO) on paired preference rollouts.

Calibrated for NVIDIA GeForce RTX 5060 Laptop GPU (< 6.0 GB VRAM limit).
STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure ASCII console output (no Unicode box-drawing characters).
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import torch

from jed_m2.config import JEDM2Config
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.environment import WebOSEnvironment, get_default_scenarios
from jed_m2.model import JEDM2Model
from jed_m2.rl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
from jed_m2.rl_pipeline import RolloutTrajectory, TrajectoryRolloutSampler
from jed_m2.trainer import JEDM2TrainerConfig, build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JED-M2 Native Large Action Model Reinforcement Learning Harness"
    )
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="grpo",
        choices=["grpo", "dpo"],
        help="RL training algorithm: 'grpo' (online group rollouts) or 'dpo' (paired preference optimization)",
    )

    # Optimization & Steps
    parser.add_argument("--iterations", type=int, default=25, help="Number of RL training iterations")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for policy optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Maximum gradient clipping norm")

    # GRPO Hyperparameters
    parser.add_argument("--group_size", type=int, default=4, help="Number of rollouts G per prompt group")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO-style surrogate objective clipping epsilon")
    parser.add_argument("--beta_kl", type=float, default=0.04, help="KL divergence penalty coefficient against reference model")
    parser.add_argument("--temperature", type=float, default=0.7, help="Exploration sampling temperature")

    # DPO Hyperparameters
    parser.add_argument("--beta_dpo", type=float, default=0.1, help="Implicit reward scaling temperature beta for DPO")

    # Environment & Tasks
    parser.add_argument(
        "--scenarios",
        type=str,
        default="ecommerce,travel,os_admin",
        help="Comma-separated list of scenario IDs to sample rollouts from",
    )
    parser.add_argument("--max_turns", type=int, default=8, help="Maximum interaction turns per episode")

    # Hardware & Performance
    parser.add_argument("--seq_len", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--no_checkpointing", action="store_true", help="Disable activation checkpointing")
    parser.add_argument("--device", type=str, default=None, help="Force execution device ('cuda' or 'cpu')")

    # Checkpoint & Logging
    parser.add_argument("--model_path", type=str, default=None, help="Optional initial model checkpoint (.pt) to seed RL policy")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_rl", help="Directory for saved RL checkpoints")
    parser.add_argument("--save_interval", type=int, default=25, help="Iteration interval to save checkpoints")
    parser.add_argument("--log_interval", type=int, default=1, help="Iteration interval to log telemetry")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # --------------------------------------------------------------------------
    # 1. Device and Precision Configuration
    # --------------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
        dtype = torch.bfloat16 if "cuda" in args.device and torch.cuda.is_available() else torch.float32
        gpu_name = torch.cuda.get_device_name(0) if "cuda" in args.device and torch.cuda.is_available() else "CPU"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        gpu_name = torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        gpu_name = "CPU"

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]

    print("=" * 80)
    print("      JED-M2 Large Action Model - Reinforcement Learning Harness")
    print("=" * 80)
    print(f"Algorithm Mode        : {args.mode.upper()}")
    print(f"Hardware Device       : {gpu_name} ({device})")
    print(f"Precision             : {dtype}")
    print(f"RL Iterations         : {args.iterations}")
    print(f"Learning Rate         : {args.lr}")
    if args.mode == "grpo":
        print(f"Group Size (G)        : {args.group_size} rollouts per goal")
        print(f"Surrogate Clip (eps)  : {args.clip_eps}")
        print(f"Reference KL Beta     : {args.beta_kl}")
        print(f"Sampling Temperature  : {args.temperature}")
    else:
        print(f"DPO Beta              : {args.beta_dpo}")
    print(f"Target Scenarios      : {', '.join(scenarios)}")
    print(f"Max Turns / Episode   : {args.max_turns}")
    print(f"Activation Checkpoint : {'DISABLED' if args.no_checkpointing else 'ENABLED'}")
    print(f"Checkpoint Directory  : {args.checkpoint_dir}")
    print("=" * 80 + "\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Model & Reference Model Initialization
    # --------------------------------------------------------------------------
    model_config = JEDM2Config(max_seq_len=args.seq_len)
    tokenizer = SimpleLAMTokenizer(model_config.vocab_size)

    # Active trainable policy
    model = JEDM2Model(model_config).to(device=device, dtype=dtype)
    if args.model_path:
        print(f"[Init] Loading policy weights from: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[Init] Policy loaded successfully (missing keys: {len(missing)}, unexpected: {len(unexpected)})")

    if not args.no_checkpointing:
        model.gradient_checkpointing_enable()

    # Frozen reference policy for KL divergence regularization
    ref_model = JEDM2Model(model_config).to(device=device, dtype=dtype)
    ref_model.load_state_dict(model.state_dict())
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # Optimizer for active policy
    trainer_cfg = JEDM2TrainerConfig(lr=args.lr, weight_decay=args.weight_decay)
    optimizer, opt_type = build_optimizer(model, config=trainer_cfg, device=device)

    # Interactive Rollout Sampler
    sampler = TrajectoryRolloutSampler(
        model=model,
        config=model_config,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )

    # --------------------------------------------------------------------------
    # 3. RL Training Execution Loop
    # --------------------------------------------------------------------------
    if args.mode == "grpo":
        grpo_config = GRPOConfig(
            group_size=args.group_size,
            clip_eps=args.clip_eps,
            beta_kl=args.beta_kl,
            temperature=args.temperature,
        )
        trainer = GRPOTrainer(
            model=model,
            ref_model=ref_model,
            config=grpo_config,
            optimizer=optimizer,
        )

        print("[Header] Iteration | GRPO Loss | Mean Rew | Mean Adv | Mean KL | Clip Frac | Grad Norm | Tok/s | Peak VRAM")
        print("-" * 105)

        for iteration in range(1, args.iterations + 1):
            t_start = time.time()
            scenario_id = scenarios[(iteration - 1) % len(scenarios)]

            # 1. Sample group of G rollouts in interactive environment
            group = sampler.sample_group_rollouts(
                scenario_id=scenario_id,
                group_size=args.group_size,
                temperature=args.temperature,
                max_turns=args.max_turns,
            )

            # 2. Collate rollouts into batched training tensors
            batch = sampler.collate_rollouts(group)

            # 3. Active Policy Gradient Step
            model.train()
            optimizer.zero_grad()

            loss, metrics = trainer.compute_loss(
                input_ids=batch["input_ids"],
                targets=batch["targets"],
                mask=batch["mask"],
                old_token_logps=batch["old_token_logps"],
                rewards=batch["rewards"],
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            # Telemetry calculations
            elapsed = time.time() - t_start
            active_tokens = int(batch["mask"].sum().item())
            throughput = active_tokens / max(1e-4, elapsed)

            peak_vram_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            )

            if iteration % args.log_interval == 0 or iteration == args.iterations:
                print(
                    f"[{iteration:03d}/{args.iterations:03d}]  "
                    f"Loss: {metrics['loss_grpo']:7.4f} | "
                    f"Rew: {metrics['mean_reward']:+6.3f} | "
                    f"Adv: {metrics['mean_advantage']:+6.3f} | "
                    f"KL: {metrics['mean_kl']:7.4f} | "
                    f"Clip: {metrics['clip_fraction'] * 100:5.1f}% | "
                    f"Grad: {grad_norm:5.2f} | "
                    f"{throughput:5.1f} t/s | "
                    f"{peak_vram_mb:6.1f} MB"
                )

            # Checkpoint saving
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                ckpt_path = os.path.join(args.checkpoint_dir, f"jed_m2_grpo_step_{iteration}.pt")
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    ckpt_path,
                )

    elif args.mode == "dpo":
        dpo_config = DPOConfig(beta=args.beta_dpo)
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            config=dpo_config,
            optimizer=optimizer,
        )

        print("[Header] Iteration | DPO Loss  | Margin  | Accuracy | Chosen Rew | Rej Rew  | Grad Norm | Tok/s | Peak VRAM")
        print("-" * 105)

        for iteration in range(1, args.iterations + 1):
            t_start = time.time()
            scenario_id = scenarios[(iteration - 1) % len(scenarios)]

            # Sample rollouts until winning/losing preference pair is formed
            group = sampler.sample_group_rollouts(
                scenario_id=scenario_id,
                group_size=max(4, args.group_size),
                temperature=max(args.temperature, 0.8),
                max_turns=args.max_turns,
            )
            pairs = sampler.create_preference_pairs(group)

            if not pairs:
                # If no natural difference, fallback to highest vs lowest reward
                sorted_g = sorted(group, key=lambda t: t.cumulative_reward, reverse=True)
                pairs = [(sorted_g[0], sorted_g[-1])]

            pref_batch = sampler.collate_preference_pairs(pairs)

            model.train()
            optimizer.zero_grad()

            loss, metrics = trainer.compute_loss(
                chosen_input_ids=pref_batch["chosen_input_ids"],
                chosen_targets=pref_batch["chosen_targets"],
                chosen_mask=pref_batch["chosen_mask"],
                rejected_input_ids=pref_batch["rejected_input_ids"],
                rejected_targets=pref_batch["rejected_targets"],
                rejected_mask=pref_batch["rejected_mask"],
            )

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            elapsed = time.time() - t_start
            active_tokens = int(
                (pref_batch["chosen_mask"].sum() + pref_batch["rejected_mask"].sum()).item()
            )
            throughput = active_tokens / max(1e-4, elapsed)

            peak_vram_mb = (
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            )

            if iteration % args.log_interval == 0 or iteration == args.iterations:
                print(
                    f"[{iteration:03d}/{args.iterations:03d}]  "
                    f"Loss: {metrics['loss_dpo']:7.4f} | "
                    f"Margin: {metrics['reward_margin']:+6.3f} | "
                    f"Acc: {metrics['dpo_accuracy']:5.2f} | "
                    f"Ch: {metrics['chosen_reward_mean']:+6.3f} | "
                    f"Rej: {metrics['rejected_reward_mean']:+6.3f} | "
                    f"Grad: {grad_norm:5.2f} | "
                    f"{throughput:5.1f} t/s | "
                    f"{peak_vram_mb:6.1f} MB"
                )

            if iteration % args.save_interval == 0 or iteration == args.iterations:
                ckpt_path = os.path.join(args.checkpoint_dir, f"jed_m2_dpo_step_{iteration}.pt")
                torch.save(
                    {
                        "iteration": iteration,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "metrics": metrics,
                    },
                    ckpt_path,
                )

    print("\n" + "=" * 80)
    print("      RL TRAINING RUN COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
