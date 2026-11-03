"""
Fine-Tuning Runner for Small Language Model Baselines (T5-Small & DistilGPT-2).
Trains baseline models on verified Mind2Web site-held-out training manifest
under identical hardware and optimization budgets.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from jed_m2.core.types import UIElement
from jed_m2.heads import ActionType
from jed_m2.real_environment import RealTaskDOMEnvironment
from jed_m2.small_lm_baselines import (
    DOMActionSequenceFormatter,
    DistilGPT2ActionAgent,
    T5SmallActionAgent,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_baseline_lm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JED-M2 Small-LM Baseline Fine-Tuning")
    parser.add_argument(
        "--model_type",
        type=str,
        default="t5_small",
        choices=["t5_small", "distilgpt2"],
        help="Target baseline architecture: t5_small (~60M) or distilgpt2 (~82M)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/manifests/real_mind2web_v1/manifest.json",
        help="Path to verified site-held-out manifest JSON file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Manifest split to train on.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum training optimization steps.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per forward pass.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for AdamW optimizer.",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=50,
        help="Maximum task episodes to ingest from manifest.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Target hardware device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",
        choices=["bfloat16", "float32"],
        help="Precision format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where fine-tuned checkpoint will be saved.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Verify dataset ingestion and single batch forward pass then exit.",
    )
    return parser.parse_args()


def collect_training_samples(
    env: RealTaskDOMEnvironment,
    formatter: DOMActionSequenceFormatter,
    max_samples: int = 500,
) -> List[Tuple[str, str]]:
    """Ingests interaction turns from RealTaskDOMEnvironment and produces (prompt, target) pairs."""
    samples: List[Tuple[str, str]] = []
    logger.info("Ingesting training samples from %d environment tasks...", len(env.tasks))

    for task_idx in range(len(env.tasks)):
        task = env.tasks[task_idx]
        goal = task.goal
        history: List[str] = []

        for turn in task.turns:
            elements: List[UIElement] = []
            if turn.html:
                # Fast parse elements
                from jed_m2.real_dataset import FastDOMParser
                parser = FastDOMParser(max_elements=formatter.max_elements)
                parser.feed(turn.html)
                elements = parser.elements

            # Fallback if no elements parsed
            if not elements:
                cx, cy = (turn.target_bbox[0] + turn.target_bbox[2]) / 2.0, (turn.target_bbox[1] + turn.target_bbox[3]) / 2.0
                elements = [
                    UIElement(
                        element_id=0,
                        tag=turn.target_tag or "button",
                        role=turn.target_role or "button",
                        text=turn.target_text or "Target",
                        bbox=turn.target_bbox,
                    )
                ]

            prompt = formatter.format_turn_prompt(goal, elements, history)
            cx, cy = (turn.target_bbox[0] + turn.target_bbox[2]) / 2.0, (turn.target_bbox[1] + turn.target_bbox[3]) / 2.0
            target = formatter.format_target_action(
                action_type=turn.action_type,
                coordinates=(cx, cy),
                argument=turn.argument,
                thought=turn.thought,
            )
            samples.append((prompt, target))
            history.append(f"{turn.action_type.name} ({cx:.2f}, {cy:.2f})")

            if len(samples) >= max_samples:
                break
        if len(samples) >= max_samples:
            break

    logger.info("Collected %d training sequence pairs.", len(samples))
    return samples


def compute_eval_loss(
    agent,
    eval_samples: List[Tuple[str, str]],
    device: torch.device,
    batch_size: int = 4,
    max_eval_batches: int = 10,
) -> float:
    """Computes mean sequence cross-entropy loss over a validation sample set."""
    if not eval_samples:
        return 0.0
    agent.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, min(len(eval_samples), max_eval_batches * batch_size), batch_size):
            batch = eval_samples[i : i + batch_size]
            prompts = [s[0] for s in batch]
            targets = [s[1] for s in batch]
            loss = agent.compute_loss(prompts, targets, device=device)
            total_loss += loss.item()
            n_batches += 1
    agent.train()
    return total_loss / max(1, n_batches)


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    output_dir = args.output_dir or f"checkpoints/baseline_{args.model_type}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"JED-M2 Small-LM Baseline Training: {args.model_type.upper()}")
    print(f"Manifest    : {args.manifest} (split: {args.split})")
    print(f"Device      : {args.device} ({args.dtype})")
    print(f"Batch Size  : {args.batch_size} (accum: {args.grad_accum_steps})")
    print(f"Output Dir  : {output_dir}")
    print("=" * 80)

    # 1. Initialize Baseline Agent
    formatter = DOMActionSequenceFormatter(max_elements=24)
    if args.model_type == "t5_small":
        agent = T5SmallActionAgent(local_files_only=True)
    elif args.model_type == "distilgpt2":
        agent = DistilGPT2ActionAgent(local_files_only=True)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    agent.to(device)
    if dtype == torch.bfloat16 and args.device == "cuda":
        agent.model.to(dtype=torch.bfloat16)

    print(f"Model Initialized. Total Parameters: {agent.total_parameters:,}")

    # 2. Ingest Training Manifest
    env = RealTaskDOMEnvironment(
        manifest_path=args.manifest,
        split=args.split,
        real_only=False,
        max_tasks=args.max_tasks,
    )
    samples = collect_training_samples(env, formatter, max_samples=args.max_steps * args.batch_size * 2)

    if not samples:
        raise RuntimeError("No training samples could be extracted from manifest.")

    if args.dry_run:
        batch_p = [s[0] for s in samples[:2]]
        batch_t = [s[1] for s in samples[:2]]
        test_loss = agent.compute_loss(batch_p, batch_t, device=device)
        print(f"[DryRun] Successfully computed initial forward loss: {test_loss.item():.4f}")
        return

    # 3. Optimization Setup
    optimizer = AdamW(agent.model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=args.lr * 0.1)

    # 4. Validation Baseline
    val_env = RealTaskDOMEnvironment(
        manifest_path=args.manifest,
        split="val",
        real_only=False,
        max_tasks=15,
    )
    val_samples = collect_training_samples(val_env, formatter, max_samples=40)
    init_val_loss = compute_eval_loss(agent, val_samples, device=device)
    print(f"Pre-Training Initial Validation Loss: {init_val_loss:.4f}")

    agent.train()
    print("\nStarting optimization loop...")
    print("-" * 80)
    print("Step    | Loss      | LR        | Elapsed   | Peak VRAM")
    print("-" * 80)

    sample_idx = 0
    t0 = time.time()
    accum_loss = 0.0
    history_losses = []

    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(args.grad_accum_steps):
            batch_slice = []
            for _ in range(args.batch_size):
                batch_slice.append(samples[sample_idx % len(samples)])
                sample_idx += 1

            prompts = [s[0] for s in batch_slice]
            targets = [s[1] for s in batch_slice]

            loss = agent.compute_loss(prompts, targets, device=device)
            scaled_loss = loss / args.grad_accum_steps
            scaled_loss.backward()
            step_loss += scaled_loss.item()

        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        accum_loss += step_loss
        history_losses.append(step_loss)

        if step % 10 == 0 or step == args.max_steps:
            elapsed = time.time() - t0
            curr_lr = scheduler.get_last_lr()[0]
            peak_vram = (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if args.device == "cuda"
                else 0.0
            )
            print(f"{step:6d}  | {step_loss:9.4f} | {curr_lr:.2e}  | {elapsed:7.1f}s  | {peak_vram:6.1f} MB")

    # 5. Final Validation & Checkpoint Serialization
    final_val_loss = compute_eval_loss(agent, val_samples, device=device)
    print("-" * 80)
    print(f"Post-Training Final Validation Loss: {final_val_loss:.4f} (Delta: {final_val_loss - init_val_loss:+.4f})")

    agent.save_pretrained(output_dir)
    meta = {
        "model_type": args.model_type,
        "manifest": args.manifest,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "lr": args.lr,
        "initial_train_loss": history_losses[0] if history_losses else None,
        "final_train_loss": history_losses[-1] if history_losses else None,
        "initial_val_loss": init_val_loss,
        "final_val_loss": final_val_loss,
        "elapsed_seconds": time.time() - t0,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if args.device == "cuda"
            else 0.0
        ),
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved fine-tuned baseline model and metadata to: {output_dir}")


if __name__ == "__main__":
    main()
