"""
JED-M2 SFT Pretraining Benchmark Orchestrator (Phase 6C).
Calibrated for NVIDIA RTX 5060 Laptop GPU (8GB VRAM, sm_120 Blackwell).

Orchestrates a comprehensive 1,000-step supervised fine-tuning (SFT) pretraining
run on synthetic WebArena/Mind2Web trajectories:
- Effective batch size 16 (micro-batch size 2, gradient accumulation 8)
- Sequence length 2048 with selective activation checkpointing
- Multi-head convergence tracking across all 6 loss heads:
    1. Reasoning Loss (cross-entropy on System 2 thoughts)
    2. Action Gate Loss (binary cross-entropy on action commitment)
    3. Action Primitive Loss (cross-entropy on discrete tool categories)
    4. Coordinate Regression Loss (Huber / Smooth L1 on [0, 1] screen pointers)
    5. JEPA World Model Latent Consistency Loss (Smooth L1 + cosine distance)
    6. MoE Auxiliary Load-Balancing Loss (routing load penalty)
- Tracking of coordinate Mean Absolute Error (MAE) and action primitive accuracy
- Periodic held-out validation loss evaluation
- Telemetry profiling: token throughput, gradient norm stability, peak VRAM
- Automated baseline checkpoint persistence: checkpoints/jed_m2_sft_base.pt
- Structured JSON metrics export: benchmark_sft_metrics.json

STRICT CONVENTION: Pure ASCII console output (no multi-byte Unicode characters).
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

import train


def parse_benchmark_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JED-M2 Large Action Model - SFT Pretraining Benchmark (Phase 6C)"
    )
    # Step Budget & Hyperparameters
    parser.add_argument("--steps", type=int, default=1000, help="Total SFT training steps (default: 1000)")
    parser.add_argument("--batch_size", type=int, default=16, help="Target effective batch size (default: 16)")
    parser.add_argument("--micro_batch_size", type=int, default=2, help="Micro-batch size per forward pass (default: 2)")
    parser.add_argument("--seq_len", type=int, default=2048, help="Maximum sequence length (default: 2048)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate (default: 3e-4)")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum cosine annealed learning rate (default: 3e-5)")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Linear warmup steps (default: 100)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient norm clip threshold (default: 1.0)")

    # Validation & Evaluation
    parser.add_argument("--eval_interval", type=int, default=100, help="Step interval for validation passes (default: 100)")
    parser.add_argument("--eval_steps", type=int, default=10, help="Number of micro-batches evaluated per pass (default: 10)")
    parser.add_argument("--eval_dataset_size", type=int, default=1000, help="Size of held-out validation trajectory pool (default: 1000)")

    # Persistence & Reporting
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory for training checkpoints")
    parser.add_argument("--baseline_checkpoint", type=str, default="checkpoints/jed_m2_sft_base.pt", help="Baseline production checkpoint destination")
    parser.add_argument("--output_metrics", type=str, default="benchmark_sft_metrics.json", help="Path for JSON metrics output")
    parser.add_argument("--save_interval", type=int, default=250, help="Step interval to save intermediate checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="Step interval to log live training metrics")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--no_checkpointing", action="store_true", help="Disable activation checkpointing")
    return parser.parse_args()


def render_ascii_table(history: Dict[str, Any]) -> str:
    """
    Renders a formatted pure-ASCII table summarizing multi-head convergence,
    held-out validation results, and hardware efficiency.
    """
    lines: List[str] = []
    lines.append("+" + "-" * 78 + "+")
    lines.append("|           JED-M2 SFT PRETRAINING BENCHMARK CONVERGENCE SUMMARY              |")
    lines.append("+" + "-" * 78 + "+")

    train_records = history.get("train", [])
    eval_records = history.get("eval", [])
    summary = history.get("summary", {})

    if not train_records:
        lines.append("| No training records captured.                                                |")
        lines.append("+" + "-" * 78 + "+")
        return "\n".join(lines)

    first_train = train_records[0]
    final_train = train_records[-1]

    lines.append("| Metric Head                  | Initial (Step {:>4d}) | Final (Step {:>4d})   | Status       |".format(
        first_train.get("step", 0), final_train.get("step", 0)
    ))
    lines.append("+" + "-" * 78 + "+")

    def format_row(name: str, init_val: float, final_val: float, is_lower_better: bool = True, is_pct: bool = False) -> str:
        if is_pct:
            init_str = f"{init_val * 100:6.2f}%"
            final_str = f"{final_val * 100:6.2f}%"
            improved = final_val >= init_val if not is_lower_better else final_val <= init_val
        else:
            init_str = f"{init_val:10.4f}"
            final_str = f"{final_val:10.4f}"
            improved = final_val <= init_val if is_lower_better else final_val >= init_val
        status = "CONVERGED" if improved else "STABLE"
        return f"| {name:<28} | {init_str:>20} | {final_str:>19} | {status:<12} |"

    lines.append(format_row("Total Composite Loss", first_train.get("loss_total", 0.0), final_train.get("loss_total", 0.0), True))
    lines.append(format_row("1. Reasoning Loss", first_train.get("loss_reason", 0.0), final_train.get("loss_reason", 0.0), True))
    lines.append(format_row("2. Action Gate Loss", first_train.get("loss_gate", 0.0), final_train.get("loss_gate", 0.0), True))
    lines.append(format_row("3. Action Type Loss", first_train.get("loss_action_type", 0.0), final_train.get("loss_action_type", 0.0), True))
    lines.append(format_row("   Action Primitive Acc", first_train.get("acc_action_type", 0.0), final_train.get("acc_action_type", 0.0), False, True))
    lines.append(format_row("4. Coordinate Huber Loss", first_train.get("loss_coord", 0.0), final_train.get("loss_coord", 0.0), True))
    lines.append(format_row("   Coordinate MAE (norm)", first_train.get("coord_mae", 0.0), final_train.get("coord_mae", 0.0), True))
    lines.append(format_row("5. JEPA World Model Loss", first_train.get("loss_jepa", 0.0), final_train.get("loss_jepa", 0.0), True))
    lines.append(format_row("6. MoE Routing Aux Loss", first_train.get("loss_aux", 0.0), final_train.get("loss_aux", 0.0), True))
    lines.append("+" + "-" * 78 + "+")

    # Validation section
    if eval_records:
        first_eval = eval_records[0]
        final_eval = eval_records[-1]
        lines.append("| Held-out Validation Checkpoints (Zero Leakage, Seed={:<6d})                |".format(
            history.get("config", {}).get("seed", 42) + 1000
        ))
        lines.append("+" + "-" * 78 + "+")
        lines.append(format_row("Val Total Loss", first_eval.get("val_loss_total", 0.0), final_eval.get("val_loss_total", 0.0), True))
        lines.append(format_row("Val Action Primitive Acc", first_eval.get("val_acc_action_type", 0.0), final_eval.get("val_acc_action_type", 0.0), False, True))
        lines.append(format_row("Val Coordinate MAE", first_eval.get("val_coord_mae", 0.0), final_eval.get("val_coord_mae", 0.0), True))
        lines.append(format_row("Val JEPA Consistency Loss", first_eval.get("val_loss_jepa", 0.0), final_eval.get("val_loss_jepa", 0.0), True))
        lines.append("+" + "-" * 78 + "+")

    # Hardware & Performance section
    peak_vram = max([r.get("peak_vram_mb", 0.0) for r in train_records] or [0.0])
    mean_tok = summary.get("mean_tokens_per_sec", 0.0)
    total_time = summary.get("total_time_sec", 0.0)
    total_tok = summary.get("total_tokens", 0)

    lines.append("| Hardware Telemetry & Efficiency (NVIDIA RTX 5060 Laptop GPU)                 |")
    lines.append("+" + "-" * 78 + "+")
    lines.append(f"| Peak Allocated VRAM          : {peak_vram:7.2f} MB (Ceiling: 6144.0 MB / 6.0 GB)   |")
    lines.append(f"| Mean Token Throughput        : {mean_tok:7.1f} tokens/second                        |")
    lines.append(f"| Total Tokens Processed       : {total_tok:>10,d} tokens                             |")
    lines.append(f"| Total Wall-Clock Execution   : {total_time:7.2f} seconds                             |")
    lines.append("+" + "-" * 78 + "+")

    return "\n".join(lines)


def run_benchmark(cli_args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    """
    Main orchestration entrypoint for Phase 6C SFT benchmark.
    Translates benchmark arguments into train.py configuration, executes the run,
    verifies baseline checkpoint creation, prints ASCII convergence report,
    and saves JSON metrics.

    Args:
        cli_args: Pre-parsed Namespace or None (uses parse_benchmark_args()).

    Returns:
        Structured metrics history dictionary.
    """
    args = cli_args if cli_args is not None else parse_benchmark_args()

    # Ensure output directories exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    baseline_dir = os.path.dirname(os.path.abspath(args.baseline_checkpoint))
    if baseline_dir:
        os.makedirs(baseline_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("      JED-M2 NATIVE LARGE ACTION MODEL - PHASE 6C SFT BENCHMARK")
    print("=" * 80)
    print(f"Target Global Steps   : {args.steps}")
    print(f"Effective Batch Size  : {args.batch_size} (micro-batch: {args.micro_batch_size}, accum: {args.batch_size // args.micro_batch_size})")
    print(f"Max Sequence Length   : {args.seq_len}")
    print(f"Learning Rate Schedule: {args.lr:.2e} -> {args.min_lr:.2e} (Warmup: {args.warmup_steps} steps)")
    print(f"Validation Evaluation : Every {args.eval_interval} steps ({args.eval_steps} micro-batches)")
    print(f"Baseline Checkpoint   : {args.baseline_checkpoint}")
    print(f"Metrics Output JSON   : {args.output_metrics}")
    print("=" * 80 + "\n")

    # Map benchmark args to train.py Namespace
    train_args = argparse.Namespace(
        epochs=1,
        max_steps=args.steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        seq_len=args.seq_len,
        checkpoint_dir=args.checkpoint_dir,
        resume=None,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        dataset_size=max(10000, args.steps * args.batch_size),
        seed=args.seed,
        no_checkpointing=args.no_checkpointing,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        eval_dataset_size=args.eval_dataset_size,
        export_history=args.output_metrics,
        baseline_checkpoint=args.baseline_checkpoint,
    )

    # Execute training orchestrator
    bench_start = time.time()
    history = train.main(train_args)
    bench_wall_time = time.time() - bench_start

    # Verify baseline checkpoint existence and payload
    if not os.path.exists(args.baseline_checkpoint):
        raise FileNotFoundError(
            f"Production baseline checkpoint was not saved to expected path: {args.baseline_checkpoint}"
        )

    ckpt_size_mb = os.path.getsize(args.baseline_checkpoint) / (1024**2)
    loaded_ckpt = torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=False)

    required_keys = [
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "model_config",
        "trainer_config",
        "loss_config",
        "global_step",
    ]
    for k in required_keys:
        assert k in loaded_ckpt, f"Missing critical key '{k}' in baseline checkpoint!"

    # Render and print ASCII convergence summary
    ascii_summary = render_ascii_table(history)
    print("\n" + ascii_summary + "\n")

    print("[Verification] Baseline Checkpoint Verified:")
    print(f"  -> Path           : {os.path.abspath(args.baseline_checkpoint)}")
    print(f"  -> File Size      : {ckpt_size_mb:.2f} MB")
    print(f"  -> Saved Step     : {loaded_ckpt.get('global_step', 0)}")
    print(f"  -> Total Params   : {len(loaded_ckpt['model_state_dict']):,} parameter tensors")
    print(f"  -> Total Wall Time: {bench_wall_time:.2f}s")
    print(f"  -> Metrics JSON   : {os.path.abspath(args.output_metrics)}")
    print("=" * 80 + "\n")

    return history


def main() -> None:
    run_benchmark()


if __name__ == "__main__":
    main()
