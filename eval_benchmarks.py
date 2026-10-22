"""
CLI Benchmark Runner and Comparative Evaluation Script for JED-M2.
Provides:
- Configurable CLI flags:
  --benchmark {mind2web,webarena,simulated}
  --model_path <path>
  --checkpoints <path1> <path2> ...
  --split {val,test}
  --num_episodes 50
  --max_steps 15
  --device {cuda,cpu}
  --dtype {bfloat16,float32}
  --output_report BENCHMARK_REPORT.md
  --seed 42
- Side-by-side comparative evaluation of multiple model checkpoints on identical task splits.
- Clean 7-bit ASCII terminal summary tables.
- Automated export of formal Markdown benchmark report.

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import torch

from jed_m2.agent import JEDM2Agent
from jed_m2.benchmark import (
    BenchmarkSummary,
    TaskEvaluator,
    evaluate_offline_dataset,
    evaluate_online_environment,
)
from jed_m2.config import JEDM2Config, JEDM2MediumConfig
from jed_m2.dataset import (
    AgentTrajectoryDataset,
    SimpleLAMTokenizer,
    SyntheticTrajectoryGenerator,
    TrajectorySample,
)
from jed_m2.environment import ScenarioDefinition, WebOSEnvironment, get_default_scenarios
from jed_m2.model import JEDM2Model
from jed_m2.real_dataset import RealWebTrajectoryDataset


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for benchmark runner."""
    parser = argparse.ArgumentParser(
        description="JED-M2 Standardized Benchmark Evaluation Runner & Comparative Analyzer"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="simulated",
        choices=["mind2web", "webarena", "simulated", "real_web"],
        help="Target benchmark suite to evaluate: mind2web, webarena, simulated Web/OS sandbox, or real_web (offline HTML DOM traces).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a single model checkpoint (.pt).",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help="List of checkpoint paths to evaluate side-by-side (e.g. SFT Base vs GRPO Step 40).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate: val or test.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Total number of episodes to evaluate per checkpoint.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=15,
        help="Maximum interaction turns per episode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Execution hardware device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Computation precision.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="nano",
        choices=["nano", "medium"],
        help="Model scale tier: nano (~53M params) or medium (~273M params)",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="BENCHMARK_REPORT.md",
        help="Output Markdown report filename.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Spatial coordinate matching tolerance radius (epsilon in [0, 1]).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic benchmark trajectory generation.",
    )
    parser.add_argument(
        "--use_jepa_planning",
        action="store_true",
        help="Enable active JEPA model-predictive latent rollout planning in agent decision loop.",
    )
    parser.add_argument(
        "--jepa_lambda",
        type=float,
        default=1.0,
        help="Planning distance penalty factor lambda for candidate action scoring.",
    )
    parser.add_argument(
        "--raw_mind2web_dir",
        type=str,
        default="data/raw_mind2web",
        help="Path to directory containing raw Mind2Web task annotations and DOM snapshots.",
    )
    return parser.parse_args()


def load_model(
    checkpoint_path: Optional[str],
    device: torch.device,
    dtype: torch.dtype,
    model_size: str = "nano",
) -> Tuple[JEDM2Model, str]:
    """
    Loads JEDM2Model from a checkpoint file or creates an initialized baseline model.

    Args:
        checkpoint_path: Path to checkpoint .pt file or None.
        device: Hardware device.
        dtype: PyTorch data type.
        model_size: Default model size tier if not specified in checkpoint.

    Returns:
        Tuple of (model, checkpoint_label).
    """
    if checkpoint_path is None or checkpoint_path.lower() == "random" or not os.path.exists(checkpoint_path):
        config = JEDM2MediumConfig() if model_size == "medium" else JEDM2Config()
        model = JEDM2Model(config)
        default_label = "Untrained Medium" if model_size == "medium" else "Untrained Base"
        label = default_label if checkpoint_path is None else os.path.basename(checkpoint_path)
    else:
        # Load checkpoint payload
        data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = data.get("model_state_dict", data) if isinstance(data, dict) else data

        # Robustly auto-detect model architecture from metadata, path, or tensor dimensions
        detected_size = model_size
        if isinstance(data, dict):
            if data.get("model_size") in ("nano", "medium"):
                detected_size = data["model_size"]
            elif "model_config" in data and isinstance(data["model_config"], dict):
                if data["model_config"].get("hidden_dim") == 768 or data["model_config"].get("num_layers") == 16:
                    detected_size = "medium"
            elif "config" in data and isinstance(data["config"], dict):
                if data["config"].get("hidden_dim") == 768 or data["config"].get("num_layers") == 16:
                    detected_size = "medium"

        if "medium" in str(checkpoint_path).lower():
            detected_size = "medium"
        elif "nano" in str(checkpoint_path).lower():
            detected_size = "nano"
        elif isinstance(state_dict, dict) and "tok_embeddings.weight" in state_dict:
            if state_dict["tok_embeddings.weight"].shape[-1] == 768:
                detected_size = "medium"
            elif state_dict["tok_embeddings.weight"].shape[-1] == 512:
                detected_size = "nano"

        config = JEDM2MediumConfig() if detected_size == "medium" else JEDM2Config()
        model = JEDM2Model(config)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())

        # If key schema differs, remap transparently
        if not ckpt_keys.issubset(model_keys):
            remapped_sd = {}
            for k, v in state_dict.items():
                k_new = k
                # If model expects mla/moe but ckpt has attn/ffn
                if any(".norm_mla." in mk for mk in model_keys) and ".norm_attn." in k:
                    k_new = k_new.replace(".norm_attn.", ".norm_mla.")
                if any(".mla." in mk for mk in model_keys) and ".attn." in k:
                    k_new = k_new.replace(".attn.", ".mla.")
                if any(".norm_moe." in mk for mk in model_keys) and ".norm_ffn." in k:
                    k_new = k_new.replace(".norm_ffn.", ".norm_moe.")
                if any(".moe." in mk for mk in model_keys) and ".ffn." in k:
                    k_new = k_new.replace(".ffn.", ".moe.")
                # If model expects attn/ffn but ckpt has mla/moe
                if any(".norm_attn." in mk for mk in model_keys) and ".norm_mla." in k:
                    k_new = k_new.replace(".norm_mla.", ".norm_attn.")
                if any(".attn." in mk for mk in model_keys) and ".mla." in k:
                    k_new = k_new.replace(".mla.", ".attn.")
                if any(".norm_ffn." in mk for mk in model_keys) and ".norm_moe." in k:
                    k_new = k_new.replace(".norm_moe.", ".norm_ffn.")
                if any(".ffn." in mk for mk in model_keys) and ".moe." in k:
                    k_new = k_new.replace(".moe.", ".ffn.")
                remapped_sd[k_new] = v
            state_dict = remapped_sd

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[load_model] Loaded with missing keys: {len(missing)}, unexpected: {len(unexpected)}")
        label = os.path.splitext(os.path.basename(checkpoint_path))[0]

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, label


def create_offline_dataset(
    benchmark_name: str,
    split: str,
    num_episodes: int,
    seed: int,
) -> List[TrajectorySample]:
    """
    Constructs a deterministic synthetic Mind2Web or WebArena evaluation trajectory dataset.

    Args:
        benchmark_name: 'mind2web' or 'webarena'.
        split: 'val' or 'test'.
        num_episodes: Number of trajectories.
        seed: Random seed.

    Returns:
        List of TrajectorySample objects.
    """
    config = JEDM2Config()
    # Vary seed by split to avoid data leakage
    split_offset = 1000 if split == "val" else 5000
    generator = SyntheticTrajectoryGenerator(config=config, seed=seed + split_offset)

    dataset: List[TrajectorySample] = []
    for i in range(num_episodes):
        sample = generator.generate_trajectory(scenario_idx=i)
        sample.sample_id = f"{benchmark_name}_{split}_{i}"
        dataset.append(sample)
    return dataset


def load_real_web_dataset(
    split: str,
    num_episodes: int,
    seed: int,
    raw_mind2web_dir: Optional[str] = "data/raw_mind2web",
) -> List[Any]:
    """
    Loads deterministic real-world web interaction trajectory episodes from RealWebTrajectoryDataset.

    Args:
        split: 'val' or 'test'.
        num_episodes: Number of trajectories.
        seed: Random seed.
        raw_mind2web_dir: Directory containing raw Mind2Web tasks and snapshots.

    Returns:
        List of RealWebTrajectorySample objects.
    """
    config = JEDM2Config()
    dataset = RealWebTrajectoryDataset(
        config=config,
        split=split,
        dataset_size=max(num_episodes, 20),
        raw_mind2web_dir=raw_mind2web_dir,
        seed=seed,
    )
    return dataset.samples[:num_episodes]


def run_benchmark_for_checkpoint(
    checkpoint_path: Optional[str],
    benchmark_name: str,
    split: str,
    num_episodes: int,
    max_steps: int,
    device_str: str,
    dtype_str: str,
    tolerance: float,
    seed: int,
    offline_trajectories: Optional[List[Any]] = None,
    model_size: str = "nano",
    use_jepa_planning: bool = False,
    jepa_lambda: float = 1.0,
    raw_mind2web_dir: Optional[str] = "data/raw_mind2web",
) -> BenchmarkSummary:
    """
    Runs benchmark evaluation for a single model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file or None.
        benchmark_name: 'mind2web', 'webarena', 'real_web', or 'simulated'.
        split: 'val' or 'test'.
        num_episodes: Total episodes.
        max_steps: Maximum turns per episode.
        device_str: 'cuda' or 'cpu'.
        dtype_str: 'bfloat16' or 'float32'.
        tolerance: Spatial tolerance radius.
        seed: Random seed.
        offline_trajectories: Pre-generated trajectory list if running offline evaluation.
        model_size: Model size tier ('nano' or 'medium').
        use_jepa_planning: Enable active model-predictive JEPA planning.
        jepa_lambda: Planning distance penalty weight.
        raw_mind2web_dir: Directory containing raw Mind2Web data.

    Returns:
        BenchmarkSummary for this checkpoint.
    """
    device = torch.device(device_str)
    dtype = torch.bfloat16 if dtype_str == "bfloat16" and "cuda" in device_str else torch.float32

    # Load model
    model, label = load_model(checkpoint_path, device=device, dtype=dtype, model_size=model_size)
    tokenizer = SimpleLAMTokenizer(model.config.vocab_size)
    agent = JEDM2Agent(
        model=model,
        config=model.config,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        use_jepa_planning=use_jepa_planning,
        jepa_planning_lambda=jepa_lambda,
    )
    evaluator = TaskEvaluator(tolerance_radius=tolerance)

    if benchmark_name in ("real_web", "mind2web"):
        # Real-world HTML DOM trajectory replay
        if offline_trajectories is None:
            offline_trajectories = load_real_web_dataset(
                split=split,
                num_episodes=num_episodes,
                seed=seed,
                raw_mind2web_dir=raw_mind2web_dir,
            )

        summary = evaluate_offline_dataset(
            agent=agent,
            trajectories=offline_trajectories,
            evaluator=evaluator,
            benchmark_name=f"{benchmark_name}_{split}",
            checkpoint_name=label,
            max_episodes=num_episodes,
        )
    elif benchmark_name == "webarena":
        # Offline dataset replay
        if offline_trajectories is None:
            offline_trajectories = create_offline_dataset(
                benchmark_name=benchmark_name,
                split=split,
                num_episodes=num_episodes,
                seed=seed,
            )

        summary = evaluate_offline_dataset(
            agent=agent,
            trajectories=offline_trajectories,
            evaluator=evaluator,
            benchmark_name=f"{benchmark_name}_{split}",
            checkpoint_name=label,
            max_episodes=num_episodes,
        )
    else:
        # Online closed-loop simulation in WebOSEnvironment
        summary = evaluate_online_environment(
            agent=agent,
            evaluator=evaluator,
            benchmark_name="webos_simulated",
            checkpoint_name=label,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed,
        )

    # Clean up model and clear GPU cache
    del agent, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    """Main CLI execution entrypoint."""
    args = parse_args()

    # Determine checkpoint list
    checkpoints: List[Optional[str]] = []
    if args.checkpoints:
        checkpoints.extend(args.checkpoints)
    elif args.model_path:
        checkpoints.append(args.model_path)
    else:
        # Default fallback: check if baseline SFT checkpoint exists
        default_sft = os.path.join("checkpoints", "jed_m2_sft_base.pt")
        if os.path.exists(default_sft):
            checkpoints.append(default_sft)
        else:
            checkpoints.append(None)

    print("=" * 80)
    print(" JED-M2 Large Action Model - Benchmark Evaluation Suite".center(80))
    print("=" * 80)
    print(f"Benchmark Suite : {args.benchmark} (split={args.split})")
    print(f"Episodes        : {args.num_episodes}")
    print(f"Max Steps/Ep    : {args.max_steps}")
    print(f"Device          : {args.device} (dtype={args.dtype})")
    print(f"Tolerance Radius: {args.tolerance}")
    print(f"Checkpoints ({len(checkpoints)}):")
    for cp in checkpoints:
        print(f"  - {cp if cp else '[Untrained Baseline]'}")
    print("=" * 80)

    # If offline dataset, pre-generate once so all checkpoints are tested on identical episodes
    offline_trajectories: Optional[List[Any]] = None
    if args.benchmark in ("real_web", "mind2web"):
        print(f"Pre-loading {args.num_episodes} real-world web / Mind2Web ({args.split}) trajectory episodes...")
        offline_trajectories = load_real_web_dataset(
            split=args.split,
            num_episodes=args.num_episodes,
            seed=args.seed,
            raw_mind2web_dir=args.raw_mind2web_dir,
        )
        print(f"Real web / Mind2Web dataset ready: {len(offline_trajectories)} episodes.\n")
    elif args.benchmark == "webarena":
        print(f"Pre-generating {args.num_episodes} {args.benchmark} ({args.split}) trajectory episodes...")
        offline_trajectories = create_offline_dataset(
            benchmark_name=args.benchmark,
            split=args.split,
            num_episodes=args.num_episodes,
            seed=args.seed,
        )
        print("Dataset ready.\n")

    summaries: List[BenchmarkSummary] = []

    for idx, cp in enumerate(checkpoints):
        label = os.path.basename(cp) if cp else f"Baseline #{idx+1}"
        print(f"[{idx+1}/{len(checkpoints)}] Evaluating: {label}...")
        t_start = time.perf_counter()

        summary = run_benchmark_for_checkpoint(
            checkpoint_path=cp,
            benchmark_name=args.benchmark,
            split=args.split,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            device_str=args.device,
            dtype_str=args.dtype,
            tolerance=args.tolerance,
            seed=args.seed,
            offline_trajectories=offline_trajectories,
            model_size=args.model_size,
            use_jepa_planning=args.use_jepa_planning,
            jepa_lambda=args.jepa_lambda,
            raw_mind2web_dir=args.raw_mind2web_dir,
        )
        elapsed = time.perf_counter() - t_start
        print(
            f"  -> Done in {elapsed:.2f}s | SR: {summary.task_success_rate:.1f}% | "
            f"Step SR: {summary.mean_step_success_rate:.3f} | Latency: {summary.mean_decision_latency_ms:.1f}ms\n"
        )
        summaries.append(summary)

    # Render ASCII Summary Table
    print("\n" + "=" * 80)
    ascii_table = TaskEvaluator.format_ascii_table(
        summaries,
        title=f"JED-M2 BENCHMARK: {args.benchmark.upper()} ({args.split.upper()})",
    )
    print(ascii_table)
    print("=" * 80 + "\n")

    # Export Markdown Report
    if args.output_report:
        notes = (
            f"Evaluated on {args.num_episodes} episodes with max_steps={args.max_steps} "
            f"and spatial tolerance epsilon={args.tolerance} on device={args.device}."
        )
        saved_path = TaskEvaluator.export_markdown_report(
            summaries=summaries,
            output_path=args.output_report,
            notes=notes,
        )
        print(f"Benchmark report exported successfully to: {os.path.abspath(args.output_report)}")


if __name__ == "__main__":
    main()
