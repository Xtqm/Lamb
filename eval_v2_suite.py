"""
JED-M2 v2.0 Empirical Validation & Ablation Suite.
Evaluates JED-M2 v2.0 against JED-M2 v1.0 and Small-LM baselines on the 11 held-out Mind2Web websites.
Supports the 4-way ablation matrix:
  (a) SFT-only baseline
  (b) SFT + Step-shaped GRPO (JED-M2 v1.0)
  (c) SFT + Outcome-supervised GRPO (JED-M2 v2.0 Policy)
  (d) SFT + Outcome GRPO + JEPA Best-of-N Test-Time Search (JED-M2 v2.0 Full)
plus Small-LM baselines (T5-Small, DistilGPT-2).

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import torch

from jed_m2.agent import JEDM2Agent
from jed_m2.benchmark import (
    BenchmarkSummary,
    TaskEvaluator,
    evaluate_offline_dataset,
)
from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.config import JEDM2Config
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model
from jed_m2.real_dataset import RealWebTrajectoryDataset
from jed_m2.small_lm_baselines import (
    DistilGPT2ActionAgent,
    SmallLMAgentWrapper,
    T5SmallActionAgent,
)
from eval_benchmarks import (
    detect_tokenizer_for_checkpoint,
    is_distilgpt2_baseline,
    is_t5_baseline,
    load_model,
)


def select_stratified_test_episodes(
    manifest_path: str = "data/manifests/real_mind2web_v1/manifest.json",
    raw_mind2web_dir: str = "data/raw_mind2web",
    episodes_per_site: int = 1,
    seed: int = 42,
) -> List[Any]:
    """
    Selects balanced evaluation trajectory episodes covering all 11 held-out test websites.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    task_map = {t["annotation_id"]: t["website"] for t in manifest.get("tasks", [])}

    config = JEDM2Config()
    dataset = RealWebTrajectoryDataset(
        config=config,
        split="test",
        dataset_size=147,
        raw_mind2web_dir=raw_mind2web_dir,
        manifest_path=manifest_path,
        real_only=True,
        seed=seed,
    )

    by_site: Dict[str, List[Any]] = {}
    for sample in dataset.samples:
        site = task_map.get(sample.sample_id, getattr(sample, "website", "unknown"))
        sample.website = site
        sample.domain = site
        by_site.setdefault(site, []).append(sample)

    selected: List[Any] = []
    for site in sorted(by_site.keys()):
        site_samples = by_site[site]
        selected.extend(site_samples[:episodes_per_site])

    return selected

    return selected


def run_evaluation(
    episodes_per_site: int = 1,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_report: str = "BENCHMARK_REPORT_V2.md",
) -> List[BenchmarkSummary]:
    device = torch.device(device_str)
    dtype = torch.bfloat16 if "cuda" in device_str and torch.cuda.is_available() else torch.float32

    print("=" * 80)
    print(" JED-M2 v2.0 Standardized Empirical Validation & Ablation Suite".center(80))
    print("=" * 80)
    print(f"Device              : {device} ({dtype})")
    print(f"Held-out Websites   : 11 sites (agoda, amazon, carmax, delta, etc.)")
    print(f"Episodes Per Site   : {episodes_per_site}")
    print(f"Output Report Path  : {output_report}")
    print("=" * 80 + "\n")

    print("[1/4] Selecting stratified held-out test episodes across 11 websites...")
    test_episodes = select_stratified_test_episodes(
        manifest_path="data/manifests/real_mind2web_v1/manifest.json",
        raw_mind2web_dir="data/raw_mind2web",
        episodes_per_site=episodes_per_site,
        seed=42,
    )
    print(f"  -> Total test trajectories: {len(test_episodes)} across {len(set(getattr(e, 'website', '') for e in test_episodes))} websites\n")

    evaluator = TaskEvaluator(tolerance_radius=0.05)

    # Matrix configurations:
    # 1. SFT Base
    # 2. Step GRPO (v1.0)
    # 3. Outcome GRPO (v2.0 Policy)
    # 4. Outcome GRPO + JEPA Best-of-N (v2.0 Full)
    # 5. T5-Small baseline
    # 6. DistilGPT-2 baseline
    eval_configs = [
        {
            "name": "JED-M2 (SFT Base)",
            "checkpoint": "checkpoints/pretrain/jed_m2_nano_pretrain_final.pt",
            "type": "jed_m2",
            "use_jepa": False,
        },
        {
            "name": "JED-M2 v1.0 (Step GRPO)",
            "checkpoint": "checkpoints/grpo_run2/jed_m2_grpo_step_40.pt",
            "type": "jed_m2",
            "use_jepa": False,
        },
        {
            "name": "JED-M2 v2.0 (Outcome GRPO)",
            "checkpoint": "checkpoints/grpo_outcome_v2/jed_m2_grpo_step_30.pt",
            "type": "jed_m2",
            "use_jepa": False,
        },
        {
            "name": "JED-M2 v2.0 Full (+JEPA Search)",
            "checkpoint": "checkpoints/grpo_outcome_v2/jed_m2_grpo_step_30.pt",
            "type": "jed_m2",
            "use_jepa": True,
        },
        {
            "name": "T5-Small (~60.5M)",
            "checkpoint": "t5_small",
            "type": "t5",
            "use_jepa": False,
        },
        {
            "name": "DistilGPT-2 (~81.9M)",
            "checkpoint": "distilgpt2",
            "type": "distilgpt2",
            "use_jepa": False,
        },
    ]

    summaries: List[BenchmarkSummary] = []

    for cfg_idx, cfg in enumerate(eval_configs):
        print(f"[{cfg_idx + 1}/{len(eval_configs)}] Evaluating Configuration: {cfg['name']}...")
        t0 = time.time()

        if cfg["type"] == "t5":
            model, _ = load_model("t5_small", device=device, dtype=dtype)
            agent = SmallLMAgentWrapper(model, device=device, dtype=dtype)
        elif cfg["type"] == "distilgpt2":
            model, _ = load_model("distilgpt2", device=device, dtype=dtype)
            agent = SmallLMAgentWrapper(model, device=device, dtype=dtype)
        else:
            model, _ = load_model(cfg["checkpoint"], device=device, dtype=dtype, model_size="nano")
            tok = detect_tokenizer_for_checkpoint(cfg["checkpoint"], model.config)
            agent = JEDM2Agent(
                model=model,
                config=model.config,
                tokenizer=tok,
                device=device,
                dtype=dtype,
                use_jepa_planning=cfg["use_jepa"],
                jepa_planning_lambda=1.0,
            )

        summary = evaluate_offline_dataset(
            agent=agent,
            trajectories=test_episodes,
            evaluator=evaluator,
            benchmark_name="mind2web_heldout_11sites",
            checkpoint_name=cfg["name"],
            max_episodes=len(test_episodes),
            max_steps=15,
            split_name="test",
        )
        elapsed = time.time() - t0
        print(f"  -> Finished in {elapsed:.2f}s | Step SR: {summary.mean_step_success_rate:.3f} | Latency: {summary.mean_decision_latency_ms:.1f}ms\n")
        summaries.append(summary)

        del agent, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Render Comparative ASCII Table
    print("\n" + "=" * 80)
    ascii_table = TaskEvaluator.format_ascii_table(
        summaries,
        title="JED-M2 v2.0 EMPIRICAL VALIDATION & ABLATION MATRIX (11 HELD-OUT WEBSITES)",
    )
    print(ascii_table)
    print("=" * 80 + "\n")

    # Generate Markdown Report
    notes = (
        f"Evaluated across 11 held-out Mind2Web websites with spatial tolerance radius epsilon=0.05 on {device_str}.\n"
        "Compares SFT Base, Step-shaped GRPO (v1.0), Outcome-supervised GRPO (v2.0 Policy), "
        "and Outcome GRPO + JEPA Best-of-N Test-Time Search (v2.0 Full) alongside T5-Small and DistilGPT-2."
    )
    TaskEvaluator.export_markdown_report(
        summaries=summaries,
        output_path=output_report,
        notes=notes,
    )
    print(f"[Done] Formal benchmark report saved to: {output_report}")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes_per_site", type=int, default=1, help="Episodes to evaluate per website (11 sites total)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_report", type=str, default="BENCHMARK_REPORT_V2.md")
    args = parser.parse_args()

    run_evaluation(
        episodes_per_site=args.episodes_per_site,
        device_str=args.device,
        output_report=args.output_report,
    )
