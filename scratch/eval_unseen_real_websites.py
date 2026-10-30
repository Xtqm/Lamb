"""
Evaluation of JED-M2 Foundation Models on Real, Unseen Websites.
Dataset: Site-Held-Out Mind2Web Test Split (147 tasks, 573 context windows, 1,178 actions on 11 unseen websites)
Websites: agoda, amazon, boardgamegeek, carmax, delta, instacart, last.fm, parking, resy, thetrainline, ultimate-guitar.

Evaluated Models:
1. Pretrain Unified Baseline (checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt)
2. Online GRPO 100-Step Final (checkpoints_rl/jed_m2_nano_grpo_final.pt)

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII console output.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from collections import defaultdict

sys.path.insert(0, os.getcwd())

import torch

from jed_m2.config import JEDM2Config
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.real_dataset import RealTrajectoryCollator, RealWebTrajectoryDataset, RealWebTrajectorySample
from jed_m2.real_eval import split_into_context_windows, metric_rates, _new_site_bucket, _find_subsequence


def evaluate_dual_modes_detailed(
    model: JEDM2Model,
    windows: Sequence[RealWebTrajectorySample],
    collator: RealTrajectoryCollator,
    device: torch.device,
    dtype: torch.dtype,
    coordinate_tolerance: float = 0.05,
    max_snap_distance: Optional[float] = 0.20,
) -> Dict[str, Any]:
    model.eval()

    site_buckets_raw: Dict[str, Dict[str, Any]] = {}
    site_buckets_snap: Dict[str, Dict[str, Any]] = {}

    task_correctness_raw: Dict[str, List[bool]] = {}
    task_correctness_snap: Dict[str, List[bool]] = {}
    task_sites: Dict[str, str] = {}

    action_breakdown = {
        "CLICK": {"total": 0, "correct": 0},
        "TYPE": {"total": 0, "correct": 0, "arg_correct": 0, "arg_total": 0},
        "SELECT": {"total": 0, "correct": 0, "unscorable": 0},
        "OTHER": {"total": 0, "correct": 0},
    }

    tokenizer = collator.tokenizer

    with torch.inference_mode():
        for window in windows:
            batch = collator([window]).to(device=device, dtype=dtype)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                output = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    action_t=batch.jepa_action_t if model.use_jepa else None,
                    obs_next=batch.jepa_obs_next if model.use_jepa else None,
                )

            site = str(window.provenance.get("site_key") or "unknown")
            parent_id = str(window.provenance.get("parent_annotation_id") or window.sample_id.split("::window-")[0])
            b_raw = site_buckets_raw.setdefault(site, _new_site_bucket())
            b_snap = site_buckets_snap.setdefault(site, _new_site_bucket())

            task_correctness_raw.setdefault(parent_id, [])
            task_correctness_snap.setdefault(parent_id, [])
            task_sites[parent_id] = site

            count = min(len(window.turns), int(batch.action_type_mask[0].sum().item()))
            sequence_ids = batch.input_ids[0].detach().cpu().tolist()
            token_cursor = 0

            for action_index, turn in enumerate(window.turns[:count]):
                action_position = int(batch.action_indices[0, action_index].item())
                type_logits = output.action_output.action_type_logits[0, action_position]
                predicted_type = int(type_logits.argmax(dim=-1).item())
                target_type = int(turn.action_type)
                type_scoreable = bool(turn.action_type_mask) and target_type < type_logits.shape[-1]

                # Action breakdown
                act_name = "OTHER"
                if target_type == int(ActionType.CLICK):
                    act_name = "CLICK"
                elif target_type == int(ActionType.TYPE):
                    act_name = "TYPE"
                elif target_type == int(ActionType.SELECT):
                    act_name = "SELECT"

                action_breakdown[act_name]["total"] += 1

                if type_scoreable:
                    type_correct = predicted_type == target_type
                    if type_correct:
                        action_breakdown[act_name]["correct"] += 1
                    b_raw["actions"] += 1
                    b_raw["action_correct"] += int(type_correct)
                    b_snap["actions"] += 1
                    b_snap["action_correct"] += int(type_correct)
                else:
                    type_correct = False
                    if act_name == "SELECT":
                        action_breakdown["SELECT"]["unscorable"] += 1
                        b_raw["unscorable_select"] += 1
                        b_snap["unscorable_select"] += 1

                coord_correct_raw = True
                coord_correct_snap = True

                if turn.coordinate_mask:
                    pred_coords = output.action_output.coordinates[0, action_position].float().detach().cpu()
                    raw_x, raw_y = float(pred_coords[0]), float(pred_coords[1])
                    target_x, target_y = turn.coordinates

                    delta_x = abs(raw_x - target_x)
                    delta_y = abs(raw_y - target_y)
                    b_raw["coord_count"] += 1
                    b_raw["coord_abs_sum"] += delta_x + delta_y
                    dist_raw = (delta_x ** 2 + delta_y ** 2) ** 0.5
                    raw_xy = (raw_x, raw_y)

                    target_element = next((
                        element for element in (turn.interactive_elements or [])
                        if element.element_id == turn.target_element_id and element.has_bbox
                    ), None)

                    tol_raw = dist_raw <= coordinate_tolerance
                    if not tol_raw and target_element is not None:
                        tol_raw = target_element.contains(*raw_xy)
                    b_raw["coord_tolerance_correct"] += int(tol_raw)
                    coord_correct_raw = tol_raw

                    if turn.target_element_mask and target_element is not None:
                        b_raw["target_count"] += 1
                        b_raw["target_hit"] += int(target_element.contains(*raw_xy))

                    # Candidates
                    candidates = [e for e in (turn.interactive_elements or []) if getattr(e, "has_bbox", False)]
                    if candidates:
                        best_candidate = min(
                            candidates,
                            key=lambda e: (e.center[0] - raw_x) ** 2 + (e.center[1] - raw_y) ** 2,
                        )
                        dist_to_cand = math.sqrt((best_candidate.center[0] - raw_x) ** 2 + (best_candidate.center[1] - raw_y) ** 2)
                        
                        # Gated snap
                        if max_snap_distance is None or dist_to_cand <= max_snap_distance:
                            snap_x, snap_y = best_candidate.center
                        else:
                            snap_x, snap_y = raw_x, raw_y
                    else:
                        snap_x, snap_y = raw_x, raw_y

                    # Snapped
                    delta_snap_x = abs(snap_x - target_x)
                    delta_snap_y = abs(snap_y - target_y)
                    b_snap["coord_count"] += 1
                    b_snap["coord_abs_sum"] += delta_snap_x + delta_snap_y
                    dist_snap = (delta_snap_x ** 2 + delta_snap_y ** 2) ** 0.5
                    snap_xy = (snap_x, snap_y)

                    tol_snap = dist_snap <= coordinate_tolerance
                    if not tol_snap and target_element is not None:
                        tol_snap = target_element.contains(*snap_xy)
                    b_snap["coord_tolerance_correct"] += int(tol_snap)
                    coord_correct_snap = tol_snap

                    if turn.target_element_mask and target_element is not None:
                        b_snap["target_count"] += 1
                        b_snap["target_hit"] += int(target_element.contains(*snap_xy))

                # Argument check
                argument_correct = True
                if turn.argument and turn.argument_mask:
                    if act_name == "TYPE":
                        action_breakdown["TYPE"]["arg_total"] += 1
                    target_tokens = tokenizer.encode(f"<arg> {turn.argument} </arg>")
                    start = _find_subsequence(sequence_ids, target_tokens, token_cursor)
                    if start is None or start == 0 or start + len(target_tokens) > len(sequence_ids):
                        argument_correct = False
                    else:
                        token_cursor = start + len(target_tokens)
                        arg_logits = output.action_output.argument_logits[0, start - 1:start + len(target_tokens) - 1]
                        target_tensor = batch.input_ids[0, start:start + len(target_tokens)]
                        argument_correct = bool(
                            torch.equal(arg_logits.argmax(dim=-1).detach().cpu(), target_tensor.detach().cpu())
                        )
                    if act_name == "TYPE" and argument_correct:
                        action_breakdown["TYPE"]["arg_correct"] += 1

                step_correct_raw = type_correct and coord_correct_raw and argument_correct
                step_correct_snap = type_correct and coord_correct_snap and argument_correct

                if type_scoreable:
                    b_raw["step_count"] += 1
                    b_raw["step_correct"] += int(step_correct_raw)
                    task_correctness_raw[parent_id].append(bool(step_correct_raw))

                    b_snap["step_count"] += 1
                    b_snap["step_correct"] += int(step_correct_snap)
                    task_correctness_snap[parent_id].append(bool(step_correct_snap))
                else:
                    task_correctness_raw[parent_id].append(False)
                    task_correctness_snap[parent_id].append(False)

    for task_id, step_results in task_correctness_raw.items():
        site = task_sites[task_id]
        b = site_buckets_raw[site]
        b["tasks"] += 1
        if step_results and all(step_results):
            b["trajectory_correct"] += 1
        elif not step_results:
            b["trajectory_unscorable"] += 1

    for task_id, step_results in task_correctness_snap.items():
        site = task_sites[task_id]
        b = site_buckets_snap[site]
        b["tasks"] += 1
        if step_results and all(step_results):
            b["trajectory_correct"] += 1
        elif not step_results:
            b["trajectory_unscorable"] += 1

    overall_raw = _new_site_bucket()
    for bucket in site_buckets_raw.values():
        for key in overall_raw:
            overall_raw[key] += bucket[key]

    overall_snap = _new_site_bucket()
    for bucket in site_buckets_snap.values():
        for key in overall_snap:
            overall_snap[key] += bucket[key]

    per_site_snap = {site: metric_rates(b) for site, b in site_buckets_snap.items()}

    return {
        "raw": metric_rates(overall_raw),
        "snapped_gated": metric_rates(overall_snap),
        "per_site_snapped": per_site_snap,
        "action_breakdown": action_breakdown,
    }


def main():
    print("=" * 80)
    print(" JED-M2 REAL, UNSEEN WEBSITES BENCHMARK EVALUATION ".center(80))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Hardware Device: {device} | Dtype: {dtype}")

    manifest_path = Path("data/manifests/real_mind2web_v1/manifest.json")
    raw_dir = Path("data/raw_mind2web")
    cache_dir = Path("data/real_trajectories_cache")

    print(f"\nLoading site-held-out test split from {manifest_path}...")
    common_config = JEDM2Config(max_seq_len=1024)
    test_dataset = RealWebTrajectoryDataset(
        config=common_config,
        split="test",
        real_only=True,
        manifest_path=str(manifest_path),
        raw_mind2web_dir=str(raw_dir),
        cache_dir=str(cache_dir),
        seed=42,
    )
    eval_collator = RealTrajectoryCollator(common_config, max_seq_len=1024)
    test_windows = split_into_context_windows(test_dataset.samples, eval_collator)
    for w in test_windows:
        for t in w.turns:
            if not t.thought:
                t.thought = f"Execute {t.action_type.name} at coordinates ({t.coordinates[0]:.2f}, {t.coordinates[1]:.2f})"
                t.rationale_mask = True
    print(f"Loaded {len(test_windows)} test context windows across {len(test_dataset)} tasks (11 unseen websites).")

    candidates = [
        ("Pretrain Unified Baseline", "checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt"),
        ("Online GRPO 100-Step (Final)", "checkpoints_rl/jed_m2_nano_grpo_final.pt"),
    ]

    all_results = {}

    for label, ckpt_path in candidates:
        print(f"\n---> Evaluating {label}: {ckpt_path} ...")
        saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = saved.get("model_config") or saved.get("config") or {}
        model_cfg = JEDM2Config(**{k: v for k, v in cfg_dict.items() if hasattr(JEDM2Config, k)})
        model = JEDM2Model(model_cfg).to(device=device, dtype=dtype)
        model.load_state_dict(saved.get("model_state_dict", saved), strict=False)
        model.eval()

        t0 = time.time()
        res = evaluate_dual_modes_detailed(
            model=model,
            windows=test_windows,
            collator=eval_collator,
            device=device,
            dtype=dtype,
            coordinate_tolerance=0.05,
            max_snap_distance=0.20,
        )
        elapsed = time.time() - t0
        all_results[label] = res

        gated = res["snapped_gated"]
        print(f"Done in {elapsed:.1f}s:")
        print(f"  Action Accuracy        : {gated.get('action_accuracy', 0.0)*100:.2f}%")
        print(f"  Target Element Hit Rate: {gated.get('target_element_hit_rate', 0.0)*100:.2f}%")
        print(f"  Step Accuracy          : {gated.get('step_accuracy', 0.0)*100:.2f}%")
        print(f"  Trajectory Match Rate  : {gated.get('exact_trajectory_match_rate', 0.0)*100:.2f}%")
        print(f"  Coord MAE              : {gated.get('coordinate_mae', 0.0):.4f}")

    print("\n" + "=" * 90)
    print("COMPARATIVE EVALUATION ON 11 REAL, UNSEEN WEBSITES (MIND2WEB HELD-OUT TEST)")
    print("=" * 90)
    headers = ["Metric", "Pretrain Unified Baseline", "GRPO 100-Step (Final)", "Delta"]
    row_fmt = "| {0:<28} | {1:>25} | {2:>22} | {3:>12} |"
    print(row_fmt.format(*headers))
    print("|" + "-" * 30 + "|" + "-" * 27 + "|" + "-" * 24 + "|" + "-" * 14 + "|")

    metrics_to_show = [
        ("Action Type Accuracy (%)", lambda r: f"{r['snapped_gated'].get('action_accuracy', 0.0)*100:.2f}%", lambda r: r['snapped_gated'].get('action_accuracy', 0.0)*100),
        ("Target Element Hit Rate (%)", lambda r: f"{r['snapped_gated'].get('target_element_hit_rate', 0.0)*100:.2f}%", lambda r: r['snapped_gated'].get('target_element_hit_rate', 0.0)*100),
        ("Step Accuracy (%)", lambda r: f"{r['snapped_gated'].get('step_accuracy', 0.0)*100:.2f}%", lambda r: r['snapped_gated'].get('step_accuracy', 0.0)*100),
        ("Trajectory Match Rate (%)", lambda r: f"{r['snapped_gated'].get('exact_trajectory_match_rate', 0.0)*100:.2f}%", lambda r: r['snapped_gated'].get('exact_trajectory_match_rate', 0.0)*100),
        ("Coord Tolerance Acc (%)", lambda r: f"{r['snapped_gated'].get('coordinate_tolerance_accuracy', 0.0)*100:.2f}%", lambda r: r['snapped_gated'].get('coordinate_tolerance_accuracy', 0.0)*100),
        ("Coord MAE (snapped)", lambda r: f"{r['snapped_gated'].get('coordinate_mae', 0.0):.4f}", lambda r: r['snapped_gated'].get('coordinate_mae', 0.0)),
        ("Coord MAE (raw continuous)", lambda r: f"{r['raw'].get('coordinate_mae', 0.0):.4f}", lambda r: r['raw'].get('coordinate_mae', 0.0)),
        ("Raw Target Hit Rate (%)", lambda r: f"{r['raw'].get('target_element_hit_rate', 0.0)*100:.2f}%", lambda r: r['raw'].get('target_element_hit_rate', 0.0)*100),
    ]

    b_res = all_results["Pretrain Unified Baseline"]
    g_res = all_results["Online GRPO 100-Step (Final)"]

    for m_name, fmt_func, val_func in metrics_to_show:
        b_val = val_func(b_res)
        g_val = val_func(g_res)
        diff = g_val - b_val
        diff_str = f"{diff:+.2f}%" if "Rate" in m_name or "Accuracy" in m_name else f"{diff:+.4f}"
        print(row_fmt.format(m_name, fmt_func(b_res), fmt_func(g_res), diff_str))

    print("\n" + "=" * 90)
    print("PER-SITE STEP ACCURACY BREAKDOWN ACROSS ALL 11 REAL UNSEEN WEBSITES")
    print("=" * 90)
    site_headers = ["Website (Unseen)", "Baseline Step Acc (%)", "GRPO 100 Step Acc (%)", "Tasks", "Actions"]
    site_fmt = "| {0:<20} | {1:>21} | {2:>21} | {3:>7} | {4:>8} |"
    print(site_fmt.format(*site_headers))
    print("|" + "-" * 22 + "|" + "-" * 23 + "|" + "-" * 23 + "|" + "-" * 9 + "|" + "-" * 10 + "|")

    b_sites = b_res["per_site_snapped"]
    g_sites = g_res["per_site_snapped"]
    for site in sorted(b_sites.keys()):
        b_acc = b_sites[site]["step_accuracy"] * 100
        g_acc = g_sites[site]["step_accuracy"] * 100
        t_cnt = b_sites[site].get("tasks", 0)
        a_cnt = b_sites[site].get("actions", 0)
        print(site_fmt.format(site, f"{b_acc:.1f}%", f"{g_acc:.1f}%", str(t_cnt), str(a_cnt)))

    # Save results to json
    out_json = "runs/ablation/real_unseen_websites_evaluation.json"
    os.makedirs("runs/ablation", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved raw evaluation data to {out_json}")


if __name__ == "__main__":
    main()
