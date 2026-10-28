"""Site-held-out real-trace evaluation and ablation run bookkeeping."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from jed_m2.heads import ActionType
from jed_m2.real_dataset import (
    RealTrajectoryCollator,
    RealWebTrajectoryDataset,
    RealWebTrajectorySample,
)


MANIFEST_SCHEMA = "jed-m2.real-web-manifest.v1"
DEFAULT_COORDINATE_TOLERANCE = 0.05


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def code_revision(cwd: Optional[Path] = None) -> Dict[str, Optional[str]]:
    """Records the git revision used to launch a run, or None when unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(cwd or Path.cwd()),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"git_sha": sha or None, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": None, "dirty": None}


def load_real_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Real-data manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if recorded != sha256_json(unhashed):
        raise ValueError(f"Real-data manifest checksum is invalid: {path}")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "ready":
        raise ValueError("A ready Mind2Web real-only manifest is required.")
    if manifest.get("source", {}).get("dataset") != "osunlp/Mind2Web":
        raise ValueError("This evaluator accepts only the declared Mind2Web source.")
    if str(manifest.get("source", {}).get("verification", "")).startswith("synthetic"):
        raise ValueError("Synthetic fixture provenance is not accepted for real-data evaluation.")
    audit = manifest.get("audit", {})
    if audit.get("pairwise_website_overlap") != 0 or audit.get("leakage_snapshot_count") != 0:
        raise ValueError("Manifest contains website or snapshot leakage across splits.")
    return manifest


def check_split_integrity(manifest: Dict[str, Any]) -> Dict[str, int]:
    split_sites = {
        split: {
            record["site_key"]
            for record in manifest["tasks"]
            if record.get("split") == split
        }
        for split in ("train", "val", "test")
    }
    overlap = {
        f"{left}_{right}": sorted(split_sites[left] & split_sites[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlap.values()):
        raise ValueError(f"Website leakage found in selected manifest: {overlap}")
    split_snapshot_sets = {
        split: {
            snapshot
            for record in manifest["tasks"]
            if record.get("split") == split
            for snapshot in record.get("snapshot_hashes", [])
        }
        for split in ("train", "val", "test")
    }
    snapshot_overlap = any(
        split_snapshot_sets[left] & split_snapshot_sets[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    if snapshot_overlap:
        raise ValueError("Snapshot leakage found across manifest splits.")
    floors = manifest.get("minimum_sites", {})
    observed = {split: len(split_sites[split]) for split in split_sites}
    unmet = {
        split: {"required": floors.get(split, 0), "available": observed[split]}
        for split in observed
        if observed[split] < int(floors.get(split, 0))
    }
    if unmet:
        raise ValueError(f"Website diversity floor not met: {unmet}")
    assignments = manifest.get("website_assignments", {})
    if assignments and any(
        assignments.get(site) != split
        for split, sites in split_sites.items()
        for site in sites
    ):
        raise ValueError("Task site assignment does not agree with the manifest split index.")
    return observed


def create_run_directory(
    output_root: Path, run_id: str, overwrite: bool = False, resume: bool = False
) -> Path:
    """Creates an isolated ablation directory and refuses collisions unless overwrite or resume is set."""
    if Path(str(output_root)).name.lower() == "ablation_report.md" or str(run_id).lower() == "ablation_report.md":
        raise ValueError("Real-data ablation outputs cannot replace ABLATION_REPORT.md.")
    output_dir = Path(output_root) / run_id
    if output_dir.exists():
        if resume:
            (output_dir / "runs").mkdir(parents=True, exist_ok=True)
            return output_dir
        if not overwrite:
            raise FileExistsError(
                f"Ablation run directory already exists: {output_dir}. "
                "Pass --overwrite to replace it, --resume to continue it, or choose a new --run-id."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "runs").mkdir()
    return output_dir


def _make_window(
    sample: RealWebTrajectorySample,
    turns: Sequence[Any],
    window_number: int,
) -> RealWebTrajectorySample:
    provenance = dict(sample.provenance)
    provenance["parent_annotation_id"] = sample.sample_id
    provenance["window_number"] = window_number
    return RealWebTrajectorySample(
        sample_id=f"{sample.sample_id}::window-{window_number:03d}",
        domain=sample.domain,
        goal=sample.goal,
        turns=list(turns),
        split=sample.split,
        total_reward=sample.total_reward,
        provenance=provenance,
    )


def _has_valid_action_markers(batch: Any, expected: int, collator: RealTrajectoryCollator) -> bool:
    count = int(batch.action_type_mask[0].sum().item())
    if count != expected:
        return False
    input_ids = batch.input_ids[0]
    for action_index in range(count):
        position = int(batch.action_indices[0, action_index].item())
        if position >= input_ids.numel() or int(input_ids[position].item()) != collator.tokenizer.ACTION_TOKEN_ID:
            return False
    return True


def split_into_context_windows(
    samples: Sequence[RealWebTrajectorySample], collator: RealTrajectoryCollator
) -> List[RealWebTrajectorySample]:
    """Partitions each real trajectory into ordered, non-overlapping action windows."""
    windows: List[RealWebTrajectorySample] = []
    total_actions = 0
    for sample in samples:
        current: List[Any] = []
        window_number = 0
        for turn in sample.turns:
            proposal = current + [turn]
            proposed_window = _make_window(sample, proposal, window_number)
            batch = collator([proposed_window])
            if _has_valid_action_markers(batch, len(proposal), collator):
                current.append(turn)
                continue
            if current:
                windows.append(_make_window(sample, current, window_number))
                window_number += 1
                current = [turn]
                single = collator([_make_window(sample, current, window_number)])
                if not _has_valid_action_markers(single, 1, collator):
                    raise ValueError(f"A single source action does not fit the configured context: {sample.sample_id}")
            else:
                raise ValueError(f"A single source action does not fit the configured context: {sample.sample_id}")
        if current:
            windows.append(_make_window(sample, current, window_number))
        total_actions += len(sample.turns)
    if sum(len(window.turns) for window in windows) != total_actions:
        raise RuntimeError("Context-windowing did not preserve each source action exactly once.")
    return windows


def count_window_tokens(
    windows: Sequence[RealWebTrajectorySample], collator: RealTrajectoryCollator
) -> Dict[str, Any]:
    counts: List[int] = []
    action_counts: List[int] = []
    for window in windows:
        batch = collator([window])
        counts.append(int((batch.input_ids != collator.tokenizer.PAD_TOKEN_ID).sum().item()))
        action_counts.append(int(batch.action_type_mask.sum().item()))
    return {
        "windows": len(windows),
        "actions": sum(action_counts),
        "tokens": sum(counts),
        "window_token_counts_sha256": sha256_json(counts),
        "window_action_counts_sha256": sha256_json(action_counts),
    }


def matched_epoch_orders(window_count: int, seeds: Sequence[int], passes: int) -> Dict[int, List[List[int]]]:
    import random

    orders: Dict[int, List[List[int]]] = {}
    for seed in seeds:
        epoch_orders: List[List[int]] = []
        for epoch in range(passes):
            order = list(range(window_count))
            random.Random(seed + 1009 * epoch).shuffle(order)
            epoch_orders.append(order)
        orders[int(seed)] = epoch_orders
    return orders


def _new_site_bucket() -> Dict[str, Any]:
    return {
        "actions": 0,
        "action_correct": 0,
        "unscorable_select": 0,
        "coord_count": 0,
        "coord_abs_sum": 0.0,
        "coord_tolerance_correct": 0,
        "target_count": 0,
        "target_hit": 0,
        "step_count": 0,
        "step_correct": 0,
        "tasks": 0,
        "trajectory_correct": 0,
        "trajectory_unscorable": 0,
    }


def metric_rates(bucket: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "actions": bucket["actions"],
        "action_accuracy": bucket["action_correct"] / bucket["actions"] if bucket["actions"] else None,
        "unscorable_select": bucket["unscorable_select"],
        "coordinate_count": bucket["coord_count"],
        "coordinate_mae": bucket["coord_abs_sum"] / (2 * bucket["coord_count"]) if bucket["coord_count"] else None,
        "coordinate_tolerance_accuracy": (
            bucket["coord_tolerance_correct"] / bucket["coord_count"] if bucket["coord_count"] else None
        ),
        "target_element_count": bucket["target_count"],
        "target_element_hit_rate": bucket["target_hit"] / bucket["target_count"] if bucket["target_count"] else None,
        "step_accuracy": bucket["step_correct"] / bucket["step_count"] if bucket["step_count"] else None,
        "step_count": bucket["step_count"],
        "tasks": bucket["tasks"],
        "exact_trajectory_match_rate": (
            bucket["trajectory_correct"] / bucket["tasks"] if bucket["tasks"] else None
        ),
        "trajectory_unscorable": bucket["trajectory_unscorable"],
    }


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int], start_at: int) -> Optional[int]:
    if not subsequence:
        return None
    final_start = len(sequence) - len(subsequence)
    for start in range(start_at, final_start + 1):
        if list(sequence[start:start + len(subsequence)]) == list(subsequence):
            return start
    return None


def score_offline_windows(
    model: Any,
    windows: Sequence[RealWebTrajectorySample],
    collator: RealTrajectoryCollator,
    device: torch.device,
    dtype: torch.dtype,
    coordinate_tolerance: float = DEFAULT_COORDINATE_TOLERANCE,
) -> Dict[str, Any]:
    """Teacher-forced offline trace replay scored per action and per website."""
    model.eval()
    site_buckets: Dict[str, Dict[str, Any]] = {}
    task_correctness: Dict[str, List[bool]] = {}
    task_sites: Dict[str, str] = {}
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
            bucket = site_buckets.setdefault(site, _new_site_bucket())
            task_correctness.setdefault(parent_id, [])
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
                if type_scoreable:
                    type_correct = predicted_type == target_type
                    bucket["actions"] += 1
                    bucket["action_correct"] += int(type_correct)
                else:
                    type_correct = False
                    if int(turn.action_type) == int(ActionType.SELECT) and type_logits.shape[-1] < 7:
                        bucket["unscorable_select"] += 1

                coordinate_correct = True
                if turn.coordinate_mask:
                    predicted_coords = output.action_output.coordinates[0, action_position].float().detach().cpu()
                    target_x, target_y = turn.coordinates
                    delta_x = abs(float(predicted_coords[0]) - target_x)
                    delta_y = abs(float(predicted_coords[1]) - target_y)
                    bucket["coord_count"] += 1
                    bucket["coord_abs_sum"] += delta_x + delta_y
                    dist = (delta_x ** 2 + delta_y ** 2) ** 0.5
                    predicted_xy = (float(predicted_coords[0]), float(predicted_coords[1]))
                    target_element = next((
                        element for element in (turn.interactive_elements or [])
                        if element.element_id == turn.target_element_id and element.has_bbox
                    ), None)
                    within_tolerance = dist <= coordinate_tolerance
                    if not within_tolerance and target_element is not None:
                        within_tolerance = target_element.contains(*predicted_xy)
                    bucket["coord_tolerance_correct"] += int(within_tolerance)
                    coordinate_correct = within_tolerance

                    if turn.target_element_mask and target_element is not None:
                        bucket["target_count"] += 1
                        bucket["target_hit"] += int(target_element.contains(*predicted_xy))

                argument_correct = True
                if turn.argument and turn.argument_mask:
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

                step_correct = type_correct and coordinate_correct and argument_correct
                if type_scoreable:
                    bucket["step_count"] += 1
                    bucket["step_correct"] += int(step_correct)
                    task_correctness[parent_id].append(bool(step_correct))
                else:
                    task_correctness[parent_id].append(False)

    for task_id, step_results in task_correctness.items():
        site = task_sites[task_id]
        bucket = site_buckets[site]
        bucket["tasks"] += 1
        if step_results and all(step_results):
            bucket["trajectory_correct"] += 1
        elif not step_results:
            bucket["trajectory_unscorable"] += 1

    overall = _new_site_bucket()
    for bucket in site_buckets.values():
        for key in overall:
            overall[key] += bucket[key]
    per_site = {site: metric_rates(bucket) for site, bucket in sorted(site_buckets.items())}
    overall_metrics = metric_rates(overall)
    macro: Dict[str, Any] = {}
    for key in (
        "action_accuracy", "coordinate_mae", "coordinate_tolerance_accuracy",
        "target_element_hit_rate", "step_accuracy", "exact_trajectory_match_rate",
    ):
        values = [metrics[key] for metrics in per_site.values() if metrics.get(key) is not None]
        macro[key] = statistics.mean(values) if values else None
        macro[f"{key}_sites"] = len(values)
    return {
        "protocol": "offline teacher-forced real-trace replay in non-overlapping context windows",
        "overall_micro": overall_metrics,
        "website_macro": macro,
        "by_website": per_site,
    }


def load_real_only_split(
    manifest_path: str | Path,
    split: str,
    raw_mind2web_dir: str | Path,
    cache_dir: str | Path,
    config: Any,
    seed: int = 42,
) -> Tuple[Dict[str, Any], Dict[str, int], RealWebTrajectoryDataset]:
    manifest = load_real_manifest(Path(manifest_path))
    website_counts = check_split_integrity(manifest)
    dataset = RealWebTrajectoryDataset(
        config=config,
        split=split,
        real_only=True,
        manifest_path=str(manifest_path),
        raw_mind2web_dir=str(raw_mind2web_dir),
        cache_dir=str(cache_dir),
        seed=seed,
    )
    return manifest, website_counts, dataset


def evaluate_real_manifest_offline(
    model: Any,
    manifest_path: str | Path,
    split: str,
    raw_mind2web_dir: str | Path,
    cache_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    max_seq_len: int,
    coordinate_tolerance: float = DEFAULT_COORDINATE_TOLERANCE,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates a checkpoint on a validated real-only split; never falls back to synthetic data."""
    from jed_m2.config import JEDM2Config

    config = getattr(model, "config", None) or JEDM2Config(max_seq_len=max_seq_len)
    manifest, website_counts, dataset = load_real_only_split(
        manifest_path=manifest_path,
        split=split,
        raw_mind2web_dir=raw_mind2web_dir,
        cache_dir=cache_dir,
        config=config,
        seed=seed,
    )
    collator = RealTrajectoryCollator(config, max_seq_len=max_seq_len)
    windows = split_into_context_windows(dataset.samples, collator)
    metrics = score_offline_windows(
        model=model,
        windows=windows,
        collator=collator,
        device=device,
        dtype=dtype,
        coordinate_tolerance=coordinate_tolerance,
    )
    metrics["manifest_sha256"] = manifest["manifest_sha256"]
    metrics["website_counts"] = website_counts
    metrics["split"] = split
    metrics["unique_tasks"] = len(dataset)
    metrics["data_audit"] = dataset.real_data_audit
    return metrics


def budgets_match(records: Iterable[Dict[str, Any]]) -> bool:
    keys = ("tokens_seen", "total_tokens", "optimizer_updates_completed", "unique_train_tasks_per_epoch")
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["seed"]), []).append(record)
    for seed_records in grouped.values():
        reference = seed_records[0]
        for record in seed_records[1:]:
            for key in keys:
                if record.get(key) != reference.get(key):
                    return False
    return True
