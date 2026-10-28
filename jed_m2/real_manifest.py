"""Inventory and leakage-resistant splitting for authentic web trajectory corpora."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit


DEFAULT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
MANIFEST_SCHEMA = "jed-m2.real-web-manifest.v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return _sha256_bytes(encoded)


def normalize_website(website: Any, fallback_url: Optional[str] = None) -> Optional[str]:
    """Returns a stable website key without confusing category labels with sites."""
    raw = str(website or "").strip()
    if not raw and fallback_url:
        raw = str(fallback_url).strip()
    if not raw:
        return None
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    key = (parsed.hostname or raw).strip().casefold().rstrip(".")
    if key.startswith("www."):
        key = key[4:]
    return key or None


def _bbox_from_candidate(candidate: Any) -> Optional[List[float]]:
    if not isinstance(candidate, dict):
        return None
    attrs = candidate.get("attributes") or {}
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except (TypeError, json.JSONDecodeError):
            return None
    if not isinstance(attrs, dict):
        return None
    bbox = attrs.get("bounding_box_rect")
    if isinstance(bbox, dict):
        bbox = [bbox.get("x"), bbox.get("y"), bbox.get("width"), bbox.get("height")]
    elif isinstance(bbox, str):
        bbox = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", bbox)[:4]
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        coords = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coords):
        return None
    if coords[2] <= 0 or coords[3] <= 0:
        return None
    return coords


def _largest_remainder(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw = {split: total * ratios[split] for split in ratios}
    counts = {split: math.floor(value) for split, value in raw.items()}
    remaining = total - sum(counts.values())
    for split in sorted(ratios, key=lambda name: (-(raw[name] - counts[name]), name))[:remaining]:
        counts[split] += 1
    return counts


def _stable_tie(seed: int, category: str, site: str) -> str:
    return _sha256_bytes(f"{seed}:{category}:{site}".encode("utf-8"))


def _load_source_manifest(raw_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    manifest_path = raw_dir / "source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run the real Mind2Web acquisition workflow first; fixtures are not accepted."
        )
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset") != "osunlp/Mind2Web" or source_manifest.get("upstream_split") != "train":
        raise ValueError("The real-data manifest must identify the official Mind2Web training split.")
    if str(source_manifest.get("verification", "")).startswith("synthetic"):
        raise ValueError("Synthetic fixture provenance cannot be used for a real-only manifest.")
    files = source_manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("The source manifest contains no verified corpus files.")

    for record in files:
        relative = Path(str(record.get("path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe source path in manifest: {relative}")
        path = raw_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Mind2Web source file is missing: {path}")
        if path.stat().st_size != int(record.get("size_bytes", -1)):
            raise ValueError(f"Mind2Web source file size changed after provenance capture: {path}")
        if _sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Mind2Web source file checksum changed after provenance capture: {path}")
    return source_manifest, files, _sha256_file(manifest_path)


def _scan_corpus(raw_dir: Path, source_files: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    duplicate_ids: List[str] = []
    seen_ids: set[str] = set()
    site_counts: Dict[str, Counter] = defaultdict(Counter)
    site_category_counts: Dict[str, Dict[str, Counter]] = defaultdict(lambda: {"domain": Counter(), "subdomain": Counter()})
    global_audit: Dict[str, Any] = {
        "source_records": 0,
        "unique_tasks": 0,
        "duplicate_task_ids": 0,
        "source_actions": 0,
        "usable_actions": 0,
        "operations": Counter(),
        "missing_observation_actions": 0,
        "rationale_actions": 0,
        "positive_candidate_actions": 0,
        "original_target_actions": 0,
        "valid_target_bbox_actions": 0,
        "invalid_records": 0,
        "snapshot_hashes": 0,
    }
    snapshot_to_sites: Dict[str, set[str]] = defaultdict(set)

    for source_file in source_files:
        relative_path = str(source_file["path"])
        path = raw_dir / relative_path
        with path.open("r", encoding="utf-8") as handle:
            chunk = json.load(handle)
        rows = chunk if isinstance(chunk, list) else [chunk]
        for record_index, row in enumerate(rows):
            global_audit["source_records"] += 1
            if not isinstance(row, dict):
                global_audit["invalid_records"] += 1
                continue
            annotation_id = str(row.get("annotation_id") or "").strip()
            website = row.get("website")
            actions = row.get("actions")
            if not annotation_id or not isinstance(actions, list) or not actions:
                global_audit["invalid_records"] += 1
                continue
            if annotation_id in seen_ids:
                duplicate_ids.append(annotation_id)
                global_audit["duplicate_task_ids"] += 1
                continue
            seen_ids.add(annotation_id)

            first_url = next((action.get("url") for action in actions if isinstance(action, dict) and action.get("url")), None)
            site_key = normalize_website(website, first_url)
            if site_key is None:
                global_audit["invalid_records"] += 1
                continue

            category = str(row.get("domain") or "unknown").strip().casefold()
            subcategory = str(row.get("subdomain") or "unknown").strip().casefold()
            task_snapshot_hashes: List[str] = []
            action_ops: Counter = Counter()
            action_coverage = Counter()
            usable_actions = 0
            for action in actions:
                if not isinstance(action, dict):
                    continue
                global_audit["source_actions"] += 1
                operation = action.get("operation") or {}
                op_name = str(operation.get("op", "unknown") if isinstance(operation, dict) else operation).upper()
                action_ops[op_name] += 1
                global_audit["operations"][op_name] += 1

                raw_html = action.get("raw_html")
                cleaned_html = action.get("cleaned_html")
                observation = cleaned_html or raw_html
                if not observation:
                    global_audit["missing_observation_actions"] += 1
                else:
                    usable_actions += 1
                    global_audit["usable_actions"] += 1
                    snapshot_hash = _sha256_bytes(str(observation).encode("utf-8"))
                    task_snapshot_hashes.append(snapshot_hash)
                    snapshot_to_sites[snapshot_hash].add(site_key)

                if action.get("thought"):
                    global_audit["rationale_actions"] += 1
                    action_coverage["rationale"] += 1
                candidates = action.get("pos_candidates") or []
                if candidates:
                    global_audit["positive_candidate_actions"] += 1
                    action_coverage["positive_candidates"] += 1
                if any(isinstance(candidate, dict) and candidate.get("is_original_target") for candidate in candidates):
                    global_audit["original_target_actions"] += 1
                    action_coverage["original_target"] += 1
                if any(_bbox_from_candidate(candidate) is not None for candidate in candidates):
                    global_audit["valid_target_bbox_actions"] += 1
                    action_coverage["candidate_bbox"] += 1

            if usable_actions == 0:
                global_audit["invalid_records"] += 1
                continue
            task_snapshot_hashes = sorted(set(task_snapshot_hashes))
            global_audit["unique_tasks"] += 1
            global_audit["snapshot_hashes"] += len(task_snapshot_hashes)
            site_counts[site_key]["tasks"] += 1
            site_counts[site_key]["actions"] += usable_actions
            site_category_counts[site_key]["domain"][category] += 1
            site_category_counts[site_key]["subdomain"][subcategory] += 1
            records.append({
                "annotation_id": annotation_id,
                "source_file": relative_path,
                "record_index": record_index,
                "website": str(website or site_key),
                "site_key": site_key,
                "domain": category,
                "subdomain": subcategory,
                "upstream_split": "train",
                "action_count": len(actions),
                "usable_action_count": usable_actions,
                "operation_counts": dict(sorted(action_ops.items())),
                "label_coverage": dict(action_coverage),
                "snapshot_hashes": task_snapshot_hashes,
            })
        del chunk, rows

    site_records: Dict[str, Dict[str, Any]] = {}
    for site_key, counts in site_counts.items():
        categories = site_category_counts[site_key]
        primary_domain = min(categories["domain"], key=lambda value: (-categories["domain"][value], value))
        primary_subdomain = min(categories["subdomain"], key=lambda value: (-categories["subdomain"][value], value))
        site_records[site_key] = {
            "tasks": counts["tasks"],
            "actions": counts["actions"],
            "domain": primary_domain,
            "subdomain": primary_subdomain,
            "domain_categories": dict(sorted(categories["domain"].items())),
            "subdomain_categories": dict(sorted(categories["subdomain"].items())),
        }
    global_audit["operations"] = dict(sorted(global_audit["operations"].items()))
    global_audit["actions_without_rationale"] = max(
        0, global_audit["source_actions"] - global_audit["rationale_actions"]
    )
    global_audit["actions_without_positive_candidates"] = max(
        0, global_audit["source_actions"] - global_audit["positive_candidate_actions"]
    )
    global_audit["actions_without_original_target"] = max(
        0, global_audit["source_actions"] - global_audit["original_target_actions"]
    )
    global_audit["actions_without_candidate_geometry"] = max(
        0, global_audit["source_actions"] - global_audit["valid_target_bbox_actions"]
    )
    global_audit["duplicate_annotation_ids"] = sorted(duplicate_ids)
    global_audit["website_count"] = len(site_records)
    global_audit["category_counts"] = {
        "domain": dict(sorted(Counter(item["domain"] for item in records).items())),
        "subdomain": dict(sorted(Counter(item["subdomain"] for item in records).items())),
    }
    global_audit["cross_site_duplicate_snapshots"] = sum(1 for sites in snapshot_to_sites.values() if len(sites) > 1)
    return records, {"summary": global_audit, "sites": site_records, "snapshot_to_sites": snapshot_to_sites}


def _assign_sites(
    sites: Dict[str, Dict[str, Any]], seed: int, ratios: Dict[str, float]
) -> Dict[str, str]:
    assignment: Dict[str, str] = {}
    by_domain: Dict[str, List[str]] = defaultdict(list)
    for site_key, metadata in sites.items():
        by_domain[str(metadata["domain"])].append(site_key)

    for domain in sorted(by_domain):
        domain_sites = by_domain[domain]
        quotas = _largest_remainder(len(domain_sites), ratios)
        ordered = sorted(domain_sites, key=lambda site: (-sites[site]["tasks"], _stable_tie(seed, domain, site)))
        assigned_sites = Counter()
        assigned_tasks = Counter()
        total_tasks = sum(sites[site]["tasks"] for site in domain_sites)
        task_targets = {split: max(1.0, total_tasks * ratios[split]) for split in ratios}
        for site in ordered:
            candidates = [split for split in ratios if assigned_sites[split] < quotas[split]]
            chosen = min(
                candidates,
                key=lambda split: (
                    (assigned_tasks[split] + sites[site]["tasks"]) / task_targets[split],
                    assigned_sites[split] / max(1, quotas[split]),
                    _stable_tie(seed, f"{domain}:{site}", split),
                ),
            )
            assignment[site] = chosen
            assigned_sites[chosen] += 1
            assigned_tasks[chosen] += sites[site]["tasks"]
    return assignment


def _audit_markdown(manifest: Dict[str, Any]) -> str:
    audit = manifest["audit"]
    lines = [
        "# Mind2Web Real-Data Split Audit",
        "",
        f"- Status: **{manifest['status']}**",
        f"- Source: `{manifest['source']['dataset']}` ({manifest['source']['upstream_split']})",
        f"- Revision: `{manifest['source']['revision']}`",
        f"- License: `{manifest['source']['license']}`",
        f"- Verification: `{manifest['source']['verification']}`",
        f"- Seed: `{manifest['seed']}`",
        f"- Manifest SHA-256: `{manifest['manifest_sha256']}`",
        "",
        "## Corpus Inventory",
        "",
        f"- Unique tasks: {audit['unique_tasks']}",
        f"- Websites: {audit['website_count']}",
        f"- Recorded actions: {audit['source_actions']}",
        f"- Actions with captured observations: {audit['usable_actions']}",
        f"- Unique observation snapshots: {audit['snapshot_hashes']}",
        f"- Duplicate task IDs excluded: {audit['duplicate_task_ids']}",
        f"- Cross-site duplicate snapshots: {audit['cross_site_duplicate_snapshots']}",
        "",
        "## Split Summary",
        "",
        "| Split | Tasks | Websites | Usable actions | Source actions | Missing observations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        detail = audit["splits"][split]
        lines.append(
            f"| {split} | {detail['tasks']} | {detail['websites']} | {detail['actions']} "
            f"| {detail['source_actions']} | {detail['actions_without_observation']} |"
        )
    lines.extend([
        "",
        "## Per-Split Annotation Coverage",
        "",
        "| Split | Rationale | Positive candidates | Original target | Candidate geometry |",
        "|---|---:|---:|---:|---:|",
    ])
    for split in ("train", "val", "test"):
        coverage = audit["splits"][split]["label_coverage"]
        lines.append(
            f"| {split} | {coverage['rationale']} | {coverage['positive_candidates']} "
            f"| {coverage['original_target']} | {coverage['candidate_bbox']} |"
        )
    lines.extend(["", "## Categories", ""])
    for category, values in audit["category_counts"].items():
        lines.append(f"- {category}: " + ", ".join(f"{name}={count}" for name, count in values.items()))
    lines.extend(["", "## Source Operations", ""])
    lines.extend(f"- {name}: {count}" for name, count in audit["operations"].items())
    lines.extend([
        "",
        "## Annotation Coverage",
        "",
        f"- Actions with recorded rationale: {audit['rationale_actions']}",
        f"- Actions without recorded rationale: {audit['actions_without_rationale']}",
        f"- Actions with positive candidates: {audit['positive_candidate_actions']}",
        f"- Actions without positive candidates: {audit['actions_without_positive_candidates']}",
        f"- Actions with original target: {audit['original_target_actions']}",
        f"- Actions without original target: {audit['actions_without_original_target']}",
        f"- Actions with candidate geometry: {audit['valid_target_bbox_actions']}",
        f"- Actions without candidate geometry: {audit['actions_without_candidate_geometry']}",
        f"- Records excluded as invalid: {audit['invalid_records']}",
        "",
        "This report contains aggregate metadata only; it does not include page HTML or task text.",
        "",
    ])
    return "\n".join(lines)


def build_real_manifest(
    raw_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    ratios: Optional[Dict[str, float]] = None,
    minimum_sites: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if any((output_path / name).exists() for name in ("manifest.json", "audit.json", "audit.md")):
        raise FileExistsError(f"Manifest output already exists in {output_path}; choose a new run directory.")

    ratios = ratios or dict(DEFAULT_RATIOS)
    minimum_sites = minimum_sites or {"train": 50, "val": 10, "test": 10}
    if set(ratios) != {"train", "val", "test"} or any(value <= 0 for value in ratios.values()):
        raise ValueError("Ratios must define positive train, val, and test fractions.")
    ratio_sum = sum(ratios.values())
    ratios = {key: value / ratio_sum for key, value in ratios.items()}

    source, source_files, source_manifest_sha = _load_source_manifest(raw_path)
    tasks, scan = _scan_corpus(raw_path, source_files)
    site_assignment = _assign_sites(scan["sites"], seed, ratios)
    for task in tasks:
        task["split"] = site_assignment[task["site_key"]]

    split_tasks: Dict[str, List[Dict[str, Any]]] = {
        split: [task for task in tasks if task["split"] == split]
        for split in ("train", "val", "test")
    }
    split_snapshot_sets: Dict[str, set[str]] = {
        split: {snapshot for task in items for snapshot in task["snapshot_hashes"]}
        for split, items in split_tasks.items()
    }
    leakage_pairs: List[Dict[str, Any]] = []
    for left_index, left in enumerate(("train", "val", "test")):
        for right in ("train", "val", "test")[left_index + 1:]:
            overlap = split_snapshot_sets[left] & split_snapshot_sets[right]
            for snapshot in sorted(overlap):
                leakage_pairs.append({"snapshot_sha256": snapshot, "splits": [left, right]})

    split_audit: Dict[str, Dict[str, Any]] = {}
    for split, items in split_tasks.items():
        coverage_counts = Counter(
            label
            for task in items
            for label, count in task.get("label_coverage", {}).items()
            for _ in range(int(count))
        )
        split_audit[split] = {
            "tasks": len(items),
            "websites": len({task["site_key"] for task in items}),
            "actions": sum(task["usable_action_count"] for task in items),
            "source_actions": sum(task["action_count"] for task in items),
            "actions_without_observation": sum(
                max(0, task["action_count"] - task["usable_action_count"]) for task in items
            ),
            "label_coverage": {
                "rationale": coverage_counts["rationale"],
                "positive_candidates": coverage_counts["positive_candidates"],
                "original_target": coverage_counts["original_target"],
                "candidate_bbox": coverage_counts["candidate_bbox"],
            },
            "operations": dict(sorted(Counter(
                operation
                for task in items
                for operation, count in task["operation_counts"].items()
                for _ in range(count)
            ).items())),
            "domain_categories": dict(sorted(Counter(task["domain"] for task in items).items())),
            "subdomain_categories": dict(sorted(Counter(task["subdomain"] for task in items).items())),
        }

    split_domain_sets = {split: {task["site_key"] for task in items} for split, items in split_tasks.items()}
    pairwise_site_overlap = sum(
        len(split_domain_sets[left] & split_domain_sets[right])
        for left_index, left in enumerate(("train", "val", "test"))
        for right in ("train", "val", "test")[left_index + 1:]
    )
    diversity_failures = {
        split: {"required": minimum_sites.get(split, 0), "available": split_audit[split]["websites"]}
        for split in split_audit
        if split_audit[split]["websites"] < minimum_sites.get(split, 0)
    }
    summary = scan["summary"]
    category_counts = {
        "domain": summary["category_counts"]["domain"],
        "subdomain": summary["category_counts"]["subdomain"],
    }
    audit = {
        **summary,
        "category_counts": category_counts,
        "splits": split_audit,
        "leakage_snapshot_count": len(leakage_pairs),
        "leakage_snapshot_examples": leakage_pairs[:50],
        "pairwise_website_overlap": pairwise_site_overlap,
        "diversity_failures": diversity_failures,
    }

    payload: Dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "ready" if not leakage_pairs and not diversity_failures and pairwise_site_overlap == 0 else "invalid",
        "source": {
            "dataset": source["dataset"],
            "source_url": source["source_url"],
            "revision": source["revision"],
            "license": source["license"],
            "upstream_split": source["upstream_split"],
            "verification": source["verification"],
            "source_manifest_sha256": source_manifest_sha,
            "files": source_files,
        },
        "seed": seed,
        "ratios": ratios,
        "minimum_sites": minimum_sites,
        "website_assignments": dict(sorted(site_assignment.items())),
        "splits": {split: [task["annotation_id"] for task in split_tasks[split]] for split in split_tasks},
        "tasks": tasks,
        "audit": audit,
    }
    payload["manifest_sha256"] = _canonical_json_sha256(payload)
    manifest_path = output_path / "manifest.json"
    audit_json_path = output_path / "audit.json"
    audit_md_path = output_path / "audit.md"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_json_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_md_path.write_text(_audit_markdown(payload), encoding="utf-8")
    if payload["status"] != "ready":
        raise ValueError(
            f"Real-data manifest is invalid; see {audit_md_path} for leakage or diversity failures."
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Mind2Web training traces and create a site-held-out manifest.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw_mind2web"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--min_train_sites", type=int, default=50)
    parser.add_argument("--min_val_sites", type=int, default=10)
    parser.add_argument("--min_test_sites", type=int, default=10)
    args = parser.parse_args()
    result = build_real_manifest(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        ratios={"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        minimum_sites={"train": args.min_train_sites, "val": args.min_val_sites, "test": args.min_test_sites},
    )
    print(json.dumps({
        "status": result["status"],
        "manifest_sha256": result["manifest_sha256"],
        "splits": result["audit"]["splits"],
        "source_verification": result["source"]["verification"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
