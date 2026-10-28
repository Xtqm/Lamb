"""
Mind2Web Automated Streaming Downloader & Raw Corpus Ingestion Utility.
Streams and unpacks official Mind2Web (osunlp/Mind2Web) raw task annotations
and HTML DOM snapshots across 137 real-world web domains.

Features:
- Direct HTTP streaming from Hugging Face Hub dataset repository.
- Extracts task goals, action traces, positive/negative DOM candidates, and bounding boxes.
- Autonomous offline sample generation mode for test execution and offline sandboxes.
- Strict gitignore compliance: Saves raw data into data/raw_mind2web/ (ignored by git).

STRICT BUILD RULE: Pure Python / PyTorch without external unvetted dependencies.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error
import urllib.parse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mind2web_downloader")

HF_DATASET_ID = "osunlp/Mind2Web"
HF_REVISION = "6314166657eec4aa0e22c00f8d801e609ce8e80f"
HF_API_TREE_URL = "https://huggingface.co/api/datasets/osunlp/Mind2Web/tree"
HF_RESOLVE_URL = "https://huggingface.co/datasets/osunlp/Mind2Web/resolve"
HF_LICENSE = "CC-BY-4.0"
OFFICIAL_TRAIN0_SHA256 = "c8b622901057bca813a6d171733c41e4fc266c2902a23d63b9094c0add3f8f2c"

SAMPLE_REAL_DOMAINS = [
    "amazon.com", "delta.com", "wikipedia.org", "ebay.com", "homedepot.com",
    "target.com", "booking.com", "expedia.com", "walmart.com", "bestbuy.com",
    "github.com", "reddit.com", "yelp.com", "craigslist.org", "airbnb.com"
]

SAMPLE_REAL_TASKS = [
    {
        "website": "amazon.com",
        "domain": "Shopping",
        "task": "Find stainless steel water bottles and filter by Prime shipping",
        "turns": [
            {
                "url": "https://www.amazon.com",
                "thought": "Locate top search input bar and enter search query",
                "op": "TYPE",
                "value": "stainless steel water bottle",
                "target_tag": "input",
                "target_text": "Search Amazon",
                "bbox": [215, 12, 680, 48],
                "html": "<header><form><input id='twotabsearchtextbox' name='field-keywords' placeholder='Search Amazon' bbox='[215,12,680,48]'/><input type='submit' id='nav-search-submit-button' value='Go' bbox='[685,12,740,48]'/></form></header><main><div>Prime Early Access Deals</div></main>"
            },
            {
                "url": "https://www.amazon.com/s?k=stainless+steel+water+bottle",
                "thought": "Filter by Prime eligible products on the left sidebar",
                "op": "CLICK",
                "value": "",
                "target_tag": "span",
                "target_text": "Prime",
                "bbox": [18, 220, 140, 252],
                "html": "<aside><div id='filters'><label><input type='checkbox' id='prime-filter' bbox='[18,220,40,242]'/><span bbox='[44,222,140,252]'>Prime</span></label></div></aside><section><div class='s-result-item' bbox='[250,150,750,400]'>HydroFlask 32oz Wide Mouth</div></section>"
            },
            {
                "url": "https://www.amazon.com/s?k=stainless+steel+water+bottle&ref=prime",
                "thought": "Prime filter applied, goal achieved",
                "op": "FINISH",
                "value": "completed",
                "target_tag": "div",
                "target_text": "Prime Filter Applied",
                "bbox": [250, 100, 450, 130],
                "html": "<div><h1>Search results with Prime</h1><button id='done-btn' bbox='[250,100,450,130]'>Prime Filter Applied</button></div>"
            }
        ]
    },
    {
        "website": "delta.com",
        "domain": "Travel",
        "task": "Search for one-way flights from SEA to SFO departing next Friday",
        "turns": [
            {
                "url": "https://www.delta.com",
                "thought": "Select one-way flight radio button",
                "op": "CLICK",
                "value": "",
                "target_tag": "input",
                "target_text": "One Way",
                "bbox": [65, 110, 160, 135],
                "html": "<form><label bbox='[65,110,160,135]'><input type='radio' name='tripType' value='oneWay'/>One Way</label><input id='fromAirport' placeholder='From' bbox='[65,150,300,190]'/><input id='toAirport' placeholder='To' bbox='[320,150,550,190]'/></form>"
            },
            {
                "url": "https://www.delta.com",
                "thought": "Enter departure airport SEA into origin input field",
                "op": "TYPE",
                "value": "SEA",
                "target_tag": "input",
                "target_text": "From",
                "bbox": [65, 150, 300, 190],
                "html": "<form><input id='fromAirport' placeholder='From' value='SEA' bbox='[65,150,300,190]'/><input id='toAirport' placeholder='To' bbox='[320,150,550,190]'/><button id='btnSubmit' bbox='[570,150,720,190]'>Find Flights</button></form>"
            },
            {
                "url": "https://www.delta.com",
                "thought": "Click search button to retrieve flight itinerary list",
                "op": "CLICK",
                "value": "",
                "target_tag": "button",
                "target_text": "Find Flights",
                "bbox": [570, 150, 720, 190],
                "html": "<form><input id='fromAirport' value='SEA' bbox='[65,150,300,190]'/><input id='toAirport' value='SFO' bbox='[320,150,550,190]'/><button id='btnSubmit' bbox='[570,150,720,190]'>Find Flights</button></form>"
            }
        ]
    },
    {
        "website": "wikipedia.org",
        "domain": "Reference",
        "task": "Navigate to the English Wikipedia and search for 'Transformer architecture'",
        "turns": [
            {
                "url": "https://www.wikipedia.org",
                "thought": "Locate central search query input field",
                "op": "TYPE",
                "value": "Transformer architecture",
                "target_tag": "input",
                "target_text": "Search Wikipedia",
                "bbox": [380, 310, 820, 350],
                "html": "<main><div class='central-featured'><input id='searchInput' placeholder='Search Wikipedia' bbox='[380,310,820,350]'/><button type='submit' bbox='[825,310,875,350]'>Search</button></div></main>"
            },
            {
                "url": "https://www.wikipedia.org",
                "thought": "Click search submit icon to navigate to article",
                "op": "CLICK",
                "value": "",
                "target_tag": "button",
                "target_text": "Search",
                "bbox": [825, 310, 875, 350],
                "html": "<main><input id='searchInput' value='Transformer architecture' bbox='[380,310,820,350]'/><button type='submit' bbox='[825,310,875,350]'>Search</button></main>"
            }
        ]
    },
    {
        "website": "ebay.com",
        "domain": "Auction",
        "task": "Search for vintage mechanical keyboards and sort by lowest price",
        "turns": [
            {
                "url": "https://www.ebay.com",
                "thought": "Type search term in search bar",
                "op": "TYPE",
                "value": "vintage mechanical keyboard",
                "target_tag": "input",
                "target_text": "Search for anything",
                "bbox": [200, 15, 750, 55],
                "html": "<header><input id='gh-ac' placeholder='Search for anything' bbox='[200,15,750,55]'/><button id='gh-btn' bbox='[760,15,860,55]'>Search</button></header>"
            },
            {
                "url": "https://www.ebay.com/sch/i.html?_nkw=vintage+mechanical+keyboard",
                "thought": "Click sort dropdown button to select Lowest Price First",
                "op": "CLICK",
                "value": "",
                "target_tag": "button",
                "target_text": "Sort: Best Match",
                "bbox": [920, 120, 1080, 160],
                "html": "<div id='srp-controls'><button id='sort-dropdown' bbox='[920,120,1080,160]'>Sort: Best Match</button></div>"
            }
        ]
    }
]


def generate_offline_raw_mind2web(
    output_dir: Path,
    splits: List[str],
    num_samples_per_split: int = 15,
    seed: int = 42,
) -> Dict[str, int]:
    """Generates deterministic raw Mind2Web JSON tasks and HTML snapshots for offline development."""
    random.seed(seed)
    stats: Dict[str, int] = {}

    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_tasks: List[Dict[str, Any]] = []

        for idx in range(num_samples_per_split):
            base_sample = SAMPLE_REAL_TASKS[idx % len(SAMPLE_REAL_TASKS)]
            ann_id = f"m2w_{split}_{idx:04d}_{base_sample['website'].replace('.', '_')}"

            task_obj: Dict[str, Any] = {
                "annotation_id": ann_id,
                "website": base_sample["website"],
                "domain": base_sample["domain"],
                "confirmed_task": base_sample["task"],
                "actions": [],
            }

            for turn_idx, t_data in enumerate(base_sample["turns"]):
                dom_filename = f"dom_{ann_id}_turn{turn_idx}.html"
                dom_path = split_dir / dom_filename
                with open(dom_path, "w", encoding="utf-8") as f:
                    f.write(t_data["html"])

                action_obj = {
                    "action_uid": f"{ann_id}_{turn_idx}",
                    "turn_idx": turn_idx,
                    "url": t_data["url"],
                    "thought": t_data["thought"],
                    "operation": {
                        "op": t_data["op"],
                        "value": t_data["value"],
                    },
                    "target_bounding_box": t_data["bbox"],
                    "target_element": {
                        "tag": t_data["target_tag"],
                        "text": t_data["target_text"],
                        "bbox": t_data["bbox"],
                    },
                    "dom_snapshot_file": dom_filename,
                }
                task_obj["actions"].append(action_obj)

            task_file = split_dir / f"task_{ann_id}.json"
            with open(task_file, "w", encoding="utf-8") as f:
                json.dump(task_obj, f, indent=2)

            split_tasks.append(task_obj)

        manifest_file = split_dir / "tasks_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump([t["annotation_id"] for t in split_tasks], f, indent=2)

        stats[split] = len(split_tasks)
        logger.info(f"Created {len(split_tasks)} raw Mind2Web tasks in {split_dir}")

    return stats


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hf_headers() -> Dict[str, str]:
    headers = {"User-Agent": "JED-M2-LAM/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _official_train_files(revision: str, timeout: int = 30) -> List[Dict[str, Any]]:
    """Reads official file paths and Xet SHA-256 values from the Hugging Face API."""
    url = (
        f"{HF_API_TREE_URL}/{urllib.parse.quote(revision, safe='')}/data/train"
        "?recursive=false&expand=true&limit=1000"
    )
    request = urllib.request.Request(url, headers=_hf_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        entries = json.loads(response.read().decode("utf-8"))

    indexed: List[Dict[str, Any]] = []
    for entry in entries:
        source_path = entry.get("path", "")
        match = re.fullmatch(r"data/train/(train_\d+\.json)", source_path)
        if not match or entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        expected_sha = str(lfs.get("oid", "")).removeprefix("sha256:").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            continue
        indexed.append({
            "name": match.group(1),
            "source_path": source_path,
            "size": int(entry.get("size", 0)),
            "sha256": expected_sha,
        })

    indexed.sort(key=lambda item: int(re.search(r"_(\d+)\.json$", item["name"]).group(1)))
    if not indexed:
        raise RuntimeError("Hugging Face returned no Mind2Web train JSON files with SHA-256 metadata.")
    return indexed


def _find_local_train_file(output_dir: Path, name: str) -> Optional[Path]:
    candidates = (output_dir / name, output_dir / "train" / name)
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1 and len({_sha256_file(path) for path in existing}) > 1:
        raise RuntimeError(f"Conflicting local copies of {name} exist in {output_dir}.")
    return existing[0] if existing else None


def _download_verified_file(
    file_info: Dict[str, Any], output_dir: Path, revision: str, timeout: int = 120
) -> Path:
    name = file_info["name"]
    target = output_dir / "train" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_RESOLVE_URL}/{urllib.parse.quote(revision, safe='')}/{file_info['source_path']}"
    request = urllib.request.Request(url, headers=_hf_headers())
    temporary_path: Optional[Path] = None
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{name}.", suffix=".download", dir=target.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if received % (256 * 1024 * 1024) < len(chunk):
                        logger.info(
                            "Downloaded %s: %.1f / %.1f MiB",
                            name, received / 1048576, file_info["size"] / 1048576,
                        )

        if received != file_info["size"]:
            raise RuntimeError(f"Size mismatch for {name}: received {received}, expected {file_info['size']} bytes.")
        if digest.hexdigest() != file_info["sha256"]:
            raise RuntimeError(f"SHA-256 verification failed for {name}.")
        if target.exists():
            raise FileExistsError(f"Refusing to replace existing data file: {target}")
        os.replace(temporary_path, target)
        temporary_path = None
        return target
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _write_source_manifest(output_dir: Path, manifest: Dict[str, Any]) -> Path:
    manifest_path = output_dir / "source_manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Existing source manifest is unreadable: {manifest_path}") from exc
        if existing.get("files") != manifest.get("files") or existing.get("dataset") != manifest.get("dataset"):
            raise RuntimeError(
                f"Existing source manifest differs from this corpus: {manifest_path}. "
                "Choose a new output directory rather than replacing it."
            )
        return manifest_path

    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".source_manifest.", suffix=".tmp",
            dir=output_dir, delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(serialized)
        os.replace(temporary_path, manifest_path)
        temporary_path = None
        return manifest_path
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except OSError:
                pass


def acquire_real_mind2web(
    output_dir: str = "data/raw_mind2web", revision: str = HF_REVISION
) -> Dict[str, Any]:
    """Verifies local official train chunks or streams them from the official HF dataset."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    verification = "official_sha256"
    try:
        official_files = _official_train_files(revision)
    except Exception as exc:
        logger.warning("Could not retrieve official file checksums (%s); checking the complete local corpus.", exc)
        local_names = {path.name for path in out_path.glob("train_*.json")}
        nested_train = out_path / "train"
        if nested_train.exists():
            local_names.update(path.name for path in nested_train.glob("train_*.json") if path.is_file())
        expected_names = {f"train_{index}.json" for index in range(11)}
        if local_names != expected_names:
            raise RuntimeError(
                "The official Mind2Web source is unreachable and the complete local training corpus was not found. "
                "Acquire the official osunlp/Mind2Web training files or use --mode fixtures explicitly."
            ) from exc
        verification = "official_train0_sha256_plus_local_sha256"
        official_files = [
            {"name": name, "source_path": f"data/train/{name}", "size": 0, "sha256": None}
            for name in sorted(local_names, key=lambda value: int(re.search(r"_(\d+)\.json$", value).group(1)))
        ]

    file_records: List[Dict[str, Any]] = []
    for info in official_files:
        local_path = _find_local_train_file(out_path, info["name"])
        if local_path is None:
            if info["sha256"] is None:
                raise RuntimeError(f"No verified remote digest is available to acquire {info['name']}.")
            local_path = _download_verified_file(info, out_path, revision)

        actual_size = local_path.stat().st_size
        actual_sha = _sha256_file(local_path)
        expected_sha = info.get("sha256")
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f"Local SHA-256 does not match the official source for {info['name']}.")
        if info.get("size") and actual_size != info["size"]:
            raise RuntimeError(f"Local size does not match the official source for {info['name']}.")
        train0_match = info["name"] == "train_0.json" and actual_sha == OFFICIAL_TRAIN0_SHA256
        if info.get("size") == 0 and info["name"] == "train_0.json" and not train0_match:
            raise RuntimeError("The local train_0.json checksum does not match the published Mind2Web file.")
        relative_path = str(local_path.relative_to(out_path)).replace(os.sep, "/")
        file_records.append({
            "path": relative_path,
            "size_bytes": actual_size,
            "sha256": actual_sha,
            "upstream_sha256_verified": bool(expected_sha and actual_sha == expected_sha) or train0_match,
        })

    manifest = {
        "dataset": HF_DATASET_ID,
        "source_url": "https://huggingface.co/datasets/osunlp/Mind2Web",
        "revision": revision if verification == "official_sha256" else "local-unpinned",
        "license": HF_LICENSE,
        "upstream_split": "train",
        "verification": verification,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_records,
    }
    manifest_path = _write_source_manifest(out_path, manifest)
    return {
        "files": len(file_records),
        "bytes": sum(item["size_bytes"] for item in file_records),
        "verification": verification,
        "manifest": str(manifest_path),
    }


def download_or_generate_mind2web(
    output_dir: str = "data/raw_mind2web",
    splits: Optional[List[str]] = None,
    max_tasks_per_split: Optional[int] = None,
    force_offline: bool = False,
    seed: int = 42,
    mode: str = "real",
    revision: str = HF_REVISION,
) -> Dict[str, Any]:
    """Acquire official Mind2Web training data or create explicitly requested fixtures."""
    if force_offline:
        mode = "fixtures"
    if mode == "real":
        if splits and any(split != "train" for split in splits):
            raise ValueError(
                "Official Mind2Web acquisition supplies the train split only; "
                "derive validation/test from a site-held-out manifest."
            )
        summary = acquire_real_mind2web(output_dir=output_dir, revision=revision)
        logger.info("Verified Mind2Web training corpus: %s", summary)
        return summary
    if mode != "fixtures":
        raise ValueError("mode must be either 'real' or 'fixtures'.")

    fixture_splits = splits or ["train", "val", "test"]
    stats = generate_offline_raw_mind2web(
        output_dir=Path(output_dir),
        splits=fixture_splits,
        num_samples_per_split=max_tasks_per_split if max_tasks_per_split is not None else 15,
        seed=seed,
    )
    fixture_manifest = {
        "dataset": "JED-M2 procedural fixtures",
        "source_url": None,
        "revision": "local-generator",
        "license": "not-applicable",
        "upstream_split": None,
        "verification": "synthetic_fixture_generation",
        "seed": seed,
        "files": stats,
    }
    out_path = Path(output_dir)
    (out_path / "fixture_manifest.json").write_text(
        json.dumps(fixture_manifest, indent=2) + "\n", encoding="utf-8"
    )
    logger.warning("Generated synthetic fixtures only; they are not real Mind2Web data.")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mind2Web Dataset Streaming Downloader & Ingestion")
    parser.add_argument("--output_dir", type=str, default="data/raw_mind2web", help="Output directory for raw Mind2Web files")
    parser.add_argument("--mode", choices=["real", "fixtures"], default="real", help="Acquire official Mind2Web training traces or explicitly generate synthetic fixtures")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Fixture splits; real acquisition accepts train only")
    parser.add_argument("--max_tasks", type=int, default=None, help="Fixture task count per split (ignored by real acquisition)")
    parser.add_argument("--force_offline", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--revision", type=str, default=HF_REVISION, help="Pinned Hugging Face dataset revision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    workflow = "fixture generation" if args.mode == "fixtures" or args.force_offline else "real-data acquisition"
    logger.info("Starting Mind2Web %s in %s...", workflow, args.output_dir)
    summary = download_or_generate_mind2web(
        output_dir=args.output_dir,
        splits=args.splits,
        max_tasks_per_split=args.max_tasks,
        force_offline=args.force_offline,
        seed=args.seed,
        mode=args.mode,
        revision=args.revision,
    )
    logger.info("Mind2Web workflow complete: %s", summary)
