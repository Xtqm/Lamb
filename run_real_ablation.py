"""Run a reproducible, real-only, site-held-out four-way JED-M2 ablation."""



from __future__ import annotations



import argparse

from dataclasses import asdict

from datetime import datetime, timezone

import gc

import hashlib

import json

import math

import os

from pathlib import Path

import random
import shutil

import statistics

import time

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple



import torch



from jed_m2.baselines import get_ablation_config

from jed_m2.config import JEDM2Config

from jed_m2.losses import JEDM2CompositeLoss

from jed_m2.model import JEDM2Model

from jed_m2.real_dataset import RealTrajectoryCollator, RealWebTrajectoryDataset

from jed_m2.real_eval import (

    budgets_match,

    check_split_integrity,

    code_revision,

    count_window_tokens,

    create_run_directory,

    load_real_manifest,

    matched_epoch_orders,

    score_offline_windows,

    sha256_json,

    split_into_context_windows,

    write_json,

)





BASELINES = [

    ("baseline_a", "Pure Transformer", "Standard MHA + Dense SwiGLU"),

    ("baseline_b", "Hybrid Attention-SSM", "Mamba SSM + MLA + Dense SwiGLU"),

    ("baseline_c", "JED-M2 without JEPA", "Mamba SSM + MLA + Sparse Top-2 MoE"),

    ("baseline_d", "Full JED-M2", "Mamba SSM + MLA + Sparse MoE + JEPA"),

]





def _clear_device(device: torch.device) -> None:

    gc.collect()

    if device.type == "cuda":

        torch.cuda.empty_cache()





def _profile_model(

    model: JEDM2Model,

    device: torch.device,

    dtype: torch.dtype,

    horizons: Sequence[int],

    latency_steps: int,

) -> Dict[str, Any]:

    model = model.to(device=device, dtype=dtype).eval()

    token = torch.tensor([42], device=device, dtype=torch.long)

    memory: Dict[str, Dict[str, float]] = {}

    with torch.inference_mode():

        for horizon in horizons:

            if device.type == "cuda":

                torch.cuda.empty_cache()

                torch.cuda.reset_peak_memory_stats(device)

            states = None

            for position in range(int(horizon)):

                _, _, states = model.step(token, states=states, pos=position)

            if device.type == "cuda":

                torch.cuda.synchronize(device)

                memory[str(horizon)] = {

                    "allocated_mb": torch.cuda.memory_allocated(device) / (1024 * 1024),

                    "peak_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),

                }

            else:

                memory[str(horizon)] = {"allocated_mb": 0.0, "peak_mb": 0.0}

            del states



        states = None

        for position in range(10):

            _, _, states = model.step(token, states=states, pos=position)

        if device.type == "cuda":

            torch.cuda.synchronize(device)

        latencies: List[float] = []

        states = None

        for position in range(latency_steps):

            start = time.perf_counter()

            _, _, states = model.step(token, states=states, pos=position)

            if device.type == "cuda":

                torch.cuda.synchronize(device)

            latencies.append((time.perf_counter() - start) * 1000.0)

    del model

    _clear_device(device)

    ordered = sorted(latencies)

    return {

        "memory_by_horizon": memory,

        "latency_ms": {

            "mean": statistics.mean(latencies) if latencies else 0.0,

            "median": statistics.median(latencies) if latencies else 0.0,

            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0,

            "std": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,

            "steps": len(latencies),

        },

    }





def _aggregate_seed_results(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:

    metric_names = (

        "action_accuracy", "coordinate_mae", "coordinate_tolerance_accuracy",

        "target_element_hit_rate", "step_accuracy", "exact_trajectory_match_rate",

    )

    aggregate: Dict[str, Any] = {}

    for metric in metric_names:

        values = [

            result["evaluation"]["overall_micro"].get(metric)

            for result in results

            if result.get("status") == "complete"

            and result.get("evaluation", {}).get("overall_micro", {}).get(metric) is not None

        ]

        aggregate[metric] = {

            "n": len(values),

            "mean": statistics.mean(values) if values else None,

            "std_population": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),

        }

    return aggregate





def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--manifest", type=Path, required=True, help="Prepared site-held-out real-data manifest JSON.")

    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw_mind2web"))

    parser.add_argument("--output-root", type=Path, default=Path("runs/ablation"))

    parser.add_argument("--run-id", type=str, default=None)

    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])

    parser.add_argument("--passes", type=int, default=3)

    parser.add_argument("--max-seq-len", type=int, default=1024)

    parser.add_argument("--grad-accumulation", type=int, default=4)

    parser.add_argument("--learning-rate", type=float, default=3e-4)

    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--coordinate-tolerance", type=float, default=0.05)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(

        "--dtype", choices=("bfloat16", "float32"),

        default="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float32",

    )

    parser.add_argument("--cache-dir", type=Path, default=Path("data/real_trajectories_cache"))

    parser.add_argument(

        "--legacy-checkpoint", type=Path,

        default=Path("checkpoints/pretrain_nano/jed_m2_nano_step_19500.pt"),

    )

    parser.add_argument("--profile-steps", type=int, default=40)

    parser.add_argument(

        "--overwrite",

        action="store_true",

        help="Replace an existing run directory with the same --run-id. Disabled by default.",

    )

    parser.add_argument(

        "--resume",

        action="store_true",

        help="Resume an existing run directory, skipping completed models/seeds.",

    )

    return parser.parse_args()





def main() -> None:

    args = _parse_args()

    if args.passes != 3:

        raise ValueError("This experiment was preregistered for exactly three complete passes.")

    if len(set(args.seeds)) < 3:

        raise ValueError("At least three distinct matched seeds are required.")

    if args.max_seq_len < 64 or args.grad_accumulation < 1:

        raise ValueError("Context length and gradient accumulation must be positive and practical.")

    device = torch.device(args.device)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    if Path(args.output_root).resolve() == Path("ABLATION_REPORT.md").resolve():

        raise ValueError("Real-data ablation reports cannot overwrite ABLATION_REPORT.md.")

    manifest = load_real_manifest(args.manifest)

    website_counts = check_split_integrity(manifest)



    common_config = JEDM2Config(max_seq_len=args.max_seq_len)

    train_dataset = RealWebTrajectoryDataset(

        config=common_config,

        split="train",

        real_only=True,

        manifest_path=str(args.manifest),

        raw_mind2web_dir=str(args.raw_dir),

        cache_dir=str(args.cache_dir),

        seed=args.seeds[0],

    )

    test_dataset = RealWebTrajectoryDataset(

        config=common_config,

        split="test",

        real_only=True,

        manifest_path=str(args.manifest),

        raw_mind2web_dir=str(args.raw_dir),

        cache_dir=str(args.cache_dir),

        seed=args.seeds[0],

    )

    train_collator = RealTrajectoryCollator(common_config, max_seq_len=args.max_seq_len)

    eval_collator = RealTrajectoryCollator(common_config, max_seq_len=args.max_seq_len)

    train_windows = split_into_context_windows(train_dataset.samples, train_collator)

    test_windows = split_into_context_windows(test_dataset.samples, eval_collator)

    train_budget = count_window_tokens(train_windows, train_collator)

    test_budget = count_window_tokens(test_windows, eval_collator)

    train_actions = sum(len(sample.turns) for sample in train_dataset.samples)

    test_actions = sum(len(sample.turns) for sample in test_dataset.samples)

    if train_actions != train_budget["actions"] or test_actions != test_budget["actions"]:

        raise ValueError("Context-windowing lost or repeated a source action.")



    run_id = args.run_id or datetime.now(timezone.utc).strftime("real-data-%Y%m%dT%H%M%SZ")

    output_dir = create_run_directory(args.output_root, run_id, overwrite=args.overwrite, resume=args.resume)



    model_profiles: Dict[str, Dict[str, Any]] = {}

    configs: Dict[str, Any] = {}

    parameter_rows: List[Dict[str, Any]] = []

    for baseline_id, name, architecture in BASELINES:

        config = get_ablation_config(baseline_id, max_seq_len=args.max_seq_len)

        configs[baseline_id] = config

        profile_model = JEDM2Model(config)

        params = profile_model.count_parameters()

        if profile_model.action_head.action_type_head.num_action_types != 7:

            raise ValueError(f"{baseline_id} does not emit the seven-class action set.")

        parameter_rows.append({

            "id": baseline_id,

            "name": name,

            "architecture": architecture,

            "configuration": asdict(config),

            "parameters": params,

            "action_head_classes": profile_model.action_head.action_type_head.num_action_types,

        })

        del profile_model

    trainable_counts = [row["parameters"]["trainable_params"] for row in parameter_rows]

    parameter_target = statistics.mean(trainable_counts)

    max_parameter_deviation = max(abs(value - parameter_target) / parameter_target for value in trainable_counts)

    if max_parameter_deviation > 0.01:

        raise ValueError(f"Trainable parameter mismatch exceeds 1%: {max_parameter_deviation:.3%}")



    train_epoch_orders = matched_epoch_orders(len(train_windows), args.seeds, args.passes)



    source = manifest["source"]

    run_manifest = {

        "schema": "jed-m2.real-ablation-run.v1",

        "run_id": run_id,

        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        "status": "preparing",

        "source": {

            "dataset": source["dataset"],

            "revision": source["revision"],

            "license": source["license"],

            "verification": source["verification"],

            "source_manifest_sha256": source["source_manifest_sha256"],

        },

        "manifest_path": str(args.manifest),

        "manifest_sha256": manifest["manifest_sha256"],

        "website_counts": website_counts,

        "split_counts": manifest["audit"]["splits"],

        "train_data_audit": train_dataset.real_data_audit,

        "test_data_audit": test_dataset.real_data_audit,

        "train_budget_per_epoch": train_budget,

        "test_budget": test_budget,

        "unique_train_tasks_per_epoch": len(train_dataset),

        "unique_test_tasks": len(test_dataset),

        "passes": args.passes,

        "seeds": args.seeds,

        "matched_order_sha256_by_seed": {

            str(seed): [

                sha256_json([train_windows[index].sample_id for index in order])

                for order in train_epoch_orders[seed]

            ]

            for seed in args.seeds

        },

        "optimizer": {

            "name": "AdamW",

            "learning_rate": args.learning_rate,

            "weight_decay": args.weight_decay,

            "gradient_accumulation": args.grad_accumulation,

            "micro_batch_size": 1,

            "precision": args.dtype,

            "context_length": args.max_seq_len,

        },

        "parameter_budget": {

            "target_trainable_parameters": round(parameter_target),

            "max_deviation_percent": max_parameter_deviation * 100,

            "tolerance_percent": 1.0,

            "models": parameter_rows,

        },

        "initialization": "fresh random initialization for every model and seed; no pretrained or legacy checkpoint weights loaded",

        "code_revision": code_revision(),

        "device": str(device),

        "hardware": {

            "name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",

            "cuda_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,

        },

        "evaluation_protocol": "offline teacher-forced real-trace replay in non-overlapping context windows",

        "limitations": [

            "The site-held-out test set is a custom split of the Mind2Web training corpus, not the official Mind2Web test split.",

            "Offline trace replay does not measure closed-loop live-web task completion.",

        ],

    }

    write_json(output_dir / "run_manifest.json", run_manifest)



    device_profile: Dict[str, Any] = {}

    run_manifest["status"] = "profiling"

    write_json(output_dir / "run_manifest.json", run_manifest)

    for baseline_id, _, _ in BASELINES:

        profile = _profile_model(

            JEDM2Model(configs[baseline_id]),

            device,

            dtype,

            horizons=(16, 32, 64, 128),

            latency_steps=args.profile_steps,

        )

        device_profile[baseline_id] = profile

        print(f"[profile] {baseline_id}: median={profile['latency_ms']['median']:.3f} ms/step")

    run_manifest["hardware_profile"] = device_profile

    run_manifest["status"] = "training"

    write_json(output_dir / "run_manifest.json", run_manifest)



    run_records: List[Dict[str, Any]] = []

    run_records_path = output_dir / "run_records.json"

    completed_runs: Dict[Tuple[str, int], Dict[str, Any]] = {}

    if args.resume and run_records_path.is_file():

        try:

            for rec in json.loads(run_records_path.read_text(encoding="utf-8")):

                if rec.get("status") == "complete":

                    completed_runs[(rec["model"], int(rec["seed"]))] = rec

        except Exception:

            pass



    total_runs = len(BASELINES) * len(args.seeds)

    run_number = 0

    for baseline_id, model_name, _ in BASELINES:

        for seed in args.seeds:

            run_number += 1

            if (baseline_id, seed) in completed_runs:

                existing_record = completed_runs[(baseline_id, seed)]

                run_records.append(existing_record)

                print(f"[{run_number}/{total_runs}] Skipping {baseline_id} seed={seed} (already complete)")

                continue



            run_dir = output_dir / "runs" / baseline_id / f"seed_{seed}"

            if run_dir.exists():

                shutil.rmtree(run_dir)

            run_dir.mkdir(parents=True, exist_ok=False)

            record: Dict[str, Any] = {

                "model": baseline_id,

                "model_name": model_name,

                "seed": seed,

                "status": "running",

                "passes": args.passes,

                "unique_train_tasks_per_epoch": len(train_dataset),

                "train_actions_per_epoch": train_actions,

                "windows_per_epoch": train_budget["windows"],

                "tokens_per_epoch": train_budget["tokens"],

                "total_examples": len(train_windows) * args.passes,

                "total_actions": train_actions * args.passes,

                "total_tokens": train_budget["tokens"] * args.passes,

                "optimizer_updates_per_epoch": math.ceil(train_budget["windows"] / args.grad_accumulation),

                "optimizer_updates": math.ceil(train_budget["windows"] / args.grad_accumulation) * args.passes,

                "parameter_counts": next(row["parameters"] for row in parameter_rows if row["id"] == baseline_id),

                "matched_order_sha256": run_manifest["matched_order_sha256_by_seed"][str(seed)],

                "started_at_utc": datetime.now(timezone.utc).isoformat(),

                "history": [],

            }

            model: Optional[JEDM2Model] = None

            try:

                torch.manual_seed(seed)

                if device.type == "cuda":

                    torch.cuda.manual_seed_all(seed)

                    torch.cuda.reset_peak_memory_stats(device)

                model = JEDM2Model(configs[baseline_id]).to(device=device)

                model.gradient_checkpointing_enable()

                model.train()

                loss_fn = JEDM2CompositeLoss(model.config)

                optimizer = torch.optim.AdamW(

                    model.parameters(),

                    lr=args.learning_rate,

                    weight_decay=args.weight_decay,

                    fused=device.type == "cuda",

                )

                token_total = 0

                update_total = 0

                start_time = time.perf_counter()

                metrics_path = run_dir / "training_metrics.jsonl"

                for epoch_index, order in enumerate(train_epoch_orders[seed], start=1):

                    epoch_rows: List[Dict[str, float]] = []

                    pending = 0

                    optimizer.zero_grad(set_to_none=True)

                    with metrics_path.open("a", encoding="utf-8") as metric_file:

                        for local_index, window_index in enumerate(order):

                            window = train_windows[window_index]

                            raw_batch = train_collator([window])

                            token_total += int((raw_batch.input_ids != train_collator.tokenizer.PAD_TOKEN_ID).sum().item())

                            batch = raw_batch.to(device=device, dtype=dtype)

                            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):

                                output = model(

                                    input_ids=batch.input_ids,

                                    attention_mask=batch.attention_mask,

                                    action_t=batch.jepa_action_t if model.use_jepa else None,

                                    obs_next=batch.jepa_obs_next if model.use_jepa else None,

                                )

                                loss, metrics = loss_fn(output, batch)

                            if not bool(torch.isfinite(loss).item()):

                                raise FloatingPointError(f"Non-finite loss at epoch {epoch_index}, window {window.sample_id}")

                            (loss / args.grad_accumulation).backward()

                            pending += 1

                            values = {key: float(value) for key, value in metrics.items()}

                            epoch_rows.append(values)

                            metric_file.write(json.dumps({

                                "epoch": epoch_index,

                                "window": window.sample_id,

                                "source_task": window.provenance.get("parent_annotation_id"),

                                **values,

                            }, allow_nan=False) + "\n")



                            is_last = local_index + 1 == len(order)

                            if pending == args.grad_accumulation or is_last:

                                if pending < args.grad_accumulation:

                                    correction = args.grad_accumulation / pending

                                    for parameter in model.parameters():

                                        if parameter.grad is not None:

                                            parameter.grad.mul_(correction)

                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                                optimizer.step()

                                optimizer.zero_grad(set_to_none=True)

                                pending = 0

                                update_total += 1

                            if (local_index + 1) % 100 == 0 or is_last:

                                print(

                                    f"[{run_number}/{total_runs}] {baseline_id} seed={seed} "

                                    f"epoch={epoch_index}/{args.passes} window={local_index + 1}/{len(order)} "

                                    f"loss={values.get('loss_total', 0.0):.4f}"

                                )

                    epoch_summary = {

                        "epoch": epoch_index,

                        "windows": len(order),

                        "actions": train_actions,

                        "tokens": train_budget["tokens"],

                        "optimizer_updates": math.ceil(len(order) / args.grad_accumulation),

                        "loss_mean": statistics.mean(row.get("loss_total", 0.0) for row in epoch_rows),

                        "loss_reason_mean": statistics.mean(row.get("loss_reason", 0.0) for row in epoch_rows),

                        "action_accuracy_mean": statistics.mean(row.get("acc_action_type", 0.0) for row in epoch_rows),

                        "coordinate_mae_mean": statistics.mean(row.get("coord_mae", 0.0) for row in epoch_rows),

                        "finite": all(math.isfinite(row.get("loss_total", 0.0)) for row in epoch_rows),

                    }

                    record["history"].append(epoch_summary)

                    write_json(run_dir / "status.json", {**record, "latest_epoch": epoch_summary})



                elapsed = time.perf_counter() - start_time

                _clear_device(device)

                evaluation = score_offline_windows(

                    model,

                    test_windows,

                    eval_collator,

                    device,

                    dtype,

                    coordinate_tolerance=args.coordinate_tolerance,

                )

                peak_memory_mb = (

                    torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0

                )

                checkpoint_path = run_dir / "checkpoint_final.pt"

                checkpoint_state = {

                    key: value.detach().to(device="cpu", dtype=torch.bfloat16)

                    for key, value in model.state_dict().items()

                }

                torch.save({

                    "model_state_dict": checkpoint_state,

                    "model_config": asdict(model.config),

                    "run_id": run_id,

                    "manifest_sha256": manifest["manifest_sha256"],

                    "seed": seed,

                    "model": baseline_id,

                    "passes": args.passes,

                }, checkpoint_path)

                record.update({

                    "status": "complete",

                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),

                    "elapsed_seconds": elapsed,

                    "tokens_seen": token_total,

                    "optimizer_updates_completed": update_total,

                    "throughput_tokens_per_second": token_total / elapsed if elapsed else 0.0,

                    "evaluation": evaluation,

                    "checkpoint": str(checkpoint_path),

                    "peak_memory_mb": peak_memory_mb,

                })

            except Exception as error:

                record.update({

                    "status": "failed",

                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),

                    "error": f"{type(error).__name__}: {error}",

                })

                try:

                    (run_dir / "error.txt").write_text(record["error"] + "\n", encoding="utf-8")

                except OSError:

                    pass

                print(f"[failed] {baseline_id} seed={seed}: {record['error']}")

            finally:

                if model is not None:

                    del model

                if "optimizer" in locals() and optimizer is not None:

                    del optimizer

                if "loss_fn" in locals() and loss_fn is not None:

                    del loss_fn

                _clear_device(device)

                record["finished_at_utc"] = record.get("finished_at_utc", datetime.now(timezone.utc).isoformat())

                run_records.append(record)

                write_json(output_dir / "run_records.json", run_records)



    legacy_record: Optional[Dict[str, Any]] = None

    if args.legacy_checkpoint.is_file():

        try:

            saved = torch.load(args.legacy_checkpoint, map_location="cpu", weights_only=False)

            legacy_config = JEDM2Config(**saved["model_config"])

            legacy_model = JEDM2Model(legacy_config)

            legacy_model.load_state_dict(saved["model_state_dict"], strict=True)

            legacy_model = legacy_model.to(device=device).eval()

            legacy_eval_collator = RealTrajectoryCollator(legacy_config, max_seq_len=args.max_seq_len)

            legacy_windows = split_into_context_windows(test_dataset.samples, legacy_eval_collator)

            legacy_metrics = score_offline_windows(legacy_model, legacy_windows, legacy_eval_collator, device, dtype)

            legacy_record = {

                "label": "legacy step-19,500 checkpoint reference; excluded from matched-baseline aggregate",

                "checkpoint": str(args.legacy_checkpoint),

                "global_step": saved.get("global_step"),

                "num_action_types": legacy_config.num_action_types,

                "head_shape": list(legacy_model.action_head.action_type_head.fc2.weight.shape),

                "metadata": saved.get("additional_info", {}),

                "evaluation": legacy_metrics,

            }

            del legacy_model, saved

            _clear_device(device)

        except Exception as error:

            legacy_record = {"status": "unavailable", "error": f"{type(error).__name__}: {error}"}



    by_model: Dict[str, List[Dict[str, Any]]] = {

        baseline_id: [record for record in run_records if record["model"] == baseline_id]

        for baseline_id, _, _ in BASELINES

    }

    report_data = {

        "schema": "jed-m2.real-ablation-report.v1",

        "run_id": run_id,

        "status": "complete" if len(run_records) == total_runs and all(r["status"] == "complete" for r in run_records) else "incomplete",

        "manifest_sha256": manifest["manifest_sha256"],

        "source": run_manifest["source"],

        "run_manifest": "run_manifest.json",

        "parameter_budget": run_manifest["parameter_budget"],

        "hardware_profile": device_profile,

        "training": {

            "passes": args.passes,

            "seeds": args.seeds,

            "unique_tasks_per_epoch": len(train_dataset),

            "actions_per_epoch": train_actions,

            "windows_per_epoch": train_budget["windows"],

            "tokens_per_epoch": train_budget["tokens"],

            "runs": run_records,

            "aggregates_by_model": {

                baseline_id: _aggregate_seed_results(records)

                for baseline_id, records in by_model.items()

            },

        },

        "legacy_reference": legacy_record,

        "test_site_count": website_counts["test"],

        "evaluation_protocol": run_manifest["evaluation_protocol"],

        "limitations": run_manifest["limitations"],

    }

    write_json(output_dir / "report.json", report_data)

    run_manifest["status"] = report_data["status"]

    run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    write_json(output_dir / "run_manifest.json", run_manifest)

    _write_markdown_report(output_dir / "report.md", report_data)

    print(f"Ablation artifacts: {output_dir.resolve()}")





def _write_markdown_report(path: Path, report: Dict[str, Any]) -> None:

    lines = [

        "# Real-Data Backbone Ablation",

        "",

        f"- Run: `{report['run_id']}`",

        f"- Status: **{report['status']}**",

        f"- Source: `{report['source']['dataset']}` revision `{report['source']['revision']}`",

        f"- License: `{report['source']['license']}`; provenance: `{report['source']['verification']}`",

        f"- Manifest SHA-256: `{report['manifest_sha256']}`",

        f"- Evaluation: {report['evaluation_protocol']}",

        "",

        "The held-out set is a custom website split of Mind2Web's training corpus. These results are offline trace replay and do not represent official Mind2Web test performance or closed-loop live-web task success.",

        "",

        "## Data and Budget",

        "",

        f"- Unique training tasks per pass: {report['training']['unique_tasks_per_epoch']}",

        f"- Source actions per pass: {report['training']['actions_per_epoch']}",

        f"- Non-overlapping context windows per pass: {report['training']['windows_per_epoch']}",

        f"- Input tokens per pass: {report['training']['tokens_per_epoch']}",

        f"- Test websites: {report['test_site_count']}",

        f"- Seeds: {', '.join(str(seed) for seed in report['training']['seeds'])}; passes per seed: {report['training']['passes']}",

        "",

        "## Matched Baselines",

        "",

        "| Model | Trainable params | Active params | Max budget deviation | Action accuracy | Coordinate MAE | Coordinate tolerance | Target-element hit | Step accuracy | Exact trajectory |",

        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",

    ]

    deviation = report["parameter_budget"]["max_deviation_percent"]

    for baseline_id, name, _ in BASELINES:

        aggregate = report["training"]["aggregates_by_model"][baseline_id]

        matching = next(row for row in report["parameter_budget"]["models"] if row["id"] == baseline_id)

        def fmt(metric: str) -> str:

            data = aggregate[metric]

            if data["mean"] is None:

                return "n/a"

            return f"{data['mean']:.4f} +/- {data['std_population']:.4f}"

        lines.append(

            f"| {name} | {matching['parameters']['trainable_params']:,} | {matching['parameters']['active_params']:,} "

            f"| {deviation:.4f}% | {fmt('action_accuracy')} | {fmt('coordinate_mae')} "

            f"| {fmt('coordinate_tolerance_accuracy')} | {fmt('target_element_hit_rate')} "

            f"| {fmt('step_accuracy')} | {fmt('exact_trajectory_match_rate')} |"

        )

    lines.extend(["", "Per-seed and per-website measurements are in `report.json`; each run record includes its token budget, update count, timing, peak hardware profile, and terminal status.", ""])

    lines.extend(["## Hardware Profile", "", "| Model | Median step latency (ms) | p95 step latency (ms) | Peak allocation at 128 steps (MB) |", "|---|---:|---:|---:|"])

    for baseline_id, name, _ in BASELINES:

        profile = report["hardware_profile"][baseline_id]

        lines.append(

            f"| {name} | {profile['latency_ms']['median']:.3f} | {profile['latency_ms']['p95']:.3f} "

            f"| {profile['memory_by_horizon']['128']['peak_mb']:.1f} |"

        )

    if report.get("legacy_reference"):

        legacy = report["legacy_reference"]

        lines.extend([

            "",

            "## Legacy Checkpoint Reference",

            "",

            f"- Label: {legacy.get('label', legacy.get('status', 'unavailable'))}",

            f"- Checkpoint: `{legacy.get('checkpoint', '')}`",

            f"- Action classes: {legacy.get('num_action_types', 'n/a')} (SELECT is unscorable)",

        ])

        if legacy.get("evaluation"):

            metrics = legacy["evaluation"]["overall_micro"]

            lines.append(f"- Action accuracy: {metrics.get('action_accuracy')}; SELECT labels unscorable: {metrics.get('unscorable_select')}")

            lines.append(f"- Exact trajectory match rate: {metrics.get('exact_trajectory_match_rate')}")

    lines.extend(["", "## Limitations", ""])

    lines.extend(f"- {item}" for item in report["limitations"])

    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")





if __name__ == "__main__":

    main()

