"""Unit and integration tests for real-data backbone ablation components."""

from pathlib import Path
import pytest
import torch

from jed_m2.baselines import get_ablation_config
from jed_m2.config import JEDM2Config
from jed_m2.model import JEDM2Model
from jed_m2.real_eval import (
    budgets_match,
    check_split_integrity,
    create_run_directory,
    load_real_manifest,
    matched_epoch_orders,
    sha256_json,
)


def test_manifest_split_integrity_real_mind2web():
    manifest_path = Path("data/manifests/real_mind2web_v1/manifest.json")
    assert manifest_path.is_file(), "Manifest file must exist"
    manifest = load_real_manifest(manifest_path)
    
    assert manifest["schema"] == "jed-m2.real-web-manifest.v1"
    assert manifest["status"] == "ready"
    assert manifest["source"]["dataset"] == "osunlp/Mind2Web"

    website_counts = check_split_integrity(manifest)
    assert website_counts["train"] >= 50, f"Expected >= 50 train sites, got {website_counts['train']}"
    assert website_counts["val"] >= 10, f"Expected >= 10 val sites, got {website_counts['val']}"
    assert website_counts["test"] >= 10, f"Expected >= 10 test sites, got {website_counts['test']}"


def test_baseline_parameter_matching_and_action_heads():
    baseline_ids = ["baseline_a", "baseline_b", "baseline_c", "baseline_d"]
    param_counts = []
    
    for b_id in baseline_ids:
        cfg = get_ablation_config(b_id, max_seq_len=512)
        model = JEDM2Model(cfg)
        params = model.count_parameters()
        trainable = params["trainable_params"]
        param_counts.append(trainable)
        assert model.action_head.action_type_head.num_action_types == 7, (
            f"{b_id} action head must have 7 action types"
        )
    
    target_mean = sum(param_counts) / len(param_counts)
    max_dev = max(abs(p - target_mean) / target_mean for p in param_counts)
    assert max_dev <= 0.01, f"Trainable parameter count mismatch exceeds 1%: {max_dev:.3%}"

    # Verify legacy config retains 6 classes
    legacy_cfg = JEDM2Config(num_action_types=6, max_seq_len=512)
    legacy_model = JEDM2Model(legacy_cfg)
    assert legacy_model.action_head.action_type_head.num_action_types == 6


def test_matched_epoch_orders_determinism():
    seeds = [42, 43, 44]
    passes = 3
    window_count = 50
    orders1 = matched_epoch_orders(window_count, seeds, passes)
    orders2 = matched_epoch_orders(window_count, seeds, passes)
    
    assert orders1 == orders2, "Order generation must be strictly deterministic"
    for s in seeds:
        assert len(orders1[s]) == passes
        for ep in range(passes):
            assert len(orders1[s][ep]) == window_count
            assert sorted(orders1[s][ep]) == list(range(window_count))


def test_run_directory_isolation_and_protections(tmp_path):
    root = tmp_path / "runs"
    run_dir = create_run_directory(root, "run-test-1", overwrite=False)
    assert run_dir.exists()
    assert (run_dir / "runs").exists()
    
    with pytest.raises(FileExistsError):
        create_run_directory(root, "run-test-1", overwrite=False)
        
    run_dir_recreated = create_run_directory(root, "run-test-1", overwrite=True)
    assert run_dir_recreated.exists()
    
    with pytest.raises(ValueError, match="ABLATION_REPORT.md"):
        create_run_directory(tmp_path, "ABLATION_REPORT.md", overwrite=False)


def test_budgets_match():
    records = [
        {"seed": 42, "tokens_seen": 1000, "total_tokens": 1000, "optimizer_updates_completed": 20, "unique_train_tasks_per_epoch": 50},
        {"seed": 42, "tokens_seen": 1000, "total_tokens": 1000, "optimizer_updates_completed": 20, "unique_train_tasks_per_epoch": 50},
        {"seed": 43, "tokens_seen": 1000, "total_tokens": 1000, "optimizer_updates_completed": 20, "unique_train_tasks_per_epoch": 50},
    ]
    assert budgets_match(records)
    
    mismatched = [
        {"seed": 42, "tokens_seen": 1000, "total_tokens": 1000, "optimizer_updates_completed": 20, "unique_train_tasks_per_epoch": 50},
        {"seed": 42, "tokens_seen": 999, "total_tokens": 1000, "optimizer_updates_completed": 20, "unique_train_tasks_per_epoch": 50},
    ]
    assert not budgets_match(mismatched)


def test_reject_synthetic_or_leaked_manifest(tmp_path):
    synthetic_manifest = {
        "schema": "jed-m2.real-web-manifest.v1",
        "status": "ready",
        "source": {"dataset": "osunlp/Mind2Web", "verification": "synthetic_fixtures"},
        "audit": {"pairwise_website_overlap": 0, "leakage_snapshot_count": 0},
    }
    synthetic_manifest["manifest_sha256"] = sha256_json(synthetic_manifest)
    p = tmp_path / "synthetic_manifest.json"
    p.write_text(torch.hub.json.dumps(synthetic_manifest))
    with pytest.raises(ValueError, match="Synthetic fixture provenance is not accepted"):
        load_real_manifest(p)

    leaked_manifest = {
        "schema": "jed-m2.real-web-manifest.v1",
        "status": "ready",
        "source": {"dataset": "osunlp/Mind2Web", "verification": "official_hash"},
        "audit": {"pairwise_website_overlap": 2, "leakage_snapshot_count": 0},
    }
    leaked_manifest["manifest_sha256"] = sha256_json(leaked_manifest)
    p2 = tmp_path / "leaked_manifest.json"
    p2.write_text(torch.hub.json.dumps(leaked_manifest))
    with pytest.raises(ValueError, match="Manifest contains website or snapshot leakage"):
        load_real_manifest(p2)


def test_score_offline_windows_metric_keys():
    from jed_m2.real_eval import metric_rates, _new_site_bucket
    bucket = _new_site_bucket()
    bucket["actions"] = 10
    bucket["action_correct"] = 8
    bucket["coord_count"] = 5
    bucket["coord_abs_sum"] = 0.5
    bucket["coord_tolerance_correct"] = 4
    bucket["target_count"] = 5
    bucket["target_hit"] = 4
    bucket["step_count"] = 10
    bucket["step_correct"] = 7
    bucket["tasks"] = 2
    bucket["trajectory_correct"] = 1
    
    rates = metric_rates(bucket)
    expected_keys = {
        "actions", "action_accuracy", "unscorable_select", "coordinate_count",
        "coordinate_mae", "coordinate_tolerance_accuracy", "target_element_count",
        "target_element_hit_rate", "step_accuracy", "step_count", "tasks",
        "exact_trajectory_match_rate", "trajectory_unscorable",
    }
    assert expected_keys.issubset(set(rates.keys()))
    assert rates["action_accuracy"] == 0.8
    assert rates["coordinate_mae"] == 0.05
    assert rates["target_element_hit_rate"] == 0.8
    assert rates["step_accuracy"] == 0.7
    assert rates["exact_trajectory_match_rate"] == 0.5


def test_markdown_report_generation(tmp_path):
    from run_real_ablation import _write_markdown_report, BASELINES
    report_data = {
        "run_id": "test-run-123",
        "status": "complete",
        "source": {"dataset": "osunlp/Mind2Web", "revision": "v1.0", "license": "MIT", "verification": "official_hash"},
        "manifest_sha256": "abcdef123456",
        "evaluation_protocol": "offline teacher-forced replay",
        "test_site_count": 11,
        "parameter_budget": {
            "max_deviation_percent": 0.05,
            "models": [
                {"id": b_id, "parameters": {"trainable_params": 1000000, "active_params": 1000000}}
                for b_id, _, _ in BASELINES
            ],
        },
        "training": {
            "unique_tasks_per_epoch": 718,
            "actions_per_epoch": 5607,
            "windows_per_epoch": 2776,
            "tokens_per_epoch": 2079566,
            "seeds": [42, 43, 44],
            "passes": 3,
            "aggregates_by_model": {
                b_id: {
                    "action_accuracy": {"mean": 0.65, "std_population": 0.02},
                    "coordinate_mae": {"mean": 0.08, "std_population": 0.01},
                    "coordinate_tolerance_accuracy": {"mean": 0.72, "std_population": 0.03},
                    "target_element_hit_rate": {"mean": 0.60, "std_population": 0.02},
                    "step_accuracy": {"mean": 0.55, "std_population": 0.02},
                    "exact_trajectory_match_rate": {"mean": 0.25, "std_population": 0.01},
                }
                for b_id, _, _ in BASELINES
            },
        },
        "hardware_profile": {
            b_id: {
                "latency_ms": {"median": 1.25, "p95": 1.50},
                "memory_by_horizon": {"128": {"peak_mb": 420.0}},
            }
            for b_id, _, _ in BASELINES
        },
        "legacy_reference": {
            "label": "legacy step-19,500 checkpoint reference",
            "checkpoint": "checkpoints/pretrain_nano/jed_m2_nano_step_19500.pt",
            "num_action_types": 6,
            "evaluation": {"overall_micro": {"action_accuracy": 0.50, "unscorable_select": 12, "exact_trajectory_match_rate": 0.15}},
        },
        "limitations": [
            "Custom site-held-out split.",
            "Offline trace replay does not measure closed-loop web execution.",
        ],
    }
    out_md = tmp_path / "report.md"
    _write_markdown_report(out_md, report_data)
    assert out_md.is_file()
    content = out_md.read_text(encoding="utf-8")
    assert "Real-Data Backbone Ablation" in content
    assert "Pure Transformer" in content
    assert "Full JED-M2" in content
    assert "Legacy Checkpoint Reference" in content
    assert "SELECT is unscorable" in content



