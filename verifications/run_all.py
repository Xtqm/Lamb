"""
Centralized Phase Verification Runner for JED-M2 Large Action Model.
Executes individual or sequential verification suites with structured ASCII reporting.

Usage:
    python verifications/run_all.py --list
    python verifications/run_all.py --phase 1
    python verifications/run_all.py --phase 6c
    python verifications/run_all.py --quick
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

PHASE_REGISTRY: Dict[str, Tuple[str, str]] = {
    "1": ("verify_phase1.py", "Architecture: MLA + Sparse Top-2 MoE"),
    "2": ("verify_phase2.py", "Backbone: JEDM2Block Multi-Layer Recurrence"),
    "3": ("verify_phase3.py", "World Model: JEPA Latent Dynamics & Rollout"),
    "4": ("verify_phase4.py", "Datasets: Trajectory Sampling & Collation"),
    "5": ("verify_phase5.py", "Training: Fused AdamW & RL Policy Scaffolding"),
    "6a": ("verify_phase6a.py", "Environment: WebOS Simulation & Native Agent"),
    "6b": ("verify_phase6b.py", "RL Pipeline: Multi-Turn Rollout Sampler"),
    "6c": ("verify_phase6c.py", "Production SFT: 1000-Step Convergence Harness"),
    "6d": ("verify_phase6d.py", "Action Heads: Dual-System Decoupled Heads"),
    "7": ("verify_phase7.py", "Browser: Playwright Headless Automation & Oracles"),
    "8": ("verify_phase8.py", "Evaluation: Standardized Benchmark Harness"),
    "9": ("verify_phase9.py", "Procedural: Dynamic Scenario Synthesis & GRPO"),
    "10": ("verify_phase10.py", "Scientific: 4-Way Backbone Ablation Study"),
    "11": ("verify_phase11.py", "Scaling: JED-M2-Medium 273M Architecture"),
    "12": ("verify_phase12.py", "Real Data: Mind2Web Trajectory Ingestion & Scan"),
    "13": ("verify_phase13.py", "Sidecar: Desktop OS & Browser Runtime"),
    "14": ("verify_phase14.py", "Associative Scan: Blelloch Scan & Active JEPA"),
}

QUICK_PHASES = ["1", "2", "4", "6a", "6d"]


def run_phase(phase_key: str, script_dir: str) -> Tuple[bool, float, str]:
    if phase_key not in PHASE_REGISTRY:
        return False, 0.0, f"Unknown phase '{phase_key}'"

    script_name, description = PHASE_REGISTRY[phase_key]
    script_path = os.path.join(script_dir, script_name)

    if not os.path.exists(script_path):
        return False, 0.0, f"Script not found: {script_path}"

    print(f"\n{'=' * 80}")
    print(f"Executing Phase {phase_key}: {description}")
    print(f"Target Script: {script_name}")
    print(f"{'=' * 80}\n")

    start_time = time.time()
    env = os.environ.copy()
    repo_root = os.path.dirname(os.path.abspath(script_dir))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run([sys.executable, script_path], env=env)
    elapsed = time.time() - start_time
    success = (result.returncode == 0)
    return success, elapsed, description


def main():
    parser = argparse.ArgumentParser(description="JED-M2 Verification Suite Runner")
    parser.add_argument("--phase", type=str, default=None, help="Phase key to run (e.g. 1, 2, 6c, 7, 14)")
    parser.add_argument("--list", action="store_true", help="List all available verification phases")
    parser.add_argument("--quick", action="store_true", help="Run fast verification phases (1, 2, 4, 6a, 6d)")
    parser.add_argument("--all", action="store_true", help="Run all verification phases sequentially")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.list:
        print("\nAvailable JED-M2 Verification Phases:")
        print("-" * 80)
        for key, (script, desc) in PHASE_REGISTRY.items():
            print(f"  Phase {key:3s} | {script:20s} | {desc}")
        print("-" * 80)
        return

    phases_to_run: List[str] = []
    if args.phase:
        phases_to_run = [args.phase]
    elif args.quick:
        phases_to_run = QUICK_PHASES
    elif args.all:
        phases_to_run = list(PHASE_REGISTRY.keys())
    else:
        parser.print_help()
        return

    results: List[Tuple[str, bool, float, str]] = []
    for p in phases_to_run:
        success, duration, desc = run_phase(p, script_dir)
        results.append((p, success, duration, desc))

    print("\n" + "=" * 80)
    print("                      VERIFICATION SUMMARY REPORT")
    print("=" * 80)
    print(f" {'Phase':<7} | {'Status':<8} | {'Duration':<10} | {'Description'}")
    print("-" * 80)
    all_passed = True
    for p, success, duration, desc in results:
        status_str = "PASS" if success else "FAIL"
        if not success:
            all_passed = False
        print(f" Phase {p:<3} | {status_str:<8} | {duration:6.2f}s    | {desc}")
    print("=" * 80)
    print(f"Overall Result: {'ALL SUITES PASSED' if all_passed else 'SOME SUITES FAILED'}\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
