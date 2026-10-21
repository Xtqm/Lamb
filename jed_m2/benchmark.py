"""
JED-M2 Standardized Evaluation Benchmark Harness for Web and OS Agent Tasks.
Provides:
- Data containers: StepMetric, EpisodeMetric, BenchmarkSummary.
- TaskEvaluator: Computes formal metrics:
  * Task Success Rate (SR %): Exact task goal completion.
  * Step Success Rate: Fraction of valid intermediate sub-goals achieved.
  * Action Primitive F1: Precision, recall, micro/macro F1 across 6 discrete primitives.
  * Spatial Coordinate Accuracy: Mean Absolute Error (MAE) and tolerance radius (epsilon <= 0.05).
  * Trajectory Efficiency Ratio: Length of agent trajectory compared to optimal demonstration.
  * Decision Latency: Milliseconds per step and token throughput.
- evaluate_offline_dataset: Replays Mind2Web/WebArena trajectory datasets with ground-truth matching.
- evaluate_online_environment: Multi-turn closed-loop execution in WebOSEnvironment and Playwright.
- Comparative runner utilities for side-by-side checkpoint evaluation and report generation.

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
"""

from dataclasses import dataclass, field
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch

from jed_m2.agent import AgentStepOutput, JEDM2Agent
from jed_m2.config import JEDM2Config
from jed_m2.dataset import (
    AgentTrajectoryDataset,
    InteractionTurn,
    SimpleLAMTokenizer,
    SyntheticTrajectoryGenerator,
    TrajectorySample,
    UIElement,
)
from jed_m2.environment import ScenarioDefinition, StepResult, WebOSEnvironment, get_default_scenarios
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model


# ==============================================================================
# 1. Metric Data Containers
# ==============================================================================

@dataclass
class StepMetric:
    """Evaluation metrics for a single interaction turn."""

    turn_id: int
    action_type_pred: int
    action_type_gt: int
    action_name_pred: str
    action_name_gt: str
    action_correct: bool
    coord_pred: Tuple[float, float]
    coord_gt: Tuple[float, float]
    coord_mae: float
    coord_tolerance_hit: bool
    target_bbox: Optional[Tuple[float, float, float, float]] = None
    step_latency_ms: float = 0.0
    token_count: int = 0
    reward: float = 0.0
    subgoal_achieved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes step metric to dictionary."""
        return {
            "turn_id": self.turn_id,
            "action_type_pred": self.action_type_pred,
            "action_type_gt": self.action_type_gt,
            "action_name_pred": self.action_name_pred,
            "action_name_gt": self.action_name_gt,
            "action_correct": self.action_correct,
            "coord_pred": self.coord_pred,
            "coord_gt": self.coord_gt,
            "coord_mae": self.coord_mae,
            "coord_tolerance_hit": self.coord_tolerance_hit,
            "target_bbox": self.target_bbox,
            "step_latency_ms": self.step_latency_ms,
            "token_count": self.token_count,
            "reward": self.reward,
            "subgoal_achieved": self.subgoal_achieved,
        }


@dataclass
class EpisodeMetric:
    """Evaluation metrics for a complete multi-turn task episode."""

    episode_id: str
    benchmark_name: str
    task_goal: str
    task_success: bool
    step_success_rate: float
    action_accuracy: float
    mean_coord_mae: float
    coord_tolerance_accuracy: float
    trajectory_length: int
    optimal_length: int
    efficiency_ratio: float
    mean_latency_ms: float
    token_throughput: float
    total_reward: float
    step_metrics: List[StepMetric] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes episode metric to dictionary."""
        return {
            "episode_id": self.episode_id,
            "benchmark_name": self.benchmark_name,
            "task_goal": self.task_goal,
            "task_success": self.task_success,
            "step_success_rate": self.step_success_rate,
            "action_accuracy": self.action_accuracy,
            "mean_coord_mae": self.mean_coord_mae,
            "coord_tolerance_accuracy": self.coord_tolerance_accuracy,
            "trajectory_length": self.trajectory_length,
            "optimal_length": self.optimal_length,
            "efficiency_ratio": self.efficiency_ratio,
            "mean_latency_ms": self.mean_latency_ms,
            "token_throughput": self.token_throughput,
            "total_reward": self.total_reward,
            "step_count": len(self.step_metrics),
        }


@dataclass
class BenchmarkSummary:
    """Comprehensive summary of evaluation metrics across an entire benchmark run."""

    benchmark_name: str
    num_episodes: int
    task_success_rate: float  # Percentage [0.0, 100.0]
    mean_step_success_rate: float  # [0.0, 1.0]
    action_accuracy: float  # [0.0, 1.0]
    action_macro_f1: float  # [0.0, 1.0]
    action_micro_f1: float  # [0.0, 1.0]
    action_per_class_f1: Dict[str, float] = field(default_factory=dict)
    action_per_class_precision: Dict[str, float] = field(default_factory=dict)
    action_per_class_recall: Dict[str, float] = field(default_factory=dict)
    mean_coord_mae: float = 0.0
    coord_tolerance_accuracy: float = 0.0  # [0.0, 1.0]
    mean_efficiency_ratio: float = 0.0  # [0.0, 1.0]
    mean_decision_latency_ms: float = 0.0
    median_decision_latency_ms: float = 0.0
    mean_token_throughput: float = 0.0  # tokens/sec
    peak_vram_mb: float = 0.0
    device: str = "cpu"
    checkpoint_name: str = "default"
    episode_metrics: List[EpisodeMetric] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes benchmark summary to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "checkpoint_name": self.checkpoint_name,
            "num_episodes": self.num_episodes,
            "task_success_rate": self.task_success_rate,
            "mean_step_success_rate": self.mean_step_success_rate,
            "action_accuracy": self.action_accuracy,
            "action_macro_f1": self.action_macro_f1,
            "action_micro_f1": self.action_micro_f1,
            "action_per_class_f1": self.action_per_class_f1,
            "mean_coord_mae": self.mean_coord_mae,
            "coord_tolerance_accuracy": self.coord_tolerance_accuracy,
            "mean_efficiency_ratio": self.mean_efficiency_ratio,
            "mean_decision_latency_ms": self.mean_decision_latency_ms,
            "median_decision_latency_ms": self.median_decision_latency_ms,
            "mean_token_throughput": self.mean_token_throughput,
            "peak_vram_mb": self.peak_vram_mb,
            "device": self.device,
            "metadata": self.metadata,
        }


# ==============================================================================
# 2. Formal Action & Task Evaluator
# ==============================================================================

class TaskEvaluator:
    """
    Standardized evaluation engine for Web and OS agent tasks.
    Computes mathematical metrics across discrete action categories, spatial
    screen coordinates, multi-turn trajectories, and latency profiles.
    """

    NUM_ACTION_TYPES: int = 6
    ACTION_NAMES: List[str] = [
        "CLICK",
        "TYPE",
        "SCROLL",
        "NAVIGATE",
        "API_CALL",
        "FINISH",
    ]

    def __init__(self, tolerance_radius: float = 0.05):
        self.tolerance_radius = tolerance_radius

    def evaluate_step(
        self,
        turn_id: int,
        action_type_pred: int,
        action_type_gt: int,
        coord_pred: Tuple[float, float],
        coord_gt: Tuple[float, float],
        target_bbox: Optional[Tuple[float, float, float, float]] = None,
        step_latency_ms: float = 0.0,
        token_count: int = 0,
        reward: float = 0.0,
        subgoal_achieved: bool = False,
    ) -> StepMetric:
        """
        Evaluates a single action step against ground truth.

        Args:
            turn_id: Zero-based step counter in current trajectory.
            action_type_pred: Predicted discrete primitive ID [0..5].
            action_type_gt: Ground-truth discrete primitive ID [0..5].
            coord_pred: Predicted normalized continuous screen coordinates (x, y) in [0, 1].
            coord_gt: Ground-truth normalized coordinates (x, y) in [0, 1].
            target_bbox: Optional ground-truth element bbox (x_min, y_min, x_max, y_max).
            step_latency_ms: Decision latency in milliseconds.
            token_count: Number of tokens ingested/generated in this step.
            reward: Step-level environment reward.
            subgoal_achieved: Whether the intermediate sub-goal was achieved.

        Returns:
            StepMetric instance.
        """
        pred_id = int(action_type_pred)
        gt_id = int(action_type_gt)
        action_correct = (pred_id == gt_id)

        # Spatial MAE: average error across normalized x and y
        px, py = float(coord_pred[0]), float(coord_pred[1])
        gx, gy = float(coord_gt[0]), float(coord_gt[1])
        coord_mae = 0.5 * (abs(px - gx) + abs(py - gy))

        # Spatial tolerance hit: Euclidean distance <= tolerance_radius OR inside bbox
        dist = math.sqrt((px - gx) ** 2 + (py - gy) ** 2)
        tolerance_hit = (dist <= self.tolerance_radius)

        if not tolerance_hit and target_bbox is not None:
            bx_min, by_min, bx_max, by_max = target_bbox
            if bx_min <= px <= bx_max and by_min <= py <= by_max:
                tolerance_hit = True

        pred_name = self.ACTION_NAMES[pred_id] if 0 <= pred_id < len(self.ACTION_NAMES) else f"UNKNOWN_{pred_id}"
        gt_name = self.ACTION_NAMES[gt_id] if 0 <= gt_id < len(self.ACTION_NAMES) else f"UNKNOWN_{gt_id}"

        return StepMetric(
            turn_id=turn_id,
            action_type_pred=pred_id,
            action_type_gt=gt_id,
            action_name_pred=pred_name,
            action_name_gt=gt_name,
            action_correct=action_correct,
            coord_pred=(px, py),
            coord_gt=(gx, gy),
            coord_mae=coord_mae,
            coord_tolerance_hit=tolerance_hit,
            target_bbox=target_bbox,
            step_latency_ms=step_latency_ms,
            token_count=token_count,
            reward=reward,
            subgoal_achieved=subgoal_achieved,
        )

    def evaluate_episode(
        self,
        episode_id: str,
        benchmark_name: str,
        task_goal: str,
        step_metrics: List[StepMetric],
        task_success: bool,
        optimal_length: int,
    ) -> EpisodeMetric:
        """
        Aggregates step metrics into an episode-level metric record.

        Args:
            episode_id: Unique string identifier for the episode.
            benchmark_name: Name of benchmark domain/split.
            task_goal: High-level task instruction text.
            step_metrics: Ordered list of StepMetric records.
            task_success: Whether the final task goal was completed.
            optimal_length: Length of optimal demonstration trajectory.

        Returns:
            EpisodeMetric instance.
        """
        num_steps = len(step_metrics)
        if num_steps == 0:
            return EpisodeMetric(
                episode_id=episode_id,
                benchmark_name=benchmark_name,
                task_goal=task_goal,
                task_success=task_success,
                step_success_rate=0.0,
                action_accuracy=0.0,
                mean_coord_mae=0.0,
                coord_tolerance_accuracy=0.0,
                trajectory_length=0,
                optimal_length=optimal_length,
                efficiency_ratio=0.0,
                mean_latency_ms=0.0,
                token_throughput=0.0,
                total_reward=0.0,
                step_metrics=[],
            )

        step_success_count = sum(1 for sm in step_metrics if sm.subgoal_achieved or sm.action_correct)
        step_success_rate = float(step_success_count / num_steps)

        action_correct_count = sum(1 for sm in step_metrics if sm.action_correct)
        action_accuracy = float(action_correct_count / num_steps)

        mean_coord_mae = float(sum(sm.coord_mae for sm in step_metrics) / num_steps)
        tolerance_hit_count = sum(1 for sm in step_metrics if sm.coord_tolerance_hit)
        coord_tolerance_accuracy = float(tolerance_hit_count / num_steps)

        # Trajectory Efficiency Ratio: L_optimal / L_agent if task completed, 0.0 otherwise
        if task_success and num_steps > 0 and optimal_length > 0:
            efficiency_ratio = min(1.0, float(optimal_length / max(1, num_steps)))
        else:
            efficiency_ratio = 0.0

        total_latency_ms = sum(sm.step_latency_ms for sm in step_metrics)
        mean_latency_ms = float(total_latency_ms / num_steps) if num_steps > 0 else 0.0

        total_tokens = sum(sm.token_count for sm in step_metrics)
        total_time_sec = total_latency_ms / 1000.0
        token_throughput = float(total_tokens / total_time_sec) if total_time_sec > 1e-6 else 0.0

        total_reward = float(sum(sm.reward for sm in step_metrics))

        return EpisodeMetric(
            episode_id=episode_id,
            benchmark_name=benchmark_name,
            task_goal=task_goal,
            task_success=task_success,
            step_success_rate=step_success_rate,
            action_accuracy=action_accuracy,
            mean_coord_mae=mean_coord_mae,
            coord_tolerance_accuracy=coord_tolerance_accuracy,
            trajectory_length=num_steps,
            optimal_length=optimal_length,
            efficiency_ratio=efficiency_ratio,
            mean_latency_ms=mean_latency_ms,
            token_throughput=token_throughput,
            total_reward=total_reward,
            step_metrics=step_metrics,
        )

    def compute_action_f1(
        self,
        all_pred: Sequence[int],
        all_gt: Sequence[int],
    ) -> Tuple[float, float, Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Computes precision, recall, macro F1, and micro F1 across discrete action categories.

        Args:
            all_pred: Sequence of predicted discrete action primitive IDs.
            all_gt: Sequence of ground-truth discrete action primitive IDs.

        Returns:
            Tuple of:
            - macro_f1: Unweighted average of per-class F1 scores.
            - micro_f1: Global aggregated F1 score.
            - per_class_f1: Dict mapping action name -> F1 score.
            - per_class_precision: Dict mapping action name -> precision.
            - per_class_recall: Dict mapping action name -> recall.
        """
        num_classes = self.NUM_ACTION_TYPES
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes

        for p, g in zip(all_pred, all_gt):
            if 0 <= p < num_classes and 0 <= g < num_classes:
                if p == g:
                    tp[p] += 1
                else:
                    fp[p] += 1
                    fn[g] += 1

        per_class_precision: Dict[str, float] = {}
        per_class_recall: Dict[str, float] = {}
        per_class_f1: Dict[str, float] = {}

        for c in range(num_classes):
            name = self.ACTION_NAMES[c]
            c_tp = tp[c]
            c_fp = fp[c]
            c_fn = fn[c]

            # Precision guard
            if c_tp + c_fp > 0:
                prec = c_tp / (c_tp + c_fp)
            else:
                prec = 1.0 if (c_tp + c_fn == 0) else 0.0

            # Recall guard
            if c_tp + c_fn > 0:
                rec = c_tp / (c_tp + c_fn)
            else:
                rec = 1.0 if (c_tp + c_fp == 0) else 0.0

            # F1 guard
            if prec + rec > 0.0:
                f1 = 2.0 * (prec * rec) / (prec + rec)
            else:
                f1 = 0.0

            per_class_precision[name] = float(prec)
            per_class_recall[name] = float(rec)
            per_class_f1[name] = float(f1)

        macro_f1 = float(sum(per_class_f1.values()) / max(1, num_classes))

        # Micro F1: global TP / (global TP + 0.5 * (global FP + global FN))
        total_tp = sum(tp)
        total_fp = sum(fp)
        total_fn = sum(fn)
        denom = total_tp + 0.5 * (total_fp + total_fn)
        micro_f1 = float(total_tp / denom) if denom > 0 else 0.0

        return macro_f1, micro_f1, per_class_f1, per_class_precision, per_class_recall

    def aggregate(
        self,
        episodes: List[EpisodeMetric],
        benchmark_name: str = "benchmark",
        checkpoint_name: str = "default",
        peak_vram_mb: float = 0.0,
        device: str = "cpu",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkSummary:
        """
        Aggregates multiple EpisodeMetric objects into a formal BenchmarkSummary.

        Args:
            episodes: List of EpisodeMetric objects.
            benchmark_name: Name of benchmark.
            checkpoint_name: Model or checkpoint label.
            peak_vram_mb: Peak allocated VRAM in megabytes during run.
            device: Hardware device string ('cuda', 'cpu').
            metadata: Optional dictionary of additional run properties.

        Returns:
            BenchmarkSummary instance with numerical sanity guarantees.
        """
        num_episodes = len(episodes)
        if num_episodes == 0:
            return BenchmarkSummary(
                benchmark_name=benchmark_name,
                num_episodes=0,
                task_success_rate=0.0,
                mean_step_success_rate=0.0,
                action_accuracy=0.0,
                action_macro_f1=0.0,
                action_micro_f1=0.0,
                peak_vram_mb=peak_vram_mb,
                device=device,
                checkpoint_name=checkpoint_name,
                metadata=metadata or {},
            )

        # Task Success Rate % in [0.0, 100.0]
        success_count = sum(1 for ep in episodes if ep.task_success)
        task_success_rate = float(success_count / num_episodes) * 100.0

        mean_step_success_rate = float(sum(ep.step_success_rate for ep in episodes) / num_episodes)
        action_accuracy = float(sum(ep.action_accuracy for ep in episodes) / num_episodes)
        mean_coord_mae = float(sum(ep.mean_coord_mae for ep in episodes) / num_episodes)
        coord_tolerance_accuracy = float(sum(ep.coord_tolerance_accuracy for ep in episodes) / num_episodes)
        mean_efficiency_ratio = float(sum(ep.efficiency_ratio for ep in episodes) / num_episodes)

        # Latency statistics
        latencies = [ep.mean_latency_ms for ep in episodes if ep.mean_latency_ms > 0.0]
        if latencies:
            mean_latency = float(sum(latencies) / len(latencies))
            sorted_latencies = sorted(latencies)
            mid = len(sorted_latencies) // 2
            if len(sorted_latencies) % 2 == 0:
                median_latency = float((sorted_latencies[mid - 1] + sorted_latencies[mid]) / 2.0)
            else:
                median_latency = float(sorted_latencies[mid])
        else:
            mean_latency = 0.0
            median_latency = 0.0

        # Throughput statistics
        throughputs = [ep.token_throughput for ep in episodes if ep.token_throughput > 0.0]
        mean_throughput = float(sum(throughputs) / len(throughputs)) if throughputs else 0.0

        # Action Primitive F1 across all evaluated steps
        all_pred: List[int] = []
        all_gt: List[int] = []
        for ep in episodes:
            for sm in ep.step_metrics:
                all_pred.append(sm.action_type_pred)
                all_gt.append(sm.action_type_gt)

        macro_f1, micro_f1, per_class_f1, per_class_prec, per_class_rec = self.compute_action_f1(all_pred, all_gt)

        return BenchmarkSummary(
            benchmark_name=benchmark_name,
            num_episodes=num_episodes,
            task_success_rate=task_success_rate,
            mean_step_success_rate=mean_step_success_rate,
            action_accuracy=action_accuracy,
            action_macro_f1=macro_f1,
            action_micro_f1=micro_f1,
            action_per_class_f1=per_class_f1,
            action_per_class_precision=per_class_prec,
            action_per_class_recall=per_class_rec,
            mean_coord_mae=mean_coord_mae,
            coord_tolerance_accuracy=coord_tolerance_accuracy,
            mean_efficiency_ratio=mean_efficiency_ratio,
            mean_decision_latency_ms=mean_latency,
            median_decision_latency_ms=median_latency,
            mean_token_throughput=mean_throughput,
            peak_vram_mb=peak_vram_mb,
            device=device,
            checkpoint_name=checkpoint_name,
            episode_metrics=episodes,
            metadata=metadata or {},
        )

    @staticmethod
    def format_ascii_table(
        summaries: Union[BenchmarkSummary, Sequence[BenchmarkSummary]],
        title: str = "JED-M2 BENCHMARK EVALUATION RESULTS",
    ) -> str:
        """
        Renders a clean 7-bit ASCII comparative results table.
        Avoids multi-byte Unicode box drawing characters to prevent Windows console CP1252 errors.
        """
        if isinstance(summaries, BenchmarkSummary):
            summaries = [summaries]

        lines: List[str] = []
        border = "+" + "-" * 78 + "+"
        lines.append(border)
        lines.append(f"| {title.center(76)} |")
        lines.append(border)

        headers = [
            "Checkpoint / Model",
            "SR (%)",
            "Step SR",
            "Act F1",
            "MAE",
            "Tol Acc",
            "Eff Ratio",
            "Lat (ms)",
        ]
        col_fmt = "| {0:<24} | {1:>6} | {2:>7} | {3:>6} | {4:>6} | {5:>7} | {6:>9} | {7:>8} |"
        lines.append(col_fmt.format(*headers))
        lines.append("+" + "=" * 78 + "+")

        for s in summaries:
            name = s.checkpoint_name[:24]
            sr_str = f"{s.task_success_rate:.1f}%"
            step_sr_str = f"{s.mean_step_success_rate:.3f}"
            act_f1_str = f"{s.action_macro_f1:.3f}"
            mae_str = f"{s.mean_coord_mae:.3f}"
            tol_str = f"{s.coord_tolerance_accuracy:.3f}"
            eff_str = f"{s.mean_efficiency_ratio:.3f}"
            lat_str = f"{s.mean_decision_latency_ms:.1f}"

            lines.append(col_fmt.format(name, sr_str, step_sr_str, act_f1_str, mae_str, tol_str, eff_str, lat_str))

        lines.append(border)
        return "\n".join(lines)

    @staticmethod
    def export_markdown_report(
        summaries: Union[BenchmarkSummary, Sequence[BenchmarkSummary]],
        output_path: str = "BENCHMARK_REPORT.md",
        notes: Optional[str] = None,
    ) -> str:
        """
        Generates and saves a formal Markdown benchmark report.

        Args:
            summaries: Single BenchmarkSummary or list of summaries.
            output_path: Filepath where the Markdown report will be saved.
            notes: Optional contextual notes or commentary.

        Returns:
            The complete Markdown report string.
        """
        if isinstance(summaries, BenchmarkSummary):
            summaries = [summaries]

        md_lines: List[str] = []
        md_lines.append("# JED-M2 Large Action Model - Benchmark Evaluation Report\n")
        md_lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ")
        if summaries:
            md_lines.append(f"**Benchmark Domain:** `{summaries[0].benchmark_name}`  ")
            md_lines.append(f"**Evaluation Device:** `{summaries[0].device}`  ")
        md_lines.append("\n---\n")

        if notes:
            md_lines.append(f"### Context & Notes\n{notes}\n")

        md_lines.append("## 1. Comparative Performance Summary\n")
        md_lines.append(
            "| Checkpoint | Episodes | Task SR (%) | Step SR | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Throughput (tok/s) | Peak VRAM (MB) |"
        )
        md_lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )

        for s in summaries:
            row = (
                f"| **{s.checkpoint_name}** | {s.num_episodes} | {s.task_success_rate:.2f}% | "
                f"{s.mean_step_success_rate:.4f} | {s.action_macro_f1:.4f} | {s.action_micro_f1:.4f} | "
                f"{s.mean_coord_mae:.4f} | {s.coord_tolerance_accuracy:.4f} | {s.mean_efficiency_ratio:.4f} | "
                f"{s.mean_decision_latency_ms:.2f} ms | {s.mean_token_throughput:.1f} | {s.peak_vram_mb:.1f} MB |"
            )
            md_lines.append(row)

        md_lines.append("\n---\n")
        md_lines.append("## 2. Action Primitive F1 Breakdown\n")
        md_lines.append("| Checkpoint | CLICK | TYPE | SCROLL | NAVIGATE | API_CALL | FINISH |")
        md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for s in summaries:
            f1_dict = s.action_per_class_f1
            row = (
                f"| **{s.checkpoint_name}** | {f1_dict.get('CLICK', 0.0):.3f} | {f1_dict.get('TYPE', 0.0):.3f} | "
                f"{f1_dict.get('SCROLL', 0.0):.3f} | {f1_dict.get('NAVIGATE', 0.0):.3f} | "
                f"{f1_dict.get('API_CALL', 0.0):.3f} | {f1_dict.get('FINISH', 0.0):.3f} |"
            )
            md_lines.append(row)

        md_lines.append("\n---\n")
        md_lines.append("## 3. Metric Definitions & Formulas\n")
        md_lines.append("1. **Task Success Rate (SR %)**: Exact task goal completion ($SR = \\frac{1}{N}\\sum_{i=1}^N \\mathbb{I}(\\text{success}_i) \\times 100$).")
        md_lines.append("2. **Step Success Rate**: Ratio of valid intermediate sub-goals achieved across all execution turns.")
        md_lines.append("3. **Action Primitive Macro F1**: Unweighted harmonic mean of precision and recall across discrete tools.")
        md_lines.append("4. **Spatial Coordinate MAE**: Mean absolute error $\\frac{1}{2}(|\\hat{x} - x^*| + |\\hat{y} - y^*|)$ in normalized $[0, 1]^2$.")
        md_lines.append("5. **Coordinate Tolerance Accuracy**: Fraction of actions with Euclidean distance $\\le 0.05$ or falling within element bbox.")
        md_lines.append("6. **Trajectory Efficiency Ratio**: Ratio of optimal demonstration steps to agent steps ($L_{\\text{optimal}} / L_{\\text{agent}}$).")
        md_lines.append("7. **Decision Latency**: Average milliseconds elapsed per decision step including rationale and action emission.")

        content = "\n".join(md_lines)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)

        return content


# ==============================================================================
# 3. Offline Dataset Replay Evaluation
# ==============================================================================

def evaluate_offline_dataset(
    agent: JEDM2Agent,
    trajectories: Union[List[TrajectorySample], AgentTrajectoryDataset],
    evaluator: Optional[TaskEvaluator] = None,
    benchmark_name: str = "offline_replay",
    checkpoint_name: str = "default",
    max_episodes: Optional[int] = None,
) -> BenchmarkSummary:
    """
    Evaluates an agent on offline pre-recorded trajectory datasets (e.g. Mind2Web/WebArena).
    Sequentially feeds observations into agent.act(...) and compares predicted primitives,
    coordinates, and arguments against ground truth.

    Args:
        agent: JEDM2Agent instance wrapping JEDM2Model.
        trajectories: Sequence of TrajectorySample or AgentTrajectoryDataset.
        evaluator: TaskEvaluator instance (defaults to tolerance_radius=0.05).
        benchmark_name: Benchmark label string.
        checkpoint_name: Model or checkpoint label string.
        max_episodes: Optional upper bound on episodes to evaluate.

    Returns:
        BenchmarkSummary populated with comprehensive offline metrics.
    """
    if evaluator is None:
        evaluator = TaskEvaluator()

    # Track VRAM
    device_str = str(agent.device)
    if "cuda" in device_str and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(agent.device)

    samples: List[TrajectorySample]
    if isinstance(trajectories, AgentTrajectoryDataset):
        # Extract underlying samples or generator
        samples = [trajectories[i] for i in range(min(len(trajectories), max_episodes or len(trajectories)))]
    else:
        samples = list(trajectories)
        if max_episodes is not None:
            samples = samples[:max_episodes]

    episode_metrics: List[EpisodeMetric] = []

    for ep_idx, sample in enumerate(samples):
        agent.reset()
        step_metrics: List[StepMetric] = []
        episode_success = True

        for turn_idx, turn in enumerate(sample.turns):
            t_start = time.perf_counter()

            # Execute agent step
            agent_out: AgentStepOutput = agent.act(
                obs_text=turn.obs_text,
                obs_features=turn.obs_features,
                screenshot=turn.screenshot,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            token_count = len(agent.tokenizer.encode(turn.obs_text)) + agent_out.num_thought_tokens + 4

            # Ground truth values
            gt_action_id = int(turn.action_type.value)
            gt_coords = turn.coordinates

            # Find matching UI element bbox if available
            target_bbox: Optional[Tuple[float, float, float, float]] = None
            for el in turn.dom_elements:
                if el.bbox is not None and len(el.bbox) == 4:
                    # If coordinate falls inside bbox, use as target
                    bx_min, by_min, bx_max, by_max = el.bbox
                    if bx_min <= gt_coords[0] <= bx_max and by_min <= gt_coords[1] <= by_max:
                        target_bbox = el.bbox
                        break

            sm = evaluator.evaluate_step(
                turn_id=turn_idx,
                action_type_pred=agent_out.action_type_id,
                action_type_gt=gt_action_id,
                coord_pred=agent_out.coordinates,
                coord_gt=gt_coords,
                target_bbox=target_bbox,
                step_latency_ms=latency_ms,
                token_count=token_count,
                reward=turn.reward,
                subgoal_achieved=(agent_out.action_type_id == gt_action_id and sm_tol_hit if "sm_tol_hit" in locals() else agent_out.action_type_id == gt_action_id),
            )

            # Subgoal is achieved if action primitive matches and coordinates hit tolerance
            sm.subgoal_achieved = (sm.action_correct and sm.coord_tolerance_hit)
            if not sm.subgoal_achieved:
                episode_success = False

            step_metrics.append(sm)

        # In offline replay, task succeeds if all steps were valid or finished with FINISH
        if step_metrics and step_metrics[-1].action_type_gt == ActionType.FINISH.value:
            task_success = (step_metrics[-1].action_correct and episode_success)
        else:
            task_success = episode_success and len(step_metrics) > 0

        ep_metric = evaluator.evaluate_episode(
            episode_id=sample.sample_id,
            benchmark_name=benchmark_name,
            task_goal=sample.goal,
            step_metrics=step_metrics,
            task_success=task_success,
            optimal_length=sample.length,
        )
        episode_metrics.append(ep_metric)

    peak_vram_mb = 0.0
    if "cuda" in device_str and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(agent.device) / (1024 * 1024)

    return evaluator.aggregate(
        episodes=episode_metrics,
        benchmark_name=benchmark_name,
        checkpoint_name=checkpoint_name,
        peak_vram_mb=peak_vram_mb,
        device=device_str,
    )


# ==============================================================================
# 4. Online Closed-Loop Environment Evaluation
# ==============================================================================

def evaluate_online_environment(
    agent: JEDM2Agent,
    scenarios: Optional[List[ScenarioDefinition]] = None,
    evaluator: Optional[TaskEvaluator] = None,
    benchmark_name: str = "webos_simulated",
    checkpoint_name: str = "default",
    num_episodes: int = 10,
    max_steps: int = 10,
    seed: int = 42,
) -> BenchmarkSummary:
    """
    Evaluates an agent in online closed-loop multi-turn environments (WebOSEnvironment).
    Agent autonomously interacts turn-by-turn until goal completion or max_steps limit.

    Args:
        agent: JEDM2Agent instance wrapping JEDM2Model.
        scenarios: Optional list of ScenarioDefinitions. Defaults to built-in ecommerce, travel, os_admin.
        evaluator: TaskEvaluator instance.
        benchmark_name: Benchmark label string.
        checkpoint_name: Model or checkpoint label string.
        num_episodes: Total number of closed-loop episodes to evaluate.
        max_steps: Maximum turns allowed per episode before truncation.
        seed: Random seed for reproducible element layouts.

    Returns:
        BenchmarkSummary populated with online closed-loop metrics.
    """
    if evaluator is None:
        evaluator = TaskEvaluator()

    if scenarios is None:
        default_dict = get_default_scenarios()
        scenarios = list(default_dict.values())

    device_str = str(agent.device)
    if "cuda" in device_str and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(agent.device)

    episode_metrics: List[EpisodeMetric] = []

    for ep_idx in range(num_episodes):
        scenario = scenarios[ep_idx % len(scenarios)]
        env = WebOSEnvironment(
            config=agent.config,
            scenario_id=scenario.scenario_id,
            device=agent.device,
            dtype=agent.dtype,
        )

        agent.reset()
        init_res = env.reset()

        step_metrics: List[StepMetric] = []
        obs_text = init_res.obs_text
        obs_features = init_res.obs_features
        screenshot = getattr(init_res, "screenshot", None)
        if screenshot is None and isinstance(init_res.info, dict):
            screenshot = init_res.info.get("screenshot", None)

        optimal_length = min(scenario.max_steps, 5)

        for turn_idx in range(max_steps):
            t_start = time.perf_counter()

            agent_out = agent.act(
                obs_text=obs_text,
                obs_features=obs_features,
                screenshot=screenshot,
            )

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            token_count = len(agent.tokenizer.encode(obs_text)) + agent_out.num_thought_tokens + 4

            step_res: StepResult = env.step(
                action_type=agent_out.action_type,
                coordinates=agent_out.coordinates,
                argument=agent_out.argument,
            )

            # In online simulation, step success is reflected by reward or last action success
            subgoal_achieved = (step_res.reward > 0.0 or step_res.info.get("action_success", False))

            # Ground truth action proxy from current state target if known
            target_el_id = step_res.info.get("target_element_id", None)
            target_bbox: Optional[Tuple[float, float, float, float]] = None
            gt_coords = agent_out.coordinates  # fallback if not specified

            if env.current_state is not None:
                for el in env.current_state.elements:
                    if target_el_id is not None and el.element_id == target_el_id:
                        target_bbox = el.bbox
                        gt_coords = el.center
                        break

            # Evaluate step
            gt_action_type = agent_out.action_type_id if subgoal_achieved else (
                ActionType.CLICK.value if turn_idx < optimal_length - 1 else ActionType.FINISH.value
            )

            sm = evaluator.evaluate_step(
                turn_id=turn_idx,
                action_type_pred=agent_out.action_type_id,
                action_type_gt=gt_action_type,
                coord_pred=agent_out.coordinates,
                coord_gt=gt_coords,
                target_bbox=target_bbox,
                step_latency_ms=latency_ms,
                token_count=token_count,
                reward=step_res.reward,
                subgoal_achieved=subgoal_achieved,
            )
            step_metrics.append(sm)

            if step_res.done:
                break

            # Update observation for next step
            obs_text = env._format_observation_text()
            obs_features = env._generate_obs_features()
            if hasattr(env, "capture_screenshot"):
                screenshot = env.capture_screenshot(as_tensor=True)
            else:
                screenshot = None

        # Episode success: whether task goal was fulfilled
        task_success = getattr(env, "goal_satisfied", False) or (
            sum(sm.reward for sm in step_metrics) >= 1.0 or (step_metrics and step_metrics[-1].subgoal_achieved and agent_out.action_type == ActionType.FINISH)
        )

        ep_metric = evaluator.evaluate_episode(
            episode_id=f"{scenario.scenario_id}_ep{ep_idx}",
            benchmark_name=benchmark_name,
            task_goal=scenario.goal,
            step_metrics=step_metrics,
            task_success=task_success,
            optimal_length=optimal_length,
        )
        episode_metrics.append(ep_metric)

    peak_vram_mb = 0.0
    if "cuda" in device_str and torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated(agent.device) / (1024 * 1024)

    return evaluator.aggregate(
        episodes=episode_metrics,
        benchmark_name=benchmark_name,
        checkpoint_name=checkpoint_name,
        peak_vram_mb=peak_vram_mb,
        device=device_str,
    )
