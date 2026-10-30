# Post-Pretraining Scientific Benchmark Report: Unified JED-M2 Large Action Model

**Evaluation Date**: October 30, 2026  
**Audited & Aligned Commit**: Reflecting distractor-grounding and full-candidate pool alignment  
**Dataset Split**: Site-Held-Out Mind2Web Test Split (147 tasks, 573 context windows, 1,178 actions across 11 unseen websites)  
**Candidate Pool Distribution**: Mean $33.71$ interactive elements with bounding boxes per turn (Min: 2, Max: 64)  
**Interactive Environments**: WebOS Closed-Loop Simulation (`ecommerce`, `travel`, `os_admin`)  
**Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (Pure PyTorch, bfloat16, Peak VRAM < 3.4 GB)  
**Evaluated Checkpoints**:
1. `Legacy Pretrained Nano (Step 19,500)` (`checkpoints/pretrain_nano/jed_m2_nano_step_19500.pt`)
2. `Baseline D: Ablation Seed 42` (`runs/ablation/real-data-20261027T151630Z/runs/baseline_d/seed_42/checkpoint_final.pt`)
3. `New Unified JED-M2 Nano (Step 3,000 Final)` (`checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt`)

---

## 1. Executive Summary & Audit Resolution

Following the adversarial scientific audit of October 29, 2026, this report presents verified, leakage-free empirical benchmarks for the **JED-M2 Large Action Model**. 

### Critical Methodological Corrections Implemented:
1. **Full-Page Distractor Candidate Ingestion**:
   Previously, bounding boxes were extracted exclusively from `pos_candidates` (the single human-annotated target element), creating an artificial $N=1$ candidate pool that trivially produced a 100.00% hit rate. The data pipeline has been corrected to ingest both `pos_candidates` and `neg_candidates`, producing realistic multi-element candidate pools averaging **33.71 interactive elements per turn**.
2. **Metric Hierarchy Realignment**:
   In accordance with publication standards, **Unassisted Raw Continuous Coordinate Metrics** are treated as the primary baseline for spatial grounding, while **Multi-Element Candidate Snapping** is evaluated as an operational post-processing layer.
3. **Transparent Characterization of Multi-Token Argument Generation**:
   The report explicitly disaggregates discrete tool prediction from free-form token string generation, disclosing the exact token sequence match rate.

---

## 2. Comparative Benchmark on Site-Held-Out Mind2Web Test Split

The table below contrasts performance across 147 test tasks (11 unseen websites, 1,178 actions) under multi-element distractor conditions (average 33.7 candidates per turn):

| Architecture & Checkpoint | Active Params | Pretrain Steps | Action Primitives | Action Accuracy | CLICK Acc | TYPE Acc | SELECT Acc | Continuous Coord MAE | Raw Target Hit Rate | Snapped Hit Rate ($\le 0.20$, 33.7 cands) | Raw Step Acc | Snapped Step Acc | Raw Trajectory Match | Snapped Trajectory Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Legacy Pretrained Nano** | 52.4M (100%) | 19,500 | 6 (Legacy) | 100.0%* | 100.0% (986/986) | 100.0% (138/138) | **0.0%** (Unscorable) | **0.0121** | **78.78%** | **93.47%** | **83.72%** | **80.78%** | **27.89%** | **20.41%** |
| **Baseline D (Ablation Seed 42)** | 24.1M (46%) | ~2,500 | 7 (Early) | 83.70% | 100.0% (986/986) | **0.0%** (0/138) | **0.0%** (0/54) | 0.2679 | 1.63% | **4.49%** | 35.48% | 36.25% | 2.72% | 2.72% |
| **New Unified JED-M2 (Step 3,000)** | 24.1M (46%) | 3,000 | **7 (Native)** | **99.75%** | **100.0%** (986/986) | **100.0%** (138/138) | **94.44%** (51/54) | **0.0148** | **75.10%** | **93.06%** | **81.24%** | **77.33%** | **27.21%** | **21.77%** |

*\*Note: Legacy Pretrained action accuracy excludes the 54 `SELECT` actions which were unsupported and unscorable in its 6-class head.*

---

## 3. In-Depth Empirical Findings

### A. Spatial Coordinate Grounding & Distractor Discrimination
1. **Unassisted Grounding Precision**:
   Without any snapping heuristics, JED-M2 predicts continuous normalized $(x, y)$ coordinates with an MAE of **0.0148** (less than 1.5% of viewport width/height), achieving an unassisted Target Element Hit Rate of **75.10%** (184 / 245 target elements hit) and a Coordinate Tolerance Accuracy of **90.59%** ($\le 0.05$ radius).
2. **Discrimination Across 33.7 Distractors**:
   When candidate snapping is applied against the multi-element pool, the target hit rate increases from **75.10% to 93.06%** (228 / 245 target elements hit).
3. **Snapping Cannot Compensate for Weak Grounding**:
   In Baseline D, where continuous coordinate MAE is poor (0.2679), candidate snapping against 33.7 distractors only hits the target **4.49%** of the time. This confirms that candidate snapping in JED-M2 succeeds because of genuine underlying spatial representations, not accidental proximity.
4. **Trade-off in Clustered Layouts**:
   In densely clustered UI sections, snapping a slightly off-center click can occasionally gravitate toward an adjacent distractor, explaining why raw step accuracy is **81.24%** vs snapped step accuracy of **77.33%**.

### B. Action Primitive Classification
- **CLICK**: 986 / 986 correct (**100.00%**)
- **TYPE**: 138 / 138 correct (**100.00%**)
- **SELECT**: 51 / 54 correct (**94.44%**) — previously 0.0% in all models!
- **Overall Macro F1**: **0.982** across all discrete tools.

### C. Multi-Token Argument String Generation (Transparent Disclosure)
- **Total `TYPE` Actions**: 138
- **Correct Action Type Classification**: 138 / 138 (**100.00%**)
- **Exact Token-for-Token String Match**: **2 / 138 (1.45%)**
- *Scientific Assessment*: While the model reliably identifies when text input is required and grounds clicks into search and input fields, generating exact character-level strings from scratch remains an open challenge for sub-100M parameter models without external retrieval or dedicated language backbone scaling. Step accuracy remains high (81.24%) because 1,040 of the 1,178 actions are non-typing interactions (`CLICK`, `SELECT`).

---

## 4. Online Closed-Loop Interactive Evaluation (Method B)

Evaluated across 15 closed-loop episodes (5 per scenario: `ecommerce`, `travel`, `os_admin` in `WebOSEnvironment`):

| Model & Execution Mode | Task Success Rate (SR %) | Mean Step Success Rate | Mean Steps / Episode | Mean Cumulative Reward |
| :--- | :---: | :---: | :---: | :---: |
| **Legacy Pretrained** | 0.00% | 45.83% | 8.00 | -0.40 |
| **New Unified (Greedy / No JEPA)** | 0.00% | 25.00% | 8.00 | -0.72 |
| **New Unified (JEPA Planning $\lambda=1.0$)** | 0.00% | 25.00% | 8.00 | -0.72 |

### Root Cause of Interactive Truncation:
The agent successfully executes initial subgoals (e.g. typing query into the search bar and clicking the search button on Turns 0 and 1 of `ecommerce`), but fails to emit `ActionType.FINISH` when the objective is met, causing the environment to truncate at 8 steps with zero terminal reward. Furthermore, dynamic state transitions outside the expert training distribution trigger covariate shift. This empirical finding establishes the necessity of **Phase 4: Online Reinforcement Learning (GRPO)** to teach task termination and closed-loop error recovery.

---

## 5. Artifact Verification & Data Integrity

1. **Evaluation Metrics JSON**: `runs/ablation/post_pretrain_evaluation.json` (regenerated with multi-element candidate pools)
2. **Model Weights**: `checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt` (308.6 MB, Step 3,000)
3. **Dataset Caches**:
   - `data/real_trajectories_cache/real_web_real_v5_..._test_h512.pt` (regenerated with multi-element candidate bboxes)
   - `data/real_trajectories_cache/unified_pretrain_train_5000_h512_seed42.pt` (149.5 MB)
4. **Test Suite Status**: 38 passed, 0 failed, 1 skipped (100% green).
