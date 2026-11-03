# JED-M2 Large Action Model - Benchmark Evaluation Report

**Generated:** 2026-11-03 22:01:50 UTC  
**Benchmark Domain:** `real_web_test`  
**Evaluation Device:** `cuda`  

---

### Context & Notes
Evaluated on 11 episodes with max_steps=5 and spatial tolerance epsilon=0.05 on device=cuda.

## 1. Comparative Performance Summary

| Checkpoint | Split | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Native FINISH F1 | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | `test` | 11 | 0.00% | 0.00% | 0.0000 | 1.0000 | 0.5714 | 0.0000 | 0.4233 | 0.0000 | 0.0000 | 93.51 ms | 457.4 MB |
| **jed_m2_nano_pretrain_final** | `test` | 11 | 0.00% | 0.00% | 0.8455 | 1.0000 | 0.8468 | 0.8654 | 0.3341 | 0.0000 | 0.0000 | 300.03 ms | 1655.6 MB |
| **jed_m2_grpo_step_40** | `test` | 11 | 0.00% | 0.00% | 0.0182 | 0.0000 | 0.4348 | 0.0192 | 0.3992 | 0.0000 | 0.0000 | 523.38 ms | 1655.6 MB |
| **T5-Small (~60.5M)** | `test` | 11 | 0.00% | 0.00% | 0.7545 | 1.0000 | 0.8385 | 0.7692 | 0.2423 | 0.3091 | 0.0000 | 410.57 ms | 295.8 MB |
| **DistilGPT-2 (~81.9M)** | `test` | 11 | 0.00% | 0.00% | 0.8273 | 1.0000 | 0.8452 | 0.8462 | 0.2517 | 0.2727 | 0.0000 | 256.59 ms | 402.6 MB |

---

## 2. Generalization Split Breakdown

| Checkpoint | Split / Category | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Action Accuracy | Coord MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | `Travel` | 5 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4272 |
| **jed_m2_sft_base** | `Shopping` | 4 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4489 |
| **jed_m2_sft_base** | `Entertainment` | 2 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3622 |
| **jed_m2_nano_pretrain_final** | `Travel` | 5 | 0.00% | 0.00% | 0.7800 | 0.7800 | 0.2932 |
| **jed_m2_nano_pretrain_final** | `Shopping` | 4 | 0.00% | 0.00% | 0.8500 | 0.8500 | 0.3791 |
| **jed_m2_nano_pretrain_final** | `Entertainment` | 2 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.3462 |
| **jed_m2_grpo_step_40** | `Travel` | 5 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3985 |
| **jed_m2_grpo_step_40** | `Shopping` | 4 | 0.00% | 0.00% | 0.0500 | 0.0500 | 0.4294 |
| **jed_m2_grpo_step_40** | `Entertainment` | 2 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3403 |
| **T5-Small (~60.5M)** | `Travel` | 5 | 0.00% | 0.00% | 0.7800 | 0.7800 | 0.1679 |
| **T5-Small (~60.5M)** | `Shopping` | 4 | 0.00% | 0.00% | 0.8500 | 0.8500 | 0.2611 |
| **T5-Small (~60.5M)** | `Entertainment` | 2 | 0.00% | 0.00% | 0.5000 | 0.5000 | 0.3907 |
| **DistilGPT-2 (~81.9M)** | `Travel` | 5 | 0.00% | 0.00% | 0.7400 | 0.7400 | 0.1445 |
| **DistilGPT-2 (~81.9M)** | `Shopping` | 4 | 0.00% | 0.00% | 0.8500 | 0.8500 | 0.3387 |
| **DistilGPT-2 (~81.9M)** | `Entertainment` | 2 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.3458 |

---

## 3. Action Primitive F1 Breakdown

| Checkpoint | CLICK | TYPE | SCROLL | NAVIGATE | API_CALL | FINISH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| **jed_m2_nano_pretrain_final** | 0.928 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **jed_m2_grpo_step_40** | 0.043 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| **T5-Small (~60.5M)** | 0.870 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **DistilGPT-2 (~81.9M)** | 0.917 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 4. Metric Definitions & Formulas

1. **Autonomous Success Rate (ASR %)**: Exact task goal completion requiring autonomous `ActionType.FINISH` emission ($ASR = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\text{goal\_satisfied}_i \land \text{action}_i == \text{FINISH}) \times 100$).
2. **Referee-Assisted Rate (RAR %)**: Milestone satisfaction verified by completion referee regardless of terminal action emission.
3. **Native FINISH F1**: Harmonic mean of precision and recall for `ActionType.FINISH` primitives.
4. **Generalization Boundaries**: Evaluated across Cross-Task (seen websites), Cross-Website (unseen websites, seen domain), and Cross-Domain (unseen domain).
4. **Step Success Rate**: Ratio of valid intermediate sub-goals achieved across all execution turns.
5. **Action Primitive Macro F1**: Unweighted harmonic mean of precision and recall across discrete tools.
6. **Spatial Coordinate MAE**: Mean absolute error $\frac{1}{2}(|\hat{x} - x^*| + |\hat{y} - y^*|)$ in normalized $[0, 1]^2$.
7. **Coordinate Tolerance Accuracy**: Fraction of actions with Euclidean distance $\le 0.05$ or falling within element bbox.
8. **Trajectory Efficiency Ratio**: Ratio of optimal demonstration steps to agent steps ($L_{\text{optimal}} / L_{\text{agent}}$).
9. **Decision Latency**: Average milliseconds elapsed per decision step including rationale and action emission.