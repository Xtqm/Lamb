# JED-M2 Large Action Model - Benchmark Evaluation Report

**Generated:** 2026-11-02 19:10:01 UTC  
**Benchmark Domain:** `real_web_test`  
**Evaluation Device:** `cuda`  

---

### Context & Notes
Evaluated on 10 episodes with max_steps=5 and spatial tolerance epsilon=0.05 on device=cuda.

## 1. Comparative Performance Summary

| Checkpoint | Split | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Native FINISH F1 | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | `test` | 10 | 0.00% | 0.00% | 0.0000 | 1.0000 | 0.5714 | 0.0000 | 0.4190 | 0.0000 | 0.0000 | 25708.57 ms | 155.2 MB |

---

## 2. Generalization Split Breakdown

| Checkpoint | Split / Category | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Action Accuracy | Coord MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | `Travel` | 5 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4264 |
| **jed_m2_sft_base** | `Shopping` | 3 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4442 |
| **jed_m2_sft_base** | `Entertainment` | 2 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3628 |

---

## 3. Action Primitive F1 Breakdown

| Checkpoint | CLICK | TYPE | SCROLL | NAVIGATE | API_CALL | FINISH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |

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