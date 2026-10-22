# JED-M2 Large Action Model - Benchmark Evaluation Report

**Generated:** 2026-10-22 00:33:46 UTC  
**Benchmark Domain:** `mind2web_test`  
**Evaluation Device:** `cuda`  

---

### Context & Notes
Evaluated on 20 episodes with max_steps=15 and spatial tolerance epsilon=0.05 on device=cuda.

## 1. Comparative Performance Summary

| Checkpoint | Episodes | Task SR (%) | Step SR | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Throughput (tok/s) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **checkpoint_step_300** | 20 | 0.00% | 0.5650 | 0.5363 | 0.5614 | 0.3015 | 0.0683 | 0.0000 | 760.75 ms | 74.7 | 135.3 MB |
| **jed_m2_grpo_step_50** | 20 | 0.00% | 0.5717 | 0.5314 | 0.5702 | 0.3016 | 0.0767 | 0.0000 | 737.88 ms | 76.9 | 135.3 MB |
| **jed_m2_grpo_step_200** | 20 | 0.00% | 0.5350 | 0.5169 | 0.5351 | 0.3015 | 0.0767 | 0.0000 | 762.06 ms | 74.6 | 135.3 MB |

---

## 2. Action Primitive F1 Breakdown

| Checkpoint | CLICK | TYPE | SCROLL | NAVIGATE | API_CALL | FINISH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **checkpoint_step_300** | 0.571 | 0.667 | 0.632 | 1.000 | 0.000 | 0.348 |
| **jed_m2_grpo_step_50** | 0.598 | 0.667 | 0.600 | 0.960 | 0.000 | 0.364 |
| **jed_m2_grpo_step_200** | 0.554 | 0.667 | 0.600 | 0.960 | 0.000 | 0.320 |

---

## 3. Metric Definitions & Formulas

1. **Task Success Rate (SR %)**: Exact task goal completion ($SR = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\text{success}_i) \times 100$).
2. **Step Success Rate**: Ratio of valid intermediate sub-goals achieved across all execution turns.
3. **Action Primitive Macro F1**: Unweighted harmonic mean of precision and recall across discrete tools.
4. **Spatial Coordinate MAE**: Mean absolute error $\frac{1}{2}(|\hat{x} - x^*| + |\hat{y} - y^*|)$ in normalized $[0, 1]^2$.
5. **Coordinate Tolerance Accuracy**: Fraction of actions with Euclidean distance $\le 0.05$ or falling within element bbox.
6. **Trajectory Efficiency Ratio**: Ratio of optimal demonstration steps to agent steps ($L_{\text{optimal}} / L_{\text{agent}}$).
7. **Decision Latency**: Average milliseconds elapsed per decision step including rationale and action emission.