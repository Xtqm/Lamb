# JED-M2 Large Action Model - Benchmark Evaluation Report

**Generated:** 2026-10-30 21:21:26 UTC  
**Benchmark Domain:** `webos_simulated`  
**Evaluation Device:** `cuda`  

---

### Context & Notes
Evaluated on 9 episodes with max_steps=15 and spatial tolerance epsilon=0.05 on device=cuda.

## 1. Comparative Performance Summary

| Checkpoint | Episodes | Task SR (%) | Step SR | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Throughput (tok/s) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_nano_pretrain_final** | 9 | 33.33% | 0.6548 | 0.7759 | 0.6522 | 0.0000 | 1.0000 | 0.2381 | 794.81 ms | 65.5 | 135.6 MB |
| **jed_m2_nano_grpo_final** | 9 | 66.67% | 0.7500 | 0.7500 | 0.6250 | 0.0000 | 1.0000 | 0.6667 | 1050.22 ms | 51.0 | 135.4 MB |

---

## 2. Action Primitive F1 Breakdown

| Checkpoint | CLICK | TYPE | SCROLL | NAVIGATE | API_CALL | FINISH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_nano_pretrain_final** | 0.765 | 0.667 | 1.000 | 1.000 | 1.000 | 0.000 |
| **jed_m2_nano_grpo_final** | 0.750 | 0.500 | 1.000 | 1.000 | 1.000 | 0.000 |

---

## 3. Metric Definitions & Formulas

1. **Task Success Rate (SR %)**: Exact task goal completion ($SR = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\text{success}_i) \times 100$).
2. **Step Success Rate**: Ratio of valid intermediate sub-goals achieved across all execution turns.
3. **Action Primitive Macro F1**: Unweighted harmonic mean of precision and recall across discrete tools.
4. **Spatial Coordinate MAE**: Mean absolute error $\frac{1}{2}(|\hat{x} - x^*| + |\hat{y} - y^*|)$ in normalized $[0, 1]^2$.
5. **Coordinate Tolerance Accuracy**: Fraction of actions with Euclidean distance $\le 0.05$ or falling within element bbox.
6. **Trajectory Efficiency Ratio**: Ratio of optimal demonstration steps to agent steps ($L_{\text{optimal}} / L_{\text{agent}}$).
7. **Decision Latency**: Average milliseconds elapsed per decision step including rationale and action emission.