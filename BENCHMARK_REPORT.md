# JED-M2 Large Action Model - Benchmark Evaluation Report

**Generated:** 2026-10-23 13:01:15 UTC  
**Benchmark Domain:** `real_web_val`  
**Evaluation Device:** `cuda`  

---

### Context & Notes
Evaluated on 15 episodes with max_steps=15 and spatial tolerance epsilon=0.05 on device=cuda.

## 1. Comparative Performance Summary

| Checkpoint | Episodes | Task SR (%) | Step SR | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Throughput (tok/s) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_medium_step_1500** | 15 | 0.00% | 0.8303 | 0.8192 | 0.8440 | 0.4382 | 0.0111 | 0.0000 | 9695.06 ms | 38.5 | 600.7 MB |
| **jed_m2_medium_step_3500** | 15 | 0.00% | 0.8103 | 0.8331 | 0.8257 | 0.3210 | 0.1192 | 0.0000 | 9664.19 ms | 38.3 | 600.5 MB |
| **jed_m2_nano_step_500** | 15 | 0.00% | 0.5403 | 0.6214 | 0.5413 | 0.3946 | 0.0074 | 0.0000 | 3907.23 ms | 95.2 | 145.4 MB |
| **jed_m2_nano_step_2000** | 15 | 0.00% | 0.8103 | 0.8331 | 0.8257 | 0.3604 | 0.1343 | 0.0000 | 3920.04 ms | 94.4 | 145.4 MB |
| **jed_m2_nano_step_3000** | 15 | 0.00% | 0.8103 | 0.8331 | 0.8257 | 0.3618 | 0.1033 | 0.0000 | 4436.07 ms | 83.3 | 145.4 MB |
| **jed_m2_nano_step_12200** | 15 | 0.00% | 0.7557 | 0.8343 | 0.7706 | 0.3485 | 0.0875 | 0.0000 | 3849.86 ms | 96.0 | 145.4 MB |

---

## 2. Action Primitive F1 Breakdown

| Checkpoint | CLICK | TYPE | SCROLL | NAVIGATE | API_CALL | FINISH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_medium_step_1500** | 0.915 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **jed_m2_medium_step_3500** | 0.904 | 0.095 | 1.000 | 1.000 | 1.000 | 1.000 |
| **jed_m2_nano_step_500** | 0.728 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| **jed_m2_nano_step_2000** | 0.904 | 0.095 | 1.000 | 1.000 | 1.000 | 1.000 |
| **jed_m2_nano_step_3000** | 0.904 | 0.095 | 1.000 | 1.000 | 1.000 | 1.000 |
| **jed_m2_nano_step_12200** | 0.868 | 0.138 | 1.000 | 1.000 | 1.000 | 1.000 |

---

## 3. Metric Definitions & Formulas

1. **Task Success Rate (SR %)**: Exact task goal completion ($SR = \frac{1}{N}\sum_{i=1}^N \mathbb{I}(\text{success}_i) \times 100$).
2. **Step Success Rate**: Ratio of valid intermediate sub-goals achieved across all execution turns.
3. **Action Primitive Macro F1**: Unweighted harmonic mean of precision and recall across discrete tools.
4. **Spatial Coordinate MAE**: Mean absolute error $\frac{1}{2}(|\hat{x} - x^*| + |\hat{y} - y^*|)$ in normalized $[0, 1]^2$.
5. **Coordinate Tolerance Accuracy**: Fraction of actions with Euclidean distance $\le 0.05$ or falling within element bbox.
6. **Trajectory Efficiency Ratio**: Ratio of optimal demonstration steps to agent steps ($L_{\text{optimal}} / L_{\text{agent}}$).
7. **Decision Latency**: Average milliseconds elapsed per decision step including rationale and action emission.