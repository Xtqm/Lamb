# JED-M2 Large Action Model - Benchmark Evaluation Report

**Generated:** 2026-11-03 22:59:10 UTC  
**Benchmark Domain:** `real_web_test`  
**Evaluation Device:** `cuda`  

---

### Context & Notes
Evaluated on 11 episodes with max_steps=15 and spatial tolerance epsilon=0.05 on device=cuda.

## 1. Comparative Performance Summary

| Checkpoint | Split | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Native FINISH F1 | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | `test` | 11 | 0.00% | 0.00% | 0.0000 | 1.0000 | 0.4286 | 0.0000 | 0.4242 | 0.0000 | 0.0000 | 87.85 ms | 458.6 MB |
| **jed_m2_nano_pretrain_final** | `test` | 11 | 0.00% | 0.00% | 0.8143 | 1.0000 | 0.6974 | 0.7889 | 0.3443 | 0.0000 | 0.0000 | 288.76 ms | 1661.5 MB |
| **T5-Small (Fine-Tuned ~60.5M)** | `test` | 11 | 0.00% | 0.00% | 0.5574 | 1.0000 | 0.6786 | 0.6000 | 0.2771 | 0.3333 | 0.0000 | 440.99 ms | 177.0 MB |
| **DistilGPT-2 (Fine-Tuned ~81.9M)** | `test` | 11 | 0.00% | 0.00% | 0.8143 | 1.0000 | 0.6974 | 0.7889 | 0.2771 | 0.3333 | 0.0000 | 201.08 ms | 217.6 MB |
| **T5-Small (~60.5M)** | `test` | 11 | 0.00% | 0.00% | 0.7234 | 1.0000 | 0.6912 | 0.7222 | 0.2668 | 0.3333 | 0.0000 | 386.37 ms | 295.8 MB |
| **DistilGPT-2 (~81.9M)** | `test` | 11 | 0.00% | 0.00% | 0.8506 | 1.0000 | 0.7817 | 0.8556 | 0.2732 | 0.2797 | 0.0000 | 315.24 ms | 402.6 MB |

---

## 2. Generalization Split Breakdown

| Checkpoint | Split / Category | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Action Accuracy | Coord MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | `Travel` | 5 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4492 |
| **jed_m2_sft_base** | `Shopping` | 4 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4146 |
| **jed_m2_sft_base** | `Entertainment` | 2 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3807 |
| **jed_m2_nano_pretrain_final** | `Travel` | 5 | 0.00% | 0.00% | 0.6800 | 0.6800 | 0.3104 |
| **jed_m2_nano_pretrain_final** | `Shopping` | 4 | 0.00% | 0.00% | 0.8893 | 0.8893 | 0.3853 |
| **jed_m2_nano_pretrain_final** | `Entertainment` | 2 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.3470 |
| **T5-Small (Fine-Tuned ~60.5M)** | `Travel` | 5 | 0.00% | 0.00% | 0.6800 | 0.6800 | 0.1761 |
| **T5-Small (Fine-Tuned ~60.5M)** | `Shopping` | 4 | 0.00% | 0.00% | 0.6829 | 0.6829 | 0.3176 |
| **T5-Small (Fine-Tuned ~60.5M)** | `Entertainment` | 2 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4487 |
| **DistilGPT-2 (Fine-Tuned ~81.9M)** | `Travel` | 5 | 0.00% | 0.00% | 0.6800 | 0.6800 | 0.1761 |
| **DistilGPT-2 (Fine-Tuned ~81.9M)** | `Shopping` | 4 | 0.00% | 0.00% | 0.8893 | 0.8893 | 0.3176 |
| **DistilGPT-2 (Fine-Tuned ~81.9M)** | `Entertainment` | 2 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.4487 |
| **T5-Small (~60.5M)** | `Travel` | 5 | 0.00% | 0.00% | 0.6800 | 0.6800 | 0.1761 |
| **T5-Small (~60.5M)** | `Shopping` | 4 | 0.00% | 0.00% | 0.8893 | 0.8893 | 0.3176 |
| **T5-Small (~60.5M)** | `Entertainment` | 2 | 0.00% | 0.00% | 0.5000 | 0.5000 | 0.3920 |
| **DistilGPT-2 (~81.9M)** | `Travel` | 5 | 0.00% | 0.00% | 0.7600 | 0.7600 | 0.2040 |
| **DistilGPT-2 (~81.9M)** | `Shopping` | 4 | 0.00% | 0.00% | 0.8893 | 0.8893 | 0.3241 |
| **DistilGPT-2 (~81.9M)** | `Entertainment` | 2 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.3446 |

---

## 3. Action Primitive F1 Breakdown

| Checkpoint | CLICK | TYPE | SCROLL | NAVIGATE | API_CALL | FINISH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **jed_m2_sft_base** | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| **jed_m2_nano_pretrain_final** | 0.882 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **T5-Small (Fine-Tuned ~60.5M)** | 0.750 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **DistilGPT-2 (Fine-Tuned ~81.9M)** | 0.882 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **T5-Small (~60.5M)** | 0.839 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **DistilGPT-2 (~81.9M)** | 0.920 | 0.552 | 1.000 | 1.000 | 1.000 | 1.000 |

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