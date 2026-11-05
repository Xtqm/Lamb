# JED-M2 Large Action Model - Benchmark Evaluation Report

**Generated:** 2026-11-05 04:15:25 UTC  
**Benchmark Domain:** `mind2web_heldout_11sites`  
**Evaluation Device:** `cuda`  

---

### Context & Notes
Evaluated across 11 held-out Mind2Web websites with spatial tolerance radius epsilon=0.05 on cuda.
Compares SFT Base, Step-shaped GRPO (v1.0), Outcome-supervised GRPO (v2.0 Policy), and Outcome GRPO + JEPA Best-of-N Test-Time Search (v2.0 Full) alongside T5-Small and DistilGPT-2.

## 1. Comparative Performance Summary

| Checkpoint | Split | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Native FINISH F1 | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JED-M2 (SFT Base)** | `test` | 11 | 0.00% | 0.00% | 0.8477 | 1.0000 | 0.6977 | 0.7921 | 0.3997 | 0.0156 | 0.0000 | 252.02 ms | 1682.0 MB |
| **JED-M2 v2.0 (Pretrained)** | `test` | 11 | 0.00% | 0.00% | 0.0235 | 0.0000 | 0.4389 | 0.0297 | 0.3972 | 0.0000 | 0.0000 | 250.78 ms | 1682.2 MB |
| **JED-M2 v2.0 (Domain-Aligned)** | `test` | 11 | 0.00% | 0.00% | 0.3911 | 0.0000 | 0.3746 | 0.4158 | 0.3230 | 0.0114 | 0.0000 | 250.88 ms | 1681.2 MB |
| **JED-M2 v2.0 (Outcome GRPO)** | `test` | 11 | 0.00% | 0.00% | 0.0356 | 0.0000 | 0.4452 | 0.0495 | 0.3984 | 0.0000 | 0.0000 | 253.15 ms | 1682.2 MB |
| **JED-M2 v2.0 Full (+JEPA Search)** | `test` | 11 | 0.00% | 0.00% | 0.0174 | 0.0000 | 0.1498 | 0.0198 | 0.2790 | 0.4405 | 0.0000 | 253.49 ms | 1682.8 MB |
| **T5-Small (~60.5M)** | `test` | 11 | 0.00% | 0.00% | 0.7727 | 1.0000 | 0.7033 | 0.7327 | 0.2804 | 0.2890 | 0.0000 | 326.69 ms | 295.8 MB |
| **DistilGPT-2 (~81.9M)** | `test` | 11 | 0.00% | 0.00% | 0.8376 | 1.0000 | 0.7712 | 0.8218 | 0.3245 | 0.1668 | 0.0000 | 187.07 ms | 402.6 MB |

---

## 2. Generalization Split Breakdown

| Checkpoint | Split / Category | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Action Accuracy | Coord MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **JED-M2 (SFT Base)** | `agoda` | 1 | 0.00% | 0.00% | 0.8889 | 0.8889 | 0.3999 |
| **JED-M2 (SFT Base)** | `amazon` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.4276 |
| **JED-M2 (SFT Base)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.3378 |
| **JED-M2 (SFT Base)** | `carmax` | 1 | 0.00% | 0.00% | 0.7000 | 0.7000 | 0.3313 |
| **JED-M2 (SFT Base)** | `delta` | 1 | 0.00% | 0.00% | 0.8667 | 0.8667 | 0.3998 |
| **JED-M2 (SFT Base)** | `instacart` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.4483 |
| **JED-M2 (SFT Base)** | `last.fm` | 1 | 0.00% | 0.00% | 0.7857 | 0.7857 | 0.4278 |
| **JED-M2 (SFT Base)** | `parking` | 1 | 0.00% | 0.00% | 0.3333 | 0.3333 | 0.5345 |
| **JED-M2 (SFT Base)** | `resy` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.2676 |
| **JED-M2 (SFT Base)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.7500 | 0.7500 | 0.4025 |
| **JED-M2 (SFT Base)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.4201 |
| **JED-M2 v2.0 (Pretrained)** | `agoda` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4189 |
| **JED-M2 v2.0 (Pretrained)** | `amazon` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4784 |
| **JED-M2 v2.0 (Pretrained)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3804 |
| **JED-M2 v2.0 (Pretrained)** | `carmax` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3467 |
| **JED-M2 v2.0 (Pretrained)** | `delta` | 1 | 0.00% | 0.00% | 0.1333 | 0.1333 | 0.3704 |
| **JED-M2 v2.0 (Pretrained)** | `instacart` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4934 |
| **JED-M2 v2.0 (Pretrained)** | `last.fm` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3446 |
| **JED-M2 v2.0 (Pretrained)** | `parking` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4729 |
| **JED-M2 v2.0 (Pretrained)** | `resy` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2950 |
| **JED-M2 v2.0 (Pretrained)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.1250 | 0.1250 | 0.3640 |
| **JED-M2 v2.0 (Pretrained)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4047 |
| **JED-M2 v2.0 (Domain-Aligned)** | `agoda` | 1 | 0.00% | 0.00% | 0.8889 | 0.8889 | 0.3735 |
| **JED-M2 v2.0 (Domain-Aligned)** | `amazon` | 1 | 0.00% | 0.00% | 0.2222 | 0.2222 | 0.3068 |
| **JED-M2 v2.0 (Domain-Aligned)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.2000 | 0.2000 | 0.2884 |
| **JED-M2 v2.0 (Domain-Aligned)** | `carmax` | 1 | 0.00% | 0.00% | 0.2000 | 0.2000 | 0.3344 |
| **JED-M2 v2.0 (Domain-Aligned)** | `delta` | 1 | 0.00% | 0.00% | 0.8667 | 0.8667 | 0.3637 |
| **JED-M2 v2.0 (Domain-Aligned)** | `instacart` | 1 | 0.00% | 0.00% | 0.3333 | 0.3333 | 0.3713 |
| **JED-M2 v2.0 (Domain-Aligned)** | `last.fm` | 1 | 0.00% | 0.00% | 0.3571 | 0.3571 | 0.3044 |
| **JED-M2 v2.0 (Domain-Aligned)** | `parking` | 1 | 0.00% | 0.00% | 0.1333 | 0.1333 | 0.3687 |
| **JED-M2 v2.0 (Domain-Aligned)** | `resy` | 1 | 0.00% | 0.00% | 0.4000 | 0.4000 | 0.2106 |
| **JED-M2 v2.0 (Domain-Aligned)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.5000 | 0.5000 | 0.2610 |
| **JED-M2 v2.0 (Domain-Aligned)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 0.2000 | 0.2000 | 0.3704 |
| **JED-M2 v2.0 (Outcome GRPO)** | `agoda` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4187 |
| **JED-M2 v2.0 (Outcome GRPO)** | `amazon` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4790 |
| **JED-M2 v2.0 (Outcome GRPO)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3807 |
| **JED-M2 v2.0 (Outcome GRPO)** | `carmax` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3467 |
| **JED-M2 v2.0 (Outcome GRPO)** | `delta` | 1 | 0.00% | 0.00% | 0.2000 | 0.2000 | 0.3706 |
| **JED-M2 v2.0 (Outcome GRPO)** | `instacart` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4931 |
| **JED-M2 v2.0 (Outcome GRPO)** | `last.fm` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3423 |
| **JED-M2 v2.0 (Outcome GRPO)** | `parking` | 1 | 0.00% | 0.00% | 0.0667 | 0.0667 | 0.4874 |
| **JED-M2 v2.0 (Outcome GRPO)** | `resy` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2950 |
| **JED-M2 v2.0 (Outcome GRPO)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.1250 | 0.1250 | 0.3640 |
| **JED-M2 v2.0 (Outcome GRPO)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4055 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `agoda` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3549 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `amazon` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.1742 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2519 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `carmax` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4410 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `delta` | 1 | 0.00% | 0.00% | 0.0667 | 0.0667 | 0.3767 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `instacart` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2849 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `last.fm` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2960 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `parking` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.1626 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `resy` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2601 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.1250 | 0.1250 | 0.1868 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2801 |
| **T5-Small (~60.5M)** | `agoda` | 1 | 0.00% | 0.00% | 0.8889 | 0.8889 | 0.3026 |
| **T5-Small (~60.5M)** | `amazon` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.1912 |
| **T5-Small (~60.5M)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.2318 |
| **T5-Small (~60.5M)** | `carmax` | 1 | 0.00% | 0.00% | 0.7000 | 0.7000 | 0.3861 |
| **T5-Small (~60.5M)** | `delta` | 1 | 0.00% | 0.00% | 0.8667 | 0.8667 | 0.3831 |
| **T5-Small (~60.5M)** | `instacart` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.3563 |
| **T5-Small (~60.5M)** | `last.fm` | 1 | 0.00% | 0.00% | 0.7857 | 0.7857 | 0.2960 |
| **T5-Small (~60.5M)** | `parking` | 1 | 0.00% | 0.00% | 0.3333 | 0.3333 | 0.1626 |
| **T5-Small (~60.5M)** | `resy` | 1 | 0.00% | 0.00% | 0.8000 | 0.8000 | 0.2542 |
| **T5-Small (~60.5M)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.1250 | 0.1250 | 0.1779 |
| **T5-Small (~60.5M)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.3428 |
| **DistilGPT-2 (~81.9M)** | `agoda` | 1 | 0.00% | 0.00% | 0.7778 | 0.7778 | 0.3521 |
| **DistilGPT-2 (~81.9M)** | `amazon` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.1912 |
| **DistilGPT-2 (~81.9M)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.8000 | 0.8000 | 0.2197 |
| **DistilGPT-2 (~81.9M)** | `carmax` | 1 | 0.00% | 0.00% | 0.7000 | 0.7000 | 0.3322 |
| **DistilGPT-2 (~81.9M)** | `delta` | 1 | 0.00% | 0.00% | 0.8667 | 0.8667 | 0.3667 |
| **DistilGPT-2 (~81.9M)** | `instacart` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.4135 |
| **DistilGPT-2 (~81.9M)** | `last.fm` | 1 | 0.00% | 0.00% | 0.7857 | 0.7857 | 0.2960 |
| **DistilGPT-2 (~81.9M)** | `parking` | 1 | 0.00% | 0.00% | 0.7333 | 0.7333 | 0.3181 |
| **DistilGPT-2 (~81.9M)** | `resy` | 1 | 0.00% | 0.00% | 0.8000 | 0.8000 | 0.2268 |
| **DistilGPT-2 (~81.9M)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.7500 | 0.7500 | 0.4910 |
| **DistilGPT-2 (~81.9M)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 1.0000 | 1.0000 | 0.3621 |

---

## 3. Action Primitive F1 Breakdown

| Checkpoint | CLICK | TYPE | SCROLL | NAVIGATE | API_CALL | FINISH |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **JED-M2 (SFT Base)** | 0.884 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **JED-M2 v2.0 (Pretrained)** | 0.072 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| **JED-M2 v2.0 (Domain-Aligned)** | 0.622 | 0.000 | 0.000 | 1.000 | 1.000 | 0.000 |
| **JED-M2 v2.0 (Outcome GRPO)** | 0.116 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| **JED-M2 v2.0 Full (+JEPA Search)** | 0.049 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| **T5-Small (~60.5M)** | 0.849 | 0.074 | 1.000 | 1.000 | 1.000 | 1.000 |
| **DistilGPT-2 (~81.9M)** | 0.898 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |

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