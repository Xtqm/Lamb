# JED-M2 Large Action Model - Benchmark Evaluation Report

**Generated:** 2026-11-05 01:05:37 UTC  
**Benchmark Domain:** `mind2web_heldout_11sites`  
**Evaluation Device:** `cuda`  

---

### Context & Notes
Evaluated across 11 held-out Mind2Web websites with spatial tolerance radius epsilon=0.05 on cuda.
Compares SFT Base, Step-shaped GRPO (v1.0), Outcome-supervised GRPO (v2.0 Policy), and Outcome GRPO + JEPA Best-of-N Test-Time Search (v2.0 Full) alongside T5-Small and DistilGPT-2.

## 1. Comparative Performance Summary

| Checkpoint | Split | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Native FINISH F1 | Macro F1 | Micro F1 | Coord MAE | Coord Tol Acc (eps<=0.05) | Efficiency Ratio | Latency (ms/step) | Peak VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **JED-M2 (SFT Base)** | `test` | 11 | 0.00% | 0.00% | 0.1284 | 1.0000 | 0.6146 | 0.1782 | 0.3914 | 0.0000 | 0.0000 | 718.58 ms | 1682.9 MB |
| **JED-M2 v1.0 (Step GRPO)** | `test` | 11 | 0.00% | 0.00% | 0.0505 | 0.0000 | 0.4454 | 0.0495 | 0.3748 | 0.0000 | 0.0000 | 556.27 ms | 1681.4 MB |
| **JED-M2 v2.0 (Outcome GRPO)** | `test` | 11 | 0.00% | 0.00% | 0.1284 | 1.0000 | 0.6146 | 0.1782 | 0.3897 | 0.0000 | 0.0000 | 843.54 ms | 1682.9 MB |
| **JED-M2 v2.0 Full (+JEPA Search)** | `test` | 11 | 0.00% | 0.00% | 0.1284 | 1.0000 | 0.6146 | 0.1782 | 0.3897 | 0.0000 | 0.0000 | 873.99 ms | 1682.9 MB |
| **T5-Small (~60.5M)** | `test` | 11 | 0.00% | 0.00% | 0.7727 | 1.0000 | 0.7033 | 0.7327 | 0.2804 | 0.2890 | 0.0000 | 414.51 ms | 295.8 MB |
| **DistilGPT-2 (~81.9M)** | `test` | 11 | 0.00% | 0.00% | 0.8376 | 1.0000 | 0.7712 | 0.8218 | 0.3245 | 0.1668 | 0.0000 | 229.52 ms | 402.6 MB |

---

## 2. Generalization Split Breakdown

| Checkpoint | Split / Category | Episodes | Autonomous SR (%) | Referee-Assisted SR (%) | Step SR | Action Accuracy | Coord MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **JED-M2 (SFT Base)** | `agoda` | 1 | 0.00% | 0.00% | 0.1111 | 0.1111 | 0.4154 |
| **JED-M2 (SFT Base)** | `amazon` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4657 |
| **JED-M2 (SFT Base)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3737 |
| **JED-M2 (SFT Base)** | `carmax` | 1 | 0.00% | 0.00% | 0.3000 | 0.3000 | 0.3357 |
| **JED-M2 (SFT Base)** | `delta` | 1 | 0.00% | 0.00% | 0.1333 | 0.1333 | 0.3666 |
| **JED-M2 (SFT Base)** | `instacart` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4853 |
| **JED-M2 (SFT Base)** | `last.fm` | 1 | 0.00% | 0.00% | 0.1429 | 0.1429 | 0.3376 |
| **JED-M2 (SFT Base)** | `parking` | 1 | 0.00% | 0.00% | 0.6000 | 0.6000 | 0.4664 |
| **JED-M2 (SFT Base)** | `resy` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2880 |
| **JED-M2 (SFT Base)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.1250 | 0.1250 | 0.3736 |
| **JED-M2 (SFT Base)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3977 |
| **JED-M2 v1.0 (Step GRPO)** | `agoda` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4098 |
| **JED-M2 v1.0 (Step GRPO)** | `amazon` | 1 | 0.00% | 0.00% | 0.5556 | 0.5556 | 0.4356 |
| **JED-M2 v1.0 (Step GRPO)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3497 |
| **JED-M2 v1.0 (Step GRPO)** | `carmax` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3291 |
| **JED-M2 v1.0 (Step GRPO)** | `delta` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3598 |
| **JED-M2 v1.0 (Step GRPO)** | `instacart` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4581 |
| **JED-M2 v1.0 (Step GRPO)** | `last.fm` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3237 |
| **JED-M2 v1.0 (Step GRPO)** | `parking` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4447 |
| **JED-M2 v1.0 (Step GRPO)** | `resy` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2751 |
| **JED-M2 v1.0 (Step GRPO)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3501 |
| **JED-M2 v1.0 (Step GRPO)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3865 |
| **JED-M2 v2.0 (Outcome GRPO)** | `agoda` | 1 | 0.00% | 0.00% | 0.1111 | 0.1111 | 0.4145 |
| **JED-M2 v2.0 (Outcome GRPO)** | `amazon` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4643 |
| **JED-M2 v2.0 (Outcome GRPO)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3710 |
| **JED-M2 v2.0 (Outcome GRPO)** | `carmax` | 1 | 0.00% | 0.00% | 0.3000 | 0.3000 | 0.3340 |
| **JED-M2 v2.0 (Outcome GRPO)** | `delta` | 1 | 0.00% | 0.00% | 0.1333 | 0.1333 | 0.3650 |
| **JED-M2 v2.0 (Outcome GRPO)** | `instacart` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4827 |
| **JED-M2 v2.0 (Outcome GRPO)** | `last.fm` | 1 | 0.00% | 0.00% | 0.1429 | 0.1429 | 0.3370 |
| **JED-M2 v2.0 (Outcome GRPO)** | `parking` | 1 | 0.00% | 0.00% | 0.6000 | 0.6000 | 0.4645 |
| **JED-M2 v2.0 (Outcome GRPO)** | `resy` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2860 |
| **JED-M2 v2.0 (Outcome GRPO)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.1250 | 0.1250 | 0.3711 |
| **JED-M2 v2.0 (Outcome GRPO)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3965 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `agoda` | 1 | 0.00% | 0.00% | 0.1111 | 0.1111 | 0.4145 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `amazon` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4643 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `boardgamegeek` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3710 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `carmax` | 1 | 0.00% | 0.00% | 0.3000 | 0.3000 | 0.3340 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `delta` | 1 | 0.00% | 0.00% | 0.1333 | 0.1333 | 0.3650 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `instacart` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.4827 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `last.fm` | 1 | 0.00% | 0.00% | 0.1429 | 0.1429 | 0.3370 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `parking` | 1 | 0.00% | 0.00% | 0.6000 | 0.6000 | 0.4645 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `resy` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.2860 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `thetrainline` | 1 | 0.00% | 0.00% | 0.1250 | 0.1250 | 0.3711 |
| **JED-M2 v2.0 Full (+JEPA Search)** | `ultimate-guitar` | 1 | 0.00% | 0.00% | 0.0000 | 0.0000 | 0.3965 |
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
| **JED-M2 (SFT Base)** | 0.000 | 0.303 | 1.000 | 1.000 | 1.000 | 1.000 |
| **JED-M2 v1.0 (Step GRPO)** | 0.118 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| **JED-M2 v2.0 (Outcome GRPO)** | 0.000 | 0.303 | 1.000 | 1.000 | 1.000 | 1.000 |
| **JED-M2 v2.0 Full (+JEPA Search)** | 0.000 | 0.303 | 1.000 | 1.000 | 1.000 | 1.000 |
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