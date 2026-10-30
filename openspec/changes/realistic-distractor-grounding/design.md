# Design: Realistic Distractor Grounding and Benchmark Alignment

## Context

In Mind2Web trajectory traces, each action object includes two candidate lists:
- `pos_candidates`: Positive target elements annotated by humans (typically 1 element).
- `neg_candidates`: Negative interactive distractors identified by the scraper (typically 50–200 elements), each annotated with `backend_node_id`, `tag`, and `bounding_box_rect`.

Currently, `RealWebTrajectoryDataset._generate_from_real_manifest` only iterates over `pos_candidates` when building `bbox_map`. As diagnosed in the adversarial audit, this created an $N=1$ candidate pool at evaluation time, turning candidate snapping into an element-presence gate rather than a test of true visual/spatial discrimination.

## Goals / Non-Goals

**Goals:**
- Ingest and bind bounding boxes from both `pos_candidates` and `neg_candidates` into `bbox_map` during real dataset parsing.
- Produce candidate pools containing 20–64 parsed interactive DOM elements with authentic spatial geometries for every turn.
- Re-evaluate JED-M2 on the site-held-out Mind2Web test split (147 tasks, 11 unseen websites) against genuine multi-element distractors.
- Establish an unvarnished, peer-review-grade benchmark report highlighting unassisted raw continuous coordinates as primary, multi-element snapping as secondary, and transparently detailing argument string matching.

**Non-Goals:**
- Modifying model neural architecture or pretraining weights (the current 3,000-step weights `jed_m2_nano_pretrain_final.pt` are evaluated as-is).
- Running full online GRPO training in this change (GRPO is the subsequent phase unlocked by this alignment).
- Changing synthetic or simulated environment element generation.

## Decisions

### Decision 1: Joint Ingestion of Positive and Negative Candidates
- **Approach**: Combine `(action.get("pos_candidates") or []) + (action.get("neg_candidates") or [])` when populating `bbox_map` and extracting candidate bounding boxes.
- **Rationale**: Both positive and negative candidates in raw Mind2Web provide `backend_node_id` and `bounding_box_rect`. Including negative candidates allows `FastDOMParser` to resolve bounding boxes for all interactive elements in the HTML tree.
- **Alternative Considered**: Parsing raw bounding boxes directly from HTML style attributes. *Rejected*: Mind2Web already extracted verified layout bounding boxes in `bounding_box_rect`; re-parsing from HTML string would be brittle and slow.

### Decision 2: Strict Isolation Between Target Scoring and Candidate Pools
- **Approach**: Target element identity (`target_element_id`) and target bounding box (`target_bbox`) continue to be derived solely from `original_candidates` in `pos_candidates`. Only the candidate pool for snapping (`turn.interactive_elements`) is expanded.
- **Rationale**: Prevents any label leakage while giving the snapping heuristic a realistic set of alternatives to choose from.

### Decision 3: Cache Key Tag Update (`v5`)
- **Approach**: Increment real-only cache prefix to `real_web_real_v5_{manifest_hash}_{split}_h{hidden_dim}.pt`.
- **Rationale**: Ensures the test dataset rebuilds its cache with negative bounding boxes populated, avoiding stale $N=1$ cached trajectory tensors.

### Decision 4: Transparent Scientific Reporting Hierarchy
- **Approach**: Structure `POST_PRETRAIN_BENCHMARK_REPORT.md` and evaluation tables such that:
  1. Raw unassisted continuous coordinates (74.58% Target Hit Rate, 0.0151 MAE) are presented as the primary baseline metric.
  2. Multi-element distractor snapping is presented as an operational DOM post-processing layer.
  3. Action classification (99.74%) and multi-token argument generation (2.17%) are reported with complete numerator/denominator transparency.

## Risks / Trade-offs

- **[Risk] Increased Cache File Size**: Storing bounding boxes for up to 64 elements per turn across 1,178 turns will slightly increase memory.
  - *Mitigation*: Bounding boxes are stored as lightweight 4-tuples of floats; test dataset cache increase is $< 5\text{ MB}$, well within system constraints.
- **[Risk] Snapped Target Element Hit Rate Will Drop from 100%**: In the presence of 50 distractors, snapping a slightly displaced click might snap to a nearby negative element.
  - *Mitigation*: This is the desired scientific outcome. An honest 75%–85% hit rate among dozens of competing buttons is methodologically sound and defensible in top-tier peer review.
