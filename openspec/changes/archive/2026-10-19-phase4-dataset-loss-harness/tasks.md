# Tasks

## 1. Synthetic Trajectory Dataset Implementation

- [x] 1.1 Implement trajectory tuple data structures (`UIElement`, `InteractionTurn`, `TrajectorySample`) with DOM representation and normalized bounding boxes in `jed_m2/dataset.py`.
- [x] 1.2 Implement the synthetic multi-turn web/OS trajectory generator modeled after WebArena / Mind2Web, and the streaming PyTorch Dataset (`AgentTrajectoryDataset`) in `jed_m2/dataset.py`.
- [x] 1.3 Implement `TrajectoryCollator` in `jed_m2/dataset.py` assembling dynamic batches with causal attention masks, action token index mapping, component loss masks (reasoning, action type, coordinates, arguments), and JEPA observation transition pairs.

## 2. Composite Loss Harness Implementation

- [x] 2.1 Implement `JEDM2LossConfig` and `JEDM2CompositeLoss` in `jed_m2/losses.py` uniting reasoning cross-entropy, action type cross-entropy, coordinate Smooth L1 / Huber regression, argument masked cross-entropy, JEPA latent consistency, and MoE load-balancing loss.
- [x] 2.2 Implement robust zero-mask guardrails and comprehensive metric dictionary logging in `jed_m2/losses.py` returning detached scalars for all sub-losses.

## 3. Public API Integration & Standalone Verification

- [x] 3.1 Re-export dataset, generator, collator, and loss harness classes in `jed_m2/__init__.py`.
- [x] 3.2 Create and run `verify_phase4.py` verifying dynamic batch collation, forward loss evaluation, backward pass with gradient assertions (zero NaNs/Infs), and memory consumption on CUDA in `bfloat16` within the 2.0 GB verification envelope.
