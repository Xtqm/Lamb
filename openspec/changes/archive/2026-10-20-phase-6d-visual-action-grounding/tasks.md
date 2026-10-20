# Tasks

## 1. Visual Configuration & Patch Encoder

- [x] 1.1 Extend `JEDM2Config` in `jed_m2/config.py` with vision hyperparameters (`image_size=224`, `patch_size=16`, `num_channels=3`) and verify defaults.
- [x] 1.2 Implement `VisionPatchEncoder` in `jed_m2/vision.py` with 2D convolutional patch projection, 2D spatial positional embeddings, and RMSNorm, and verify output shapes `[B, N_p, hidden_dim]` for both 224x224 and 448x448 inputs.

## 2. Visually Grounded Action Heads & JEPA World Model

- [x] 2.1 Extend `CoordinateHead` and `NativeActionHead` in `jed_m2/heads.py` to support visual cross-attention grounding over patch tokens with fallback for text-only inputs, and verify forward output parity.
- [x] 2.2 Extend `LatentStateEncoder` and `JEPAWorldModel` in `jed_m2/jepa.py` to accept multimodal observation inputs (DOM features + visual patch features), and verify online encoding, target encoding, and loss evaluation.

## 3. Multimodal LAM Backbone Integration

- [x] 3.1 Extend `JEDM2Model` in `jed_m2/model.py` to accept visual inputs (`images` or `visual_tokens`), perform early-fusion sequence concatenation, construct multimodal attention masks, and pass visual tokens to action heads.
- [x] 3.2 Extend `JEDM2Model.step(...)` for recurrent rollout with visual context, and update `jed_m2/__init__.py` to export `VisionPatchEncoder` and vision types.

## 4. Synthetic Visual Trajectory Dataset

- [x] 4.1 Extend `SyntheticTrajectoryGenerator` in `jed_m2/dataset.py` with synthetic screenshot rendering mapping UI element bounding boxes to colored canvas patches.
- [x] 4.2 Extend `TrajectoryCollator` and `TrajectoryBatch` in `jed_m2/dataset.py` to collate and transfer batch screenshot image tensors `[B, 3, H, W]`, and verify batch collation.

## 5. Verification & Memory Profiling

- [x] 5.1 Create `verify_phase6d.py` and run verification asserting:
  - Visual patch encoder forward shape correctness across resolutions.
  - Multimodal sequence forward and backward pass stability with clean gradients and zero NaNs/Infs.
  - JEPA target encoder gradient isolation (`grad is None`).
  - Spatial coordinate regression sensitivity to visual patch features.
  - Multimodal recurrent step rollout state integrity.
  - Peak GPU VRAM strictly below the 6.0 GB ceiling on CUDA in `bfloat16`.
- [x] 5.2 Update `PROJECT_CONTEXT.md` to reflect Phase 6D completion and metrics.

## Workflow follow-up

- Archive the change after implementation and validation are completed.
