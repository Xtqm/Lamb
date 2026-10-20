# Spec Delta

## ADDED Requirements

### Requirement: Synthetic Screenshot Rendering and Visual Feature Synthesis
The trajectory generator SHALL generate synthetic screenshot image representations `[3, H, W]` or visual patch features for interaction turns, corresponding to rendered UI element bounding boxes and screen canvas.

#### Scenario: Synthetic screenshot generation per turn
- **WHEN** interaction turns are generated in `SyntheticTrajectoryGenerator`
- **THEN** each turn provides a synthetic screenshot tensor of shape `[3, H, W]` with pixel features corresponding to the UI elements

### Requirement: Multimodal Batch Collation with Screenshot Tensors
The trajectory collator SHALL assemble screenshot images alongside text tokens, producing batched image tensors `[B, 3, H, W]` in `TrajectoryBatch`.

#### Scenario: Collate batch with visual screenshot images
- **WHEN** a list of multimodal trajectory samples is collated by `TrajectoryCollator`
- **THEN** the resulting `TrajectoryBatch` includes an `images` tensor of shape `[B, 3, H, W]` aligned with the trajectory batch
