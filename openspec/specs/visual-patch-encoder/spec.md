# visual-patch-encoder Specification

## Purpose

Provides a lightweight, pure PyTorch visual screenshot patch encoder that projects RGB screen images into spatial patch embedding tokens for multimodal Large Action Models.

## Requirements

### Requirement: Screen Image Patch Tokenization
The system SHALL accept RGB screen images of shape `[B, 3, H, W]` and partition them into non-overlapping spatial patches of configured patch size `P x P`, projecting them to visual tokens of dimension `hidden_dim`.

#### Scenario: Visual patch token generation
- **WHEN** an RGB screen image batch of shape `[B, 3, H, W]` is passed to the visual patch encoder
- **THEN** the encoder returns visual patch tokens of shape `[B, (H/P) * (W/P), hidden_dim]`

#### Scenario: Variable resolution support
- **WHEN** images of standard screen resolutions such as 224x224 or 448x448 are provided
- **THEN** the encoder partitions patches without errors and produces valid patch sequences matching the spatial grid dimensions

### Requirement: 2D Spatial Positional Embeddings
The visual patch encoder SHALL apply 2D spatial positional embeddings to retain 2D grid coordinates and relative geometry across visual tokens.

#### Scenario: Positional embedding addition
- **WHEN** patch embeddings are extracted from a 2D image grid
- **THEN** each patch token is augmented with 2D spatial position encodings corresponding to its row and column coordinates

### Requirement: Pure PyTorch Implementation Constraint
The visual patch encoder SHALL be implemented 100% in native PyTorch without external Hugging Face packages or compiled C++/CUDA extensions.

#### Scenario: Standalone execution without external vision libraries
- **WHEN** the patch encoder module is imported and executed
- **THEN** it executes using standard PyTorch primitives without invoking external vision frameworks
