# Proposal

## Why

The current JED-M2 architecture relies exclusively on discrete text and serialized DOM trees for environment perception and spatial pointer coordinate prediction. Pure DOM representations struggle with visual layout ambiguities, dynamic canvas rendering, and pixel-level element alignments in modern Web and OS environments. Introducing a lightweight, pure PyTorch visual patch encoder and dual multimodal observation grounding enables the agent to directly perceive GUI screenshots, anchor spatial $(x, y)$ coordinate predictions in visual features, and maintain joint vision-language predictive representations in the JEPA world model, all while strictly adhering to the 6.0 GB VRAM hardware budget and zero external Hugging Face dependency constraints.

## What Changes

- **Pure PyTorch Visual Patch Encoder (`jed_m2/vision.py`)**:
  - Implement `VisionPatchEncoder` mapping RGB screenshots $[B, 3, H, W]$ (e.g., $224 \times 224$ or $448 \times 448$) into sequence patch tokens $[B, N_{\text{patches}}, \text{hidden\_dim}]$ using a 2D convolutional or linear patch projection with 2D sinusoidal or learnable positional embeddings.
  - Zero external Hugging Face or torchvision model dependencies.
- **Dual Multimodal LAM Backbone (`jed_m2/model.py`)**:
  - Extend `JEDM2Model.forward(...)` and `step(...)` to accept visual inputs (`images` $[B, 3, H, W]$ or `visual_tokens` $[B, N_v, D]$).
  - Implement early-fusion token concatenation (`[visual_tokens, text_tokens]`) and attention mask integration.
  - Extend coordinate prediction with cross-attention / early-fusion visual grounding for precise spatial coordinate regression.
- **Multimodal JEPA Latent World Model (`jed_m2/jepa.py`)**:
  - Extend `JEPAWorldModel` and `LatentStateEncoder` to accept multimodal observation inputs (fusing visual patch representations with dense DOM features) for both online encoding and EMA target tracking.
- **Visually Grounded Action Heads (`jed_m2/heads.py`)**:
  - Extend `CoordinateHead` and `NativeActionHead` to support visual cross-attention grounding, querying visual patch tokens with action hidden states to regress normalized coordinates $(x, y) \in [0, 1]$.
- **Synthetic Visual Trajectory Dataset (`jed_m2/dataset.py`)**:
  - Add synthetic screenshot rendering / element patch feature generation to `SyntheticTrajectoryGenerator`.
  - Extend `InteractionTurn` and `TrajectoryBatch` to include screenshot tensors $[B, 3, H, W]$ and visual attention masks.
- **Phase 6D Verification Harness (`verify_phase6d.py`)**:
  - Create a standalone verification script verifying multimodal forward/backward pass, gradient integrity, visual coordinate grounding accuracy, and strict compliance with the 6.0 GB VRAM ceiling on CUDA in `bfloat16`.

## Capabilities

### New Capabilities
- `visual-patch-encoder`: Lightweight pure PyTorch visual screenshot patch tokenization, 2D spatial positional embedding, and hidden dimension projection without external dependencies.

### Modified Capabilities
- `unified-lam-backbone`: Dual multimodal observation processing combining visual patch tokens and DOM/text tokens with early-fusion and visual coordinate grounding.
- `jepa-world-model`: Multimodal observation encoding and latent state transition prediction over combined visual and DOM representations.
- `dual-system-action-heads`: Visual cross-attention spatial coordinate regression grounding continuous $(x, y)$ pointer predictions in visual patch features.
- `synthetic-trajectory-dataset`: Synthetic screenshot image generation and multimodal batch collation of visual tensors alongside DOM/text tokens.

## Impact

- **Code Base**:
  - New module `jed_m2/vision.py`.
  - Updates to `jed_m2/config.py` (adding visual encoder hyperparameters: image size, patch size, vision channels).
  - Updates to `jed_m2/jepa.py`, `jed_m2/heads.py`, `jed_m2/model.py`, and `jed_m2/dataset.py`.
  - Public export updates in `jed_m2/__init__.py`.
  - New verification script `verify_phase6d.py`.
- **APIs**:
  - `JEDM2Model.forward` accepts optional `images` or `visual_tokens`.
  - `NativeActionHead.forward` and `CoordinateHead.forward` accept optional `visual_tokens` for visual cross-attention grounding.
  - `JEPAWorldModel.forward` accepts multimodal observation features or visual token summaries.
  - `TrajectoryBatch` includes `images` $[B, 3, H, W]$.
- **Hardware & VRAM**:
  - Memory footprint remains well within the 6.0 GB physical VRAM ceiling (typically $< 1.5$ GB in `bfloat16` for batch size 2 and $224 \times 224$ images).
- **Dependencies**:
  - Zero external Hugging Face packages or compiled C++/CUDA extensions; 100% native PyTorch.
