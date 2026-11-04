# Design

## Context

JED-M2 is a 45M parameter Large Action Model operating under strict physical constraints:
- NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7, sm_120 Blackwell architecture, `torch.bfloat16`).
- Physical memory budget ceiling: $\le 6.0$ GB peak VRAM during active forward/backward passes.
- Pure PyTorch rule: zero external compiled C++/CUDA packages and zero external Hugging Face model dependencies (`from_pretrained` is forbidden).
- Prior phases established text/DOM tokenization, Mamba SSM recurrence, MLA latent attention, MoE routing, JEPA world modeling, dual-system action heads, SFT convergence, and RL rollout pipelines.

This design introduces native visual perception to form a complete multimodal Vision-LAM.

## Goals / Non-Goals

**Goals:**
- Implement a lightweight, pure PyTorch visual patch encoder (`jed_m2/vision.py`) supporting RGB screenshot images $[B, 3, H, W]$ at resolutions such as $224 \times 224$ and $448 \times 448$.
- Generate visual patch tokens of dimension `hidden_dim` ($D=512$) with 2D spatial positional embeddings.
- Implement dual multimodal observation ingestion in `JEDM2Model` via early-fusion sequence concatenation and multimodal attention masking.
- Implement cross-attention visual grounding in `CoordinateHead` and `NativeActionHead` to anchor spatial $(x, y) \in [0, 1]$ coordinate regression directly in visual patch features.
- Extend `JEPAWorldModel` to support multimodal observation encoding and transition prediction.
- Extend `SyntheticTrajectoryGenerator` and `TrajectoryCollator` in `jed_m2/dataset.py` to synthesize screenshot tensors $[B, 3, H, W]$ with colored element patches matching UI bounding boxes.
- Implement `verify_phase6d.py` validating multimodal forward/backward passes, visual coordinate grounding, zero NaNs/Infs, target network gradient isolation, and strict VRAM ceiling compliance ($< 6.0$ GB).

**Non-Goals:**
- External heavy vision foundation models (ViT-H, CLIP, SigLIP) or Hugging Face pretrained weights.
- Multi-frame video temporal modeling (each interaction turn consumes the active screenshot state).
- Modifying the core internal recurrent block layers (`JEDM2Block` containing Mamba SSM, MLA, and MoE remains intact).

## Decisions

### 1. Pure PyTorch 2D Convolutional Patch Projection
- **Choice**: Use a 2D convolution `nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)` followed by flattening spatial dimensions and applying `RMSNorm`.
- **Rationale**: A 2D convolution with kernel size and stride equal to `patch_size` is mathematically identical to a linear patch projection, highly optimized in PyTorch/cuDNN, and natively operates on 2D image grids without complex indexing.
- **Alternatives Considered**:
  - `unfold` + `nn.Linear`: Requires explicit tensor reshaping and extra buffer allocations.
  - External torchvision models: Violates the zero external model build rule.

### 2. 2D Spatial Positional Embeddings
- **Choice**: Parameterized 2D spatial positional embedding grid `[1, max_patches, hidden_dim]` added to patch tokens, with bicubic or bilinear interpolation fallback if variable resolutions are ingested.
- **Rationale**: Standard 1D position embeddings lose 2D spatial topological continuity (row and column relationships), whereas 2D positional embeddings preserve horizontal and vertical spatial relationships essential for $(x, y)$ coordinate regression.

### 3. Early-Fusion Sequence Concatenation in Backbone
- **Choice**: Visual patch tokens are prefixed to the text/DOM token embeddings: `x = torch.cat([visual_tokens, text_embeds], dim=1)`.
- **Rationale**: Placing visual tokens before text tokens allows the multi-head latent attention (MLA) in all 3 `JEDM2Block` layers to causally attend to visual patches while generating System 2 thoughts and System 1 action tokens.
- **Attention Masking**: Visual patch tokens attend bidirectionally among themselves, while text tokens attend causally to previous text tokens and all visual patch tokens.

### 4. Cross-Attention Visual Grounding for Coordinate Regression
- **Choice**: In `CoordinateHead`, introduce a lightweight cross-attention visual grounding layer:
  $$h_{\text{grounded}} = \text{RMSNorm}(h + \text{MultiheadAttention}(Q=h, K=v, V=v))$$
  where $h$ is the action token hidden state `[B, 1, D]` and $v$ is the visual patch sequence `[B, N_v, D]`. The grounded representation is then passed to the coordinate MLP:
  $$\hat{c} = \text{Sigmoid}(\text{Linear}(\text{GELU}(\text{Linear}(h_{\text{grounded}})))) \in [0, 1]^2$$
- **Rationale**: While early fusion allows implicit attention, explicit visual cross-attention in the coordinate head creates a direct gradient path between spatial coordinate errors and the visual patch representations corresponding to target UI elements.
- **Fallback**: When `visual_tokens` is `None`, $h_{\text{grounded}} = h$, providing 100% backward compatibility for text-only workflows.

### 5. Multimodal Observation State Encoding in JEPA
- **Choice**: Pool visual patch tokens via spatial mean pooling `[B, N_v, D] -> [B, D]`, concatenate with dense DOM feature representation, and project to `jepa_latent_dim` ($D_z=256$) via `RMSNorm` and linear projection.
- **Rationale**: Keeps the abstract latent state $z_t \in \mathbb{R}^{256}$ compact and consistent with Phase 2 specifications while enriching $z_t$ with visual appearance information.

### 6. Synthetic Screenshot Rendering in Dataset
- **Choice**: In `SyntheticTrajectoryGenerator`, construct a synthetic 3-channel RGB image tensor `[3, 224, 224]` initialized to a clean background color (e.g., light gray `#F0F2F5`), rendering colored rectangles corresponding to the normalized bounding boxes of generated UI elements.
- **Rationale**: Generates deterministic, visually grounded screenshot images where pixel color, contrast, and coordinates align exactly with the DOM elements and target click coordinates, requiring zero external graphics or rendering engines (e.g. no Chromium or Selenium).

## Risks / Trade-offs

- **[Risk] Activation memory pressure with large patch counts**: At $448 \times 448$ and patch size 16, $N_p = 784$ patches. Combined with sequence length 2048, total sequence length is $2832$, potentially increasing activation memory.
  - **Mitigation**: Standard resolution is $224 \times 224$ ($N_p = 196$), and selective gradient checkpointing (`gradient_checkpointing_enable()`) is fully active. Peak VRAM will remain $< 1.5$ GB on CUDA in `bfloat16`, well under the 6.0 GB ceiling.
- **[Risk] Backward compatibility across existing test suites (Phases 1-6C)**: Any signature changes to `JEDM2Model.forward`, `step`, or `JEPAWorldModel` could break existing tests.
  - **Mitigation**: All visual arguments (`images`, `visual_tokens`) default to `None`. All text-only code paths remain completely identical to previous phases.
- **[Risk] Numerical instability in visual cross-attention**:
  - **Mitigation**: Standard PyTorch `F.scaled_dot_product_attention` with pre-RMSNorm and residual connections guarantees numerical stability in `torch.bfloat16`.

## Migration Plan

1. Implement `jed_m2/vision.py` (`VisionPatchEncoder`, `VisionConfig`).
2. Update `jed_m2/config.py` with vision hyperparameters (`image_size=224`, `patch_size=16`, `num_channels=3`).
3. Update `jed_m2/heads.py` (`CoordinateHead` cross-attention visual grounding).
4. Update `jed_m2/jepa.py` (multimodal observation encoding in `LatentStateEncoder`).
5. Update `jed_m2/model.py` (`JEDM2Model` forward and step multimodal integration).
6. Update `jed_m2/dataset.py` (synthetic screenshot rendering and `TrajectoryBatch` collation).
7. Update `jed_m2/__init__.py` to export vision components.
8. Create and run `verify_phase6d.py`.
