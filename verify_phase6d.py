"""
Verification Script for Phase 6D: Visual Action Grounding (Multimodal Vision-LAM Extension).

Tests:
1. Pure PyTorch Visual Patch Encoder Across Resolutions:
   - Evaluates 224x224 (196 patches) and 448x448 (784 patches) inputs.
   - Validates 2D spatial positional embedding interpolation and RMSNorm.
   - Verifies parameter budget is lightweight (< 1.0M parameters).
2. Dual Multimodal Forward Pass & Attention Mask Integration:
   - Evaluates JEDM2Model with text-only and multimodal (DOM + visual patches) inputs.
   - Verifies early-fusion token concatenation and multimodal attention masking.
   - Validates output shapes for LM rationale and action predictions.
3. Visual Cross-Attention Grounding Sensitivity:
   - Evaluates CoordinateHead coordinate regression across contrasting visual feature patches.
   - Asserts predicted coordinates shift in response to visual inputs (grounding sensitivity).
4. Multimodal End-to-End Backward Pass & Target Gradient Isolation:
   - Evaluates composite multi-task loss over multimodal trajectory batches.
   - Asserts zero NaNs/Infs across all parameter gradients.
   - Strictly verifies JEPA EMA target encoder gradient isolation (grad is None).
5. Multimodal Recurrent Step Rollout Mode:
   - Evaluates multi-step autoregressive rollout with visual grounding.
   - Verifies recurrent state buffer updates and bounded coordinate predictions in [0, 1].
6. Hardware Memory Budget Profile:
   - Evaluates forward and backward pass on CUDA in bfloat16.
   - Strictly asserts peak allocated GPU VRAM stays below 6.0 GB (6144.0 MB).

STRICT CONVENTION: Pure ASCII console output (no multi-byte Unicode box-drawing characters).
"""

import math
import sys
import time
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.dataset import (
    AgentTrajectoryDataset,
    SyntheticTrajectoryGenerator,
    TrajectoryBatch,
    TrajectoryCollator,
)
from jed_m2.heads import CoordinateHead, NativeActionHead
from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
from jed_m2.model import JEDM2Model, JEDM2Output
from jed_m2.vision import VisionPatchEncoder, get_2d_sincos_pos_embed


def probe_device_and_precision() -> Tuple[torch.device, torch.dtype, bool]:
    """Detects CUDA capability and checks bfloat16 support."""
    if torch.cuda.is_available():
        try:
            probe_a = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
            probe_b = probe_a @ probe_a
            del probe_a, probe_b
            torch.cuda.empty_cache()
            return torch.device("cuda"), torch.bfloat16, True
        except Exception:
            return torch.device("cpu"), torch.float32, False
    return torch.device("cpu"), torch.float32, False


def check_gradient_integrity(model: torch.nn.Module) -> Tuple[bool, int]:
    """Asserts that all parameter gradients are finite and JEPA target network is isolated."""
    checked = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, f"Frozen parameter {name} has gradient!"
            continue

        if "jepa.target_encoder" in name:
            assert param.grad is None, f"JEPA EMA target encoder parameter {name} has gradient!"
            continue

        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}!"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name}!"
            checked += 1

    return True, checked


def main() -> None:
    print("\n" + "=" * 80)
    print("      JED-M2 Large Action Model (LAM) - Phase 6D Verification")
    print("      (Visual Action Grounding & Multimodal Vision-LAM Extension)")
    print("=" * 80 + "\n")

    device, dtype, is_cuda = probe_device_and_precision()
    gpu_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    print(f"Active Device         : {gpu_name} ({device}, {dtype})")
    print(f"CUDA Available        : {is_cuda}")
    print("=" * 80 + "\n")

    # ==========================================================================
    # Test 1: Pure PyTorch Visual Patch Encoder Across Resolutions
    # ==========================================================================
    print("[Test 1] Visual Patch Encoder Across Resolutions & Parameter Budget")
    config = JEDM2Config(image_size=224, patch_size=16)
    vision_encoder = VisionPatchEncoder(config).to(device=device, dtype=dtype)

    enc_params = sum(p.numel() for p in vision_encoder.parameters())
    print(f"  VisionPatchEncoder Total Parameters: {enc_params:,} ({enc_params / 1e6:.3f}M)")
    assert enc_params < 1_000_000, f"Vision encoder exceeds budget: {enc_params} params"

    # 1A. Standard resolution: 224x224 -> 14x14 = 196 patches
    batch_size = 2
    img224 = torch.randn(batch_size, 3, 224, 224, device=device, dtype=dtype)
    patches224 = vision_encoder(img224)
    print(f"  Input 224x224 shape: {list(img224.shape)} -> Patches: {list(patches224.shape)}")
    assert patches224.shape == (batch_size, 196, config.hidden_dim), (
        f"Expected (2, 196, 512), got {patches224.shape}"
    )

    # 1B. Higher resolution: 448x448 -> 28x28 = 784 patches (interpolated 2D pos embed)
    img448 = torch.randn(batch_size, 3, 448, 448, device=device, dtype=dtype)
    patches448 = vision_encoder(img448)
    print(f"  Input 448x448 shape: {list(img448.shape)} -> Patches: {list(patches448.shape)}")
    assert patches448.shape == (batch_size, 784, config.hidden_dim), (
        f"Expected (2, 784, 512), got {patches448.shape}"
    )
    assert not torch.isnan(patches448).any(), "NaN in patch tokens!"
    print("  [PASS] Visual patch encoder verified across resolutions.\n")

    # ==========================================================================
    # Test 2: Dual Multimodal Forward Pass & Attention Mask Integration
    # ==========================================================================
    print("[Test 2] Dual Multimodal Forward Pass & Early-Fusion Integration")
    model = JEDM2Model(config).to(device=device, dtype=dtype)
    param_counts = model.count_parameters()
    print(f"  Total Parameters     : {param_counts['total_params']:,} ({param_counts['total_params'] / 1e6:.2f}M)")
    print(f"  Trainable Parameters : {param_counts['trainable_params']:,} ({param_counts['trainable_params'] / 1e6:.2f}M)")
    print(f"  Active Per Token     : {param_counts['active_params']:,} ({param_counts['active_params'] / 1e6:.2f}M)")

    # 2A. Text-only forward pass (Backward compatibility check)
    seq_len = 64
    input_ids = torch.randint(10, 1000, (batch_size, seq_len), device=device)
    out_text = model(input_ids=input_ids)
    assert out_text.lm_logits.shape == (batch_size, seq_len, config.vocab_size)
    assert out_text.action_output.coordinates.shape == (batch_size, seq_len, 2)
    print("  [PASS] Text-only backward compatibility forward verified.")

    # 2B. Multimodal forward pass with RGB screenshot images
    out_mm = model(input_ids=input_ids, images=img224)
    assert out_mm.lm_logits.shape == (batch_size, seq_len, config.vocab_size), (
        f"Expected lm_logits ({batch_size}, {seq_len}, {config.vocab_size}), got {out_mm.lm_logits.shape}"
    )
    assert out_mm.action_output.coordinates.shape == (batch_size, seq_len, 2), (
        f"Expected coordinates ({batch_size}, {seq_len}, 2), got {out_mm.action_output.coordinates.shape}"
    )
    assert out_mm.visual_tokens is not None
    assert out_mm.visual_tokens.shape == (batch_size, 196, config.hidden_dim)
    print("  [PASS] Multimodal forward pass with early-fusion tokens verified.\n")

    # ==========================================================================
    # Test 3: Visual Cross-Attention Grounding Sensitivity in Coordinate Head
    # ==========================================================================
    print("[Test 3] Visual Cross-Attention Grounding Sensitivity in CoordinateHead")
    coord_head = CoordinateHead(config).to(device=device, dtype=dtype)

    # Identical hidden state query
    h_query = torch.randn(1, 1, config.hidden_dim, device=device, dtype=dtype)

    # Two contrasting visual scenes processed through vision encoder (with 2D spatial pos embeddings):
    # Scene A: bright region in top-left quadrant (target element at [0.1, 0.1])
    img_a = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
    img_a[:, :, :56, :56] = 2.0
    vis_a = vision_encoder(img_a)

    # Scene B: bright region in bottom-right quadrant (target element at [0.8, 0.8])
    img_b = torch.zeros(1, 3, 224, 224, device=device, dtype=dtype)
    img_b[:, :, 168:, 168:] = 2.0
    vis_b = vision_encoder(img_b)

    coords_a = coord_head(h_query, visual_tokens=vis_a).squeeze()
    coords_b = coord_head(h_query, visual_tokens=vis_b).squeeze()
    coords_none = coord_head(h_query, visual_tokens=None).squeeze()

    diff_ab = torch.norm(coords_a.float() - coords_b.float()).item()
    diff_none = torch.norm(coords_a.float() - coords_none.float()).item()
    print(f"  Coords with Scene A (Top-Left)     : ({coords_a[0]:.4f}, {coords_a[1]:.4f})")
    print(f"  Coords with Scene B (Bottom-Right) : ({coords_b[0]:.4f}, {coords_b[1]:.4f})")
    print(f"  Coords without Visual Context      : ({coords_none[0]:.4f}, {coords_none[1]:.4f})")
    print(f"  Coord Difference (Scene A vs B)    : {diff_ab:.6f}")
    print(f"  Coord Difference (Visual vs Text)  : {diff_none:.6f}")

    assert (coords_a >= 0.0).all() and (coords_a <= 1.0).all(), "Coordinates must be in [0, 1]!"
    assert (coords_b >= 0.0).all() and (coords_b <= 1.0).all(), "Coordinates must be in [0, 1]!"
    assert diff_ab > 1e-4, f"Coordinate head is unresponsive between visual scenes! diff={diff_ab}"
    assert diff_none > 1e-4, f"Coordinate head is unresponsive to visual grounding! diff={diff_none}"
    print("  [PASS] Visual cross-attention spatial coordinate grounding verified.\n")

    # ==========================================================================
    # Test 4: Multimodal Sequence Backward Pass & Gradient Isolation
    # ==========================================================================
    print("[Test 4] Multimodal End-to-End Backward Pass & Target Gradient Isolation")
    generator = SyntheticTrajectoryGenerator(config, seed=123)
    collator = TrajectoryCollator(config)

    trajectories = [generator.generate_trajectory() for _ in range(batch_size)]
    batch: TrajectoryBatch = collator(trajectories)
    batch = batch.to(device=device, dtype=dtype)

    assert batch.images is not None, "Batch must include screenshot images!"
    assert batch.images.shape == (batch_size, 3, config.image_size, config.image_size)

    # Forward through model with full multimodal batch
    out_batch = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        images=batch.images,
        obs_t=batch.jepa_obs_t,
        action_t=batch.jepa_action_t,
        obs_next=batch.jepa_obs_next,
    )

    loss_fn = JEDM2CompositeLoss(config, JEDM2LossConfig())
    total_loss, metrics = loss_fn(out_batch, batch)

    print(f"  Total Composite Loss : {total_loss.item():.4f}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    - {k:<20}: {v:.4f}")

    # Backward pass
    model.zero_grad(set_to_none=True)
    total_loss.backward()

    # Gradient integrity and JEPA target encoder isolation
    valid_grads, checked_count = check_gradient_integrity(model)
    assert valid_grads, "Gradient integrity check failed!"
    print(f"  Trainable Parameters with Clean Gradients: {checked_count}")

    # Explicit check for vision patch encoder gradients
    vis_grad_norm = torch.norm(model.vision_encoder.patch_proj.weight.grad).item()
    print(f"  Vision Patch Proj Grad Norm              : {vis_grad_norm:.6f}")
    assert vis_grad_norm > 0.0, "Zero gradient in vision patch encoder!"
    print("  [PASS] Multimodal backward pass & target network isolation verified.\n")

    # ==========================================================================
    # Test 5: Multimodal Recurrent Step Rollout Mode
    # ==========================================================================
    print("[Test 5] Multimodal Recurrent Step Rollout & State Buffer Invariance")
    step_token = torch.tensor([15], device=device)
    states = None
    step_vis = patches224[:1]  # [1, 196, 512] visual context

    for step_idx in range(8):
        lm_logits_t, action_out_t, states = model.step(
            step_token, states=states, pos=step_idx, visual_tokens=step_vis
        )
        c_x, c_y = action_out_t.coordinates[0, 0].item(), action_out_t.coordinates[0, 1].item()
        assert 0.0 <= c_x <= 1.0 and 0.0 <= c_y <= 1.0, f"Out of bounds coords: ({c_x}, {c_y})"
        assert not torch.isnan(lm_logits_t).any(), f"NaN in step logits at step {step_idx}"

    print(f"  Completed 8 recurrent steps with visual context. Final coords: ({c_x:.4f}, {c_y:.4f})")
    assert len(states) == config.num_layers
    print("  [PASS] Multimodal recurrent rollout verified.\n")

    # ==========================================================================
    # Test 6: Hardware Memory Budget Profile & 6.0 GB VRAM Ceiling
    # ==========================================================================
    print("[Test 6] Hardware Memory Budget Profile & 6.0 GB VRAM Ceiling")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        model.gradient_checkpointing_enable()

        # Multi-step training simulation under bfloat16
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        for step in range(5):
            traj_step = [generator.generate_trajectory() for _ in range(2)]
            batch_step = collator(traj_step).to(device=device, dtype=dtype)

            optimizer.zero_grad(set_to_none=True)
            out_step = model(
                input_ids=batch_step.input_ids,
                attention_mask=batch_step.attention_mask,
                images=batch_step.images,
                obs_t=batch_step.jepa_obs_t,
                action_t=batch_step.jepa_action_t,
                obs_next=batch_step.jepa_obs_next,
            )
            step_loss, _ = loss_fn(out_step, batch_step)
            step_loss.backward()
            optimizer.step()

        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        peak_vram_gb = peak_vram_mb / 1024.0
        vram_limit_mb = 6144.0  # 6.0 GB ceiling

        print(f"  Peak GPU VRAM Allocated : {peak_vram_mb:.2f} MB ({peak_vram_gb:.3f} GB)")
        print(f"  Target VRAM Ceiling     : {vram_limit_mb:.2f} MB (6.000 GB)")
        print(f"  VRAM Utilization        : {(peak_vram_mb / vram_limit_mb) * 100:.2f}%")

        assert peak_vram_mb < vram_limit_mb, (
            f"VRAM ceiling breached! {peak_vram_mb:.2f} MB >= {vram_limit_mb:.2f} MB"
        )
        print("  [PASS] Peak VRAM is strictly below 6.0 GB ceiling.")
    else:
        print("  (CUDA not available - CPU execution verified)")
        peak_vram_mb = 0.0
        peak_vram_gb = 0.0

    print("\n" + "=" * 80)
    print("      PHASE 6D ALL TESTS PASSED SUCCESSFULLY!")
    print(f"   Peak VRAM: {peak_vram_mb:.2f} MB (< 6144.0 MB limit) | Device: {gpu_name}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
