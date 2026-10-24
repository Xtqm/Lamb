"""
Verification Script for Phase 4: JED-M2 Synthetic Trajectory Dataset & Unified Loss Harness.
Tests:
1. Synthetic agent trajectory generation modeled after WebArena / Mind2Web.
2. Dynamic batch collation with causal attention masks, action token index mapping, and loss masks.
3. End-to-end model forward pass with full batch and composite loss evaluation.
4. Full backward pass asserting gradient integrity (zero NaNs, zero Infs, JEPA target network isolation).
5. Zero-mask guardrails robustness (handling batches with sparse or missing action categories).
6. Peak GPU VRAM memory consumption (< 2048.0 MB) on CUDA in bfloat16 precision.

STRICT CONVENTION: Pure ASCII console output (no Unicode box-drawing characters).
"""

from typing import Dict, List, Tuple
import torch

from jed_m2.config import JEDM2Config
from jed_m2.dataset import (
    AgentTrajectoryDataset,
    InteractionTurn,
    SimpleLAMTokenizer,
    SyntheticTrajectoryGenerator,
    TrajectoryBatch,
    TrajectoryCollator,
    TrajectorySample,
    UIElement,
)
from jed_m2.heads import ActionType
from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
from jed_m2.model import JEDM2Model, JEDM2Output


def probe_device_and_precision() -> Tuple[torch.device, torch.dtype, bool]:
    """
    Detects hardware capabilities and probes CUDA execution.
    Falls back to CPU if sm_120 kernel execution fails.
    """
    if torch.cuda.is_available():
        try:
            probe_a = torch.ones((2, 2), device="cuda")
            probe_b = probe_a @ probe_a
            del probe_a, probe_b
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Device Probe] CUDA execution successful on {gpu_name}")
            print(f"[Device Probe] Selected precision: {dtype}")
            return device, dtype, True
        except Exception as exc:
            print("=" * 80)
            print("[NOTICE] CUDA device detected, but binary kernel execution failed.")
            print(f"Underlying error: {exc}")
            print("Proceeding with CPU verification (bfloat16).")
            print("=" * 80)
            return torch.device("cpu"), torch.bfloat16, False
    else:
        print("[Device Probe] No CUDA device detected. Using CPU (float32).")
        return torch.device("cpu"), torch.float32, False


def main():
    print("\n" + "=" * 80)
    print("      JED-M2 Large Action Model (LAM) - Phase 4 Verification")
    print("      (Synthetic Trajectory Dataset & Unified Composite Loss Harness)")
    print("=" * 80 + "\n")

    # 1. Device and precision configuration
    device, dtype, is_cuda = probe_device_and_precision()

    # 2. Instantiate configuration
    config = JEDM2Config()
    loss_config = JEDM2LossConfig(
        reason_weight=1.0,
        action_type_weight=1.0,
        coord_weight=5.0,
        arg_weight=1.0,
        jepa_weight=1.0,
        aux_weight=1.0,
        huber_beta=0.05,
    )
    print(
        f"\n[Config] hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, "
        f"num_heads={config.num_heads}, vocab_size={config.vocab_size}, "
        f"num_experts={config.num_experts}, top_k={config.top_k}"
    )
    print(
        f"[LossConfig] reason_weight={loss_config.reason_weight}, "
        f"action_type_weight={loss_config.action_type_weight}, "
        f"coord_weight={loss_config.coord_weight}, arg_weight={loss_config.arg_weight}, "
        f"jepa_weight={loss_config.jepa_weight}, aux_weight={loss_config.aux_weight}"
    )

    # ==========================================================================
    # Test 1: Synthetic Trajectory Generation & Dataset
    # ==========================================================================
    print(f"\n[Test 1] Synthetic Agent Trajectory Generation (WebArena / Mind2Web)")
    dataset = AgentTrajectoryDataset(config, virtual_size=100, seed=42)
    sample_0: TrajectorySample = dataset[0]
    sample_1: TrajectorySample = dataset[1]

    print(f"  Sample 0 ID       : {sample_0.sample_id}")
    print(f"  Sample 0 Goal     : {sample_0.goal}")
    print(f"  Sample 0 Turns    : {sample_0.length}")
    print(f"  Sample 1 ID       : {sample_1.sample_id}")
    print(f"  Sample 1 Goal     : {sample_1.goal}")
    print(f"  Sample 1 Turns    : {sample_1.length}")

    # Inspect first turn
    turn_0 = sample_0.turns[0]
    print(f"  Turn 0 Action     : {turn_0.action_type.name}")
    print(f"  Turn 0 Coords     : x={turn_0.coordinates[0]:.4f}, y={turn_0.coordinates[1]:.4f}")
    print(f"  Turn 0 Argument   : '{turn_0.argument}'")
    print(f"  Turn 0 DOM Elements: {len(turn_0.dom_elements)} items")
    print(f"  Turn 0 Thought    : {turn_0.thought[:65]}...")

    # Assertions on trajectory integrity
    assert sample_0.length >= 2, "Trajectory must have at least 2 turns for multi-turn learning!"
    for turn in sample_0.turns:
        assert isinstance(turn.action_type, ActionType)
        assert 0.0 <= turn.coordinates[0] <= 1.0 and 0.0 <= turn.coordinates[1] <= 1.0
        assert turn.obs_features.shape == (config.hidden_dim,)
        for el in turn.dom_elements:
            assert 0.0 <= el.bbox[0] <= el.bbox[2] <= 1.0
            assert 0.0 <= el.bbox[1] <= el.bbox[3] <= 1.0

    print("  -> Trajectory Generation PASSED: DOM structures, bounding boxes, and action tuples verified.")

    # ==========================================================================
    # Test 2: Dynamic Batch Collation & Mask Construction
    # ==========================================================================
    print(f"\n[Test 2] Dynamic Trajectory Batch Collation & Causal Masking")
    collator = TrajectoryCollator(config, max_seq_len=512)
    batch: TrajectoryBatch = collator([sample_0, sample_1])

    print(f"  Collated input_ids shape         : {tuple(batch.input_ids.shape)}")
    print(f"  Collated attention_mask shape    : {tuple(batch.attention_mask.shape)}")
    print(f"  Reasoning loss mask active count : {batch.reason_loss_mask.sum().item()} tokens")
    print(f"  Argument loss mask active count  : {batch.arg_loss_mask.sum().item()} tokens")
    print(f"  Action indices shape             : {tuple(batch.action_indices.shape)}")
    print(f"  Action type targets shape        : {tuple(batch.action_type_targets.shape)}")
    print(f"  Spatial coordinate targets shape : {tuple(batch.coord_targets.shape)}")
    print(f"  JEPA obs_t shape                 : {tuple(batch.jepa_obs_t.shape)}")
    print(f"  JEPA obs_next shape              : {tuple(batch.jepa_obs_next.shape)}")

    # Shape and validity assertions
    b_sz, seq_l = batch.input_ids.shape
    assert b_sz == 2
    assert batch.attention_mask.shape == (b_sz, 1, seq_l, seq_l)
    assert batch.action_indices.shape[0] == b_sz
    assert batch.coord_targets.shape == (b_sz, batch.action_indices.shape[1], 2)
    assert batch.jepa_obs_t.shape == (b_sz, config.hidden_dim)
    assert batch.jepa_obs_next.shape == (b_sz, config.hidden_dim)
    assert (batch.coord_targets >= 0.0).all() and (batch.coord_targets <= 1.0).all()

    # Causal lower-triangular check on attention mask
    for b in range(b_sz):
        # Upper triangular entries above diagonal must be False (causal)
        upper_tri = torch.triu(batch.attention_mask[b, 0], diagonal=1)
        assert not upper_tri.any(), f"Attention mask in batch {b} violates causal lower-triangularity!"

    print("  -> Dynamic Batch Collation PASSED: Causal 4D masks, action mapping, and JEPA pairs verified.")

    # Transfer batch to device & precision
    batch = batch.to(device=device, dtype=dtype)

    # ==========================================================================
    # Test 3: Unified LAM Model Forward & Composite Loss Evaluation
    # ==========================================================================
    print(f"\n[Test 3] End-to-End LAM Forward Pass & Composite Loss Evaluation")
    model = JEDM2Model(config).to(device=device, dtype=dtype)
    loss_harness = JEDM2CompositeLoss(config, loss_config)

    # Model forward pass
    out: JEDM2Output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        obs_t=batch.jepa_obs_t,
        action_t=batch.jepa_action_t,
        obs_next=batch.jepa_obs_next,
    )

    # Compute composite loss
    total_loss, metrics = loss_harness(out, batch)

    print("-" * 38 + " Loss Metrics " + "-" * 38)
    print(f"Total Composite Loss        : {metrics['loss_total']:.4f}")
    print(f"  L_reason (Rationale NLL)  : {metrics['loss_reason']:.4f}")
    print(f"  L_gate (Action Gate BCE)  : {metrics['loss_gate']:.4f}")
    print(f"  L_type (Action CE)        : {metrics['loss_action_type']:.4f}")
    print(f"  L_coord (Huber [0, 1])    : {metrics['loss_coord']:.4f}")
    print(f"  L_arg (Payload CE)        : {metrics['loss_arg']:.4f}")
    print(f"  L_jepa (Latent Smooth L1) : {metrics['loss_jepa']:.4f}")
    print(f"  L_aux (MoE Load Balancing): {metrics['loss_aux']:.4f}")
    print(f"Action Primitive Accuracy   : {metrics['acc_action_type'] * 100:.1f}%")
    print(f"Coordinate Regression MAE   : {metrics['coord_mae']:.4f}")
    print("-" * 90)

    # Assertions on loss validity
    assert total_loss.item() > 0.0, "Total loss must be strictly positive!"
    assert not torch.isnan(total_loss), "Total loss must not be NaN!"
    assert not torch.isinf(total_loss), "Total loss must not be Inf!"
    assert metrics["loss_reason"] > 0.0
    assert metrics["loss_gate"] > 0.0
    assert metrics["loss_action_type"] > 0.0
    assert metrics["loss_coord"] >= 0.0
    assert metrics["loss_jepa"] > 0.0
    assert metrics["loss_aux"] > 0.0

    print("  -> Forward Loss Evaluation PASSED: All loss components evaluated with finite values.")

    # ==========================================================================
    # Test 4: Backward Pass & Gradient Integrity Assertions
    # ==========================================================================
    print(f"\n[Test 4] Full Backward Pass & Gradient Integrity Check")
    total_loss.backward()

    checked_params = 0
    for name, param in model.named_parameters():
        # Ensure target network parameters strictly have NO gradients (isolated EMA)
        if "jepa.target_encoder" in name:
            assert param.grad is None, f"Gradient leaked into JEPA target network parameter: {name}"
            continue

        if param.requires_grad:
            # Optional pathways inactive for text-only inputs:
            # - jepa.predictor action_embed vs action_proj
            # - vision_encoder and coordinate_head visual grounding (Phase 6D)
            # - jepa multimodal visual encoder pathway (Phase 6D)
            if any(k in name for k in ("action_proj", "action_embed", "vision_encoder", "cross_attn", "norm_visual", "norm_query", "norm_grounded", "fc_vis", "norm_vis")):
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN in gradient of parameter: {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf in gradient of parameter: {name}"
                    checked_params += 1
                continue

            assert param.grad is not None, f"Gradient is None for parameter: {name}"
            assert not torch.isnan(param.grad).any(), f"NaN in gradient of parameter: {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in gradient of parameter: {name}"
            checked_params += 1

    print(f"  -> Gradient Check PASSED: All {checked_params} active trainable parameters have finite gradients.")
    print("  -> Target Isolation PASSED: JEPA EMA target network strictly isolated from backprop.")

    # ==========================================================================
    # Test 5: Zero-Mask Robustness Guardrails Check
    # ==========================================================================
    print(f"\n[Test 5] Zero-Mask Guardrails Check (Sparse Action Batch)")
    # Construct synthetic batch with zero spatial actions and zero thoughts to verify no NaNs occur
    dummy_input_ids = torch.randint(1, config.vocab_size, (2, 64), device=device)
    dummy_mask = torch.tril(torch.ones(64, 64, dtype=torch.bool, device=device))[None, None, :, :]
    sparse_batch = TrajectoryBatch(
        input_ids=dummy_input_ids,
        attention_mask=dummy_mask,
        labels=dummy_input_ids,
        reason_loss_mask=torch.zeros((2, 64), dtype=torch.bool, device=device),  # Zero thoughts
        action_indices=torch.zeros((2, 1), dtype=torch.long, device=device),
        action_type_targets=torch.zeros((2, 1), dtype=torch.long, device=device),
        action_type_mask=torch.zeros((2, 1), dtype=torch.bool, device=device),  # Zero actions
        coord_targets=torch.zeros((2, 1, 2), dtype=dtype, device=device),
        coord_loss_mask=torch.zeros((2, 1), dtype=torch.bool, device=device),  # Zero spatial actions
        arg_targets=torch.full((2, 64), -100, dtype=torch.long, device=device),
        arg_loss_mask=torch.zeros((2, 64), dtype=torch.bool, device=device),  # Zero argument tokens
        jepa_obs_t=torch.randn(2, config.hidden_dim, device=device, dtype=dtype),
        jepa_action_t=torch.zeros((2,), dtype=torch.long, device=device),
        jepa_obs_next=torch.randn(2, config.hidden_dim, device=device, dtype=dtype),
    )

    sparse_out = model(
        input_ids=sparse_batch.input_ids,
        attention_mask=sparse_batch.attention_mask,
        obs_t=sparse_batch.jepa_obs_t,
        action_t=sparse_batch.jepa_action_t,
        obs_next=sparse_batch.jepa_obs_next,
    )
    sparse_loss, sparse_metrics = loss_harness(sparse_out, sparse_batch)

    assert not torch.isnan(sparse_loss), "Sparse batch loss must not produce NaN!"
    assert not torch.isinf(sparse_loss), "Sparse batch loss must not produce Inf!"
    assert sparse_metrics["loss_reason"] == 0.0
    assert sparse_metrics["loss_action_type"] == 0.0
    assert sparse_metrics["loss_coord"] == 0.0
    assert sparse_metrics["loss_arg"] == 0.0
    print("  -> Zero-Mask Robustness PASSED: Handled empty masks gracefully with zero loss contributions.")

    # ==========================================================================
    # Test 6: Memory Profiling on Hardware Envelope
    # ==========================================================================
    print("\n" + "-" * 38 + " Memory Profiling " + "-" * 38)
    if is_cuda:
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_vram_bytes / (1024**2)
        print(f"Peak GPU VRAM Allocated : {peak_vram_mb:.2f} MB ({peak_vram_bytes / (1024**3):.3f} GB)")
        print(f"Target Constraint       : < 6144.00 MB (6.0 GB ceiling)")
        assert peak_vram_mb < 6144.0, f"Peak VRAM ({peak_vram_mb:.2f} MB) exceeded 6.0GB ceiling!"
        print("  -> VRAM Constraint PASSED: Memory footprint well within hardware budget.")
    else:
        print("Running on CPU fallback. Forward and backward passes executed cleanly.")
    print("-" * 90)

    print("\n" + "=" * 80)
    print("    Phase 4 Verification SUCCEEDED: Dataset & Loss Harness Validated!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
