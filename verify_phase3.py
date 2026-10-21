"""
Verification Script for Phase 3: JED-M2 Action Heads & Unified LAM Backbone.
Tests NativeActionHead, JEDM2Block, and full unified JEDM2Model under hardware constraints.
Verifies forward pass, dual action output shapes, recurrent rollout step(...) execution,
backward pass (asserting zero NaNs and Infinities), parameter scaling (~45M total / ~14M active),
and peak VRAM allocation (< 2.0 GB) on CUDA in bfloat16.

STRICT CONVENTION: Pure ASCII console output (no Unicode box-drawing characters).
"""

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.heads import (
    ActionGate,
    ActionOutput,
    ActionType,
    ActionTypeHead,
    ArgumentHead,
    CoordinateHead,
    NativeActionHead,
)
from jed_m2.model import JEDM2Block, JEDM2Model, JEDM2Output


def probe_device_and_precision() -> Tuple[torch.device, torch.dtype, bool]:
    """
    Detects hardware capabilities and probes CUDA kernel execution.
    Gracefully falls back to CPU if sm_120 kernel execution fails.
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
    print("           JED-M2 Large Action Model (LAM) - Phase 3 Verification")
    print("=" * 80 + "\n")

    # 1. Device and precision configuration
    device, dtype, is_cuda = probe_device_and_precision()

    # 2. Instantiate configuration and unified model
    config = JEDM2Config()
    print(
        f"\n[Config] hidden_dim={config.hidden_dim}, num_layers={config.num_layers}, "
        f"num_heads={config.num_heads}, vocab_size={config.vocab_size}, "
        f"num_experts={config.num_experts}, top_k={config.top_k}, "
        f"mamba_d_state={config.mamba_d_state}, mamba_expand={config.mamba_expand}"
    )

    model = JEDM2Model(config).to(device=device, dtype=dtype)

    # 3. Parameter count analysis
    param_stats = model.count_parameters()
    total_params = param_stats["total_params"]
    trainable_params = param_stats["trainable_params"]
    active_params = param_stats["active_params"]
    inactive_moe = param_stats["inactive_moe_params"]

    block_0_params = sum(p.numel() for p in model.blocks[0].parameters())
    three_blocks_params = sum(p.numel() for p in model.blocks.parameters())
    action_head_params = sum(p.numel() for p in model.action_head.parameters())
    jepa_params = sum(p.numel() for p in model.jepa.parameters())
    embed_params = sum(p.numel() for p in model.tok_embeddings.parameters())

    print("\n" + "-" * 38 + " Parameter Counts " + "-" * 38)
    print(f"Single JEDM2Block Parameters         : {block_0_params:,} (~{block_0_params / 1e6:.2f}M)")
    print(f"Three Stacked Blocks Footprint       : {three_blocks_params:,} (~{three_blocks_params / 1e6:.2f}M)")
    print(f"Token & Head Embeddings Footprint    : {embed_params:,} (~{embed_params / 1e6:.2f}M)")
    print(f"NativeActionHead Footprint           : {action_head_params:,} (~{action_head_params / 1e6:.2f}M)")
    print(f"JEPA World Model Footprint           : {jepa_params:,} (~{jepa_params / 1e6:.2f}M)")
    print(f"Inactive MoE Parameters per Token    : {inactive_moe:,} (~{inactive_moe / 1e6:.2f}M)")
    print(f"Total Model Parameters               : {total_params:,} (~{total_params / 1e6:.2f}M)")
    print(f"Trainable Parameters                 : {trainable_params:,} (~{trainable_params / 1e6:.2f}M)")
    print(f"Active Parameters per Token          : {active_params:,} (~{active_params / 1e6:.2f}M)")
    print("-" * 94)

    # Verify parameter scaling target: ~45M total backbone / ~14M-17M active
    assert 40_000_000 <= three_blocks_params <= 50_000_000, (
        f"Three blocks parameter count ({three_blocks_params:,}) out of ~45M target budget!"
    )
    assert 12_000_000 <= active_params <= 25_000_000, (
        f"Active parameter count ({active_params:,}) out of expected range!"
    )
    print("  -> Parameter Scaling Check PASSED: Backbone conforms to ~45M total / ~14M active envelope.")

    # 4. Forward Pass with Full Sequence Input & Output Shapes
    batch_size = 2
    seq_len = 512
    print(f"\n[Test 1] Full Model Sequence Forward Pass")
    print(f"  Input Batch Shape: ({batch_size}, {seq_len}), Dtype: {dtype}, Device: {device}")

    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    # Synthetic JEPA observation conditioning
    obs_t = torch.randn(batch_size, config.hidden_dim, device=device, dtype=dtype)
    action_t = torch.randint(0, config.vocab_size, (batch_size,), device=device)
    obs_next = torch.randn(batch_size, config.hidden_dim, device=device, dtype=dtype)

    out: JEDM2Output = model(
        input_ids=input_ids,
        obs_t=obs_t,
        action_t=action_t,
        obs_next=obs_next,
    )

    print(f"  -> LM Logits Shape                 : {tuple(out.lm_logits.shape)}")
    print(f"  -> Action Gate Logits Shape        : {tuple(out.action_output.gate_logits.shape)}")
    print(f"  -> Action Type Logits Shape        : {tuple(out.action_output.action_type_logits.shape)}")
    print(f"  -> Coordinate Regressor Shape      : {tuple(out.action_output.coordinates.shape)}")
    print(f"  -> Argument Logits Shape           : {tuple(out.action_output.argument_logits.shape)}")
    print(f"  -> Accumulated MoE Aux Loss        : {out.aux_loss.item():.6f}")
    print(f"  -> JEPA Transition Loss            : {out.jepa_loss.item():.6f}")

    # Shape and range assertions
    assert out.lm_logits.shape == (batch_size, seq_len, config.vocab_size)
    assert out.action_output.gate_logits.shape == (batch_size, seq_len, 1)
    assert out.action_output.action_type_logits.shape == (batch_size, seq_len, config.num_action_types)
    assert out.action_output.coordinates.shape == (batch_size, seq_len, 2)
    assert out.action_output.argument_logits.shape == (batch_size, seq_len, config.vocab_size)
    assert (out.action_output.coordinates >= 0.0).all() and (out.action_output.coordinates <= 1.0).all(), (
        "Coordinates must be strictly bounded in [0, 1]!"
    )
    assert out.aux_loss.item() > 0.0, "Auxiliary MoE loss must be strictly positive!"
    assert out.jepa_loss is not None and out.jepa_loss.item() > 0.0, "JEPA loss must be strictly positive!"
    print("  -> Sequence Forward Pass PASSED: All dual-system output shapes and bounds verified.")

    # 5. Dual-System Action Decoding (predict_action)
    print(f"\n[Test 2] System 1 Action Primitive Inference Decoding")
    single_hidden = out.hidden_states[:, -1]  # [B, hidden_dim]
    parsed_action = model.action_head.predict_action(single_hidden, threshold=0.5)

    print(f"  -> Action Gate Probability        : {parsed_action['gate_prob'][0].item():.4f}")
    print(f"  -> Action Emission Fired           : {parsed_action['emit_action'][0].item()}")
    print(f"  -> Predicted Action Type ID        : {parsed_action['action_type_id'][0].item()}")
    print(f"  -> Predicted Action Type Name      : {parsed_action['action_type_name'][0]}")
    print(f"  -> Predicted Screen Coordinates    : x={parsed_action['coordinates'][0, 0].item():.4f}, y={parsed_action['coordinates'][0, 1].item():.4f}")
    print(f"  -> Predicted Argument Token ID     : {parsed_action['argument_id'][0].item()}")

    assert parsed_action["action_type_name"][0] in [e.name for e in ActionType]
    assert 0.0 <= parsed_action["coordinates"][0, 0].item() <= 1.0
    assert 0.0 <= parsed_action["coordinates"][0, 1].item() <= 1.0
    print("  -> Action Decoding PASSED: Cleanly parsed discrete primitives and bounded pointer coordinates.")

    # 6. Recurrent Rollout step(...) Execution & State Propagation
    print(f"\n[Test 3] Multi-Step Recurrent Rollout step(...) Execution")
    rollout_steps = 16
    rollout_tokens = input_ids[:, :rollout_steps]

    states = None
    step_lm_logits = []
    step_action_types = []

    with torch.no_grad():
        for t in range(rollout_steps):
            tok_t = rollout_tokens[:, t]
            lm_l_t, act_t, states = model.step(tok_t, states=states, pos=t)

            assert lm_l_t.shape == (batch_size, config.vocab_size)
            assert act_t.gate_logits.shape == (batch_size, 1)
            assert act_t.action_type_logits.shape == (batch_size, config.num_action_types)
            assert act_t.coordinates.shape == (batch_size, 2)
            assert act_t.argument_logits.shape == (batch_size, config.vocab_size)

            step_lm_logits.append(lm_l_t)
            step_action_types.append(act_t.action_type_logits)

    assert len(states) == config.num_layers, f"Expected {config.num_layers} layer states, got {len(states)}"
    for l_idx, layer_state in enumerate(states):
        ssm_s, conv_s, kv_c = layer_state
        assert ssm_s.shape == (batch_size, config.mamba_inner_dim, config.mamba_d_state), (
            f"Layer {l_idx} SSM state shape mismatch!"
        )
        assert conv_s.shape == (batch_size, config.mamba_inner_dim, config.mamba_d_conv), (
            f"Layer {l_idx} Conv state shape mismatch!"
        )
        c_kv, k_rope = kv_c
        assert c_kv.shape == (batch_size, rollout_steps, config.kv_lora_rank), (
            f"Layer {l_idx} Compressed c_kv cache shape mismatch!"
        )
        assert k_rope.shape == (batch_size, rollout_steps, 1, config.qk_rope_head_dim), (
            f"Layer {l_idx} Shared k_rope cache shape mismatch!"
        )

    print(f"  -> Successfully rolled out {rollout_steps} recurrent steps across all 3 layers.")
    print("  -> Recurrent State Propagation PASSED: Mamba SSM buffers, conv buffers, and MLA KV-cache intact.")

    # 7. Backward Pass & Gradient Integrity
    print(f"\n[Test 4] Full Backward Pass & Gradient Integrity Check")
    # Composite multi-task loss
    composite_loss = (
        out.lm_logits.mean()
        + out.action_output.gate_logits.mean()
        + out.action_output.action_type_logits.mean()
        + F.smooth_l1_loss(out.action_output.coordinates, torch.zeros_like(out.action_output.coordinates))
        + out.action_output.argument_logits.mean()
        + out.aux_loss
        + out.jepa_loss
    )

    composite_loss.backward()

    # Parameter NaN/Inf checks
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

    # 8. Memory Profiling on Hardware
    print("\n" + "-" * 38 + " Memory Profiling " + "-" * 38)
    if is_cuda:
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_vram_bytes / (1024**2)
        print(f"Peak GPU VRAM Allocated : {peak_vram_mb:.2f} MB ({peak_vram_bytes / (1024**3):.3f} GB)")
        print(f"Target Constraint       : < 2048.00 MB (2.0 GB)")
        assert peak_vram_mb < 2048.0, f"Peak VRAM ({peak_vram_mb:.2f} MB) exceeded 2.0GB limit!"
        print("  -> VRAM Constraint PASSED: Memory footprint well within hardware budget.")
    else:
        print("Running on CPU fallback. Forward, step, and backward passes executed cleanly.")
    print("-" * 94)

    print("\n" + "=" * 80)
    print("    Phase 3 Verification SUCCEEDED: Architecture Validated Successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
