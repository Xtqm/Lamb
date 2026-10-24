"""
Verification Script for Phase 2: JED-M2 Sequential Recurrence & Latent World Model.
Tests MambaBlock (Pure PyTorch SSM) and JEPAWorldModel under hardware constraints.
Verifies forward pass, step equivalence, backward pass (no NaNs/Infs), and peak VRAM in bfloat16.
"""

import os
import sys
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)



from typing import Tuple
import torch

from jed_m2.config import JEDM2Config
from jed_m2.mamba import MambaBlock
from jed_m2.jepa import JEPAWorldModel
from jed_m2.mla import MultiHeadLatentAttention
from jed_m2.moe import SparseTop2MoE


def probe_device_and_precision() -> Tuple[torch.device, torch.dtype, bool]:
    """
    Detects hardware capabilities and probes CUDA kernel execution.
    Gracefully falls back to CPU if sm_120 kernel image is missing.
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


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Return total or trainable number of parameters in a module."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def main():
    print("\n" + "=" * 80)
    print("           JED-M2 Large Action Model (LAM) - Phase 2 Verification")
    print("=" * 80 + "\n")

    # 1. Device and precision configuration
    device, dtype, is_cuda = probe_device_and_precision()

    # 2. Instantiate configuration and modules
    config = JEDM2Config()
    print(
        f"\n[Config] hidden_dim={config.hidden_dim}, mamba_expand={config.mamba_expand}, "
        f"mamba_d_state={config.mamba_d_state}, mamba_d_conv={config.mamba_d_conv}, "
        f"jepa_latent_dim={config.jepa_latent_dim}, jepa_ema_decay={config.jepa_ema_decay}"
    )

    mamba = MambaBlock(config).to(device=device, dtype=dtype)
    jepa = JEPAWorldModel(config).to(device=device, dtype=dtype)

    # For cumulative architecture parameter tracking
    mla = MultiHeadLatentAttention(config).to(device=device, dtype=dtype)
    moe = SparseTop2MoE(config).to(device=device, dtype=dtype)

    mamba_params = count_parameters(mamba)
    jepa_total_params = count_parameters(jepa)
    jepa_trainable_params = count_parameters(jepa, trainable_only=True)
    mla_params = count_parameters(mla)
    moe_params = count_parameters(moe)

    phase2_total = mamba_params + jepa_total_params
    cumulative_total = mla_params + moe_params + mamba_params + jepa_trainable_params

    print("\n" + "-" * 38 + " Parameter Counts " + "-" * 38)
    print(f"MambaBlock Parameters                : {mamba_params:,} (~{mamba_params / 1e6:.2f}M)")
    print(f"JEPAWorldModel Total Parameters      : {jepa_total_params:,} (~{jepa_total_params / 1e6:.2f}M)")
    print(f"  |- Trainable Online & Predictor    : {jepa_trainable_params:,} (~{jepa_trainable_params / 1e6:.2f}M)")
    print(f"  |- EMA Target (Non-trainable)      : {jepa_total_params - jepa_trainable_params:,}")
    print(f"Phase 2 Total Module Footprint       : {phase2_total:,} (~{phase2_total / 1e6:.2f}M)")
    print(f"Cumulative Model Footprint (Phases 1-2): {cumulative_total:,} (~{cumulative_total / 1e6:.2f}M)")
    print("-" * 94)

    # 3. Mamba Sequence Forward Pass
    batch_size = 2
    seq_len = 1024
    hidden_dim = config.hidden_dim
    print(f"\n[Test 1] Mamba Sequence Forward Pass")
    print(f"  Input Batch Shape: ({batch_size}, {seq_len}, {hidden_dim}), Precision: {dtype}")

    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)

    mamba_out = mamba(x)
    print(f"  -> Mamba Output Shape: {tuple(mamba_out.shape)} (Expected: ({batch_size}, {seq_len}, {hidden_dim}))")
    assert mamba_out.shape == (batch_size, seq_len, hidden_dim), "Mamba output shape mismatch!"

    # 4. Mamba Step Recurrent Equivalence Verification
    print(f"\n[Test 2] Mamba Recurrent step(...) Mathematical Equivalence")
    eval_len = 64
    x_eval = x[:, :eval_len].detach()

    with torch.no_grad():
        # Evaluate full sequence mode on slice
        seq_slice_out = mamba(x_eval)

        # Rollout step mode token-by-token
        ssm_state, conv_state = None, None
        step_outputs = []
        for t in range(eval_len):
            out_t, ssm_state, conv_state = mamba.step(x_eval[:, t], ssm_state, conv_state)
            step_outputs.append(out_t)
        step_slice_out = torch.stack(step_outputs, dim=1)

        diff = (seq_slice_out - step_slice_out).abs().max().item()
        mean_diff = (seq_slice_out - step_slice_out).abs().mean().item()
        print(f"  -> Max absolute difference across {eval_len} steps: {diff:.6e}")
        print(f"  -> Mean absolute difference across {eval_len} steps: {mean_diff:.6e}")
        assert diff < 0.01, f"Step equivalence mismatch exceeded tolerance: {diff} >= 0.01"
        print("  -> Step Equivalence Check PASSED: O(1) step(...) strictly matches sequence forward mode.")

    # 5. Mamba Backward Pass & Gradient Integrity
    print(f"\n[Test 3] Mamba Backward Pass & Gradient Integrity")
    mamba_loss = mamba_out.mean()
    mamba_loss.backward()

    assert x.grad is not None, "Input gradient x.grad is None!"
    assert not torch.isnan(x.grad).any(), "NaN in Mamba input gradient!"
    assert not torch.isinf(x.grad).any(), "Inf in Mamba input gradient!"

    for name, param in mamba.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in Mamba param: {name}"
        assert not torch.isinf(param).any(), f"Inf in Mamba param: {name}"
        assert param.grad is not None, f"Grad is None for Mamba param: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in Mamba grad: {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in Mamba grad: {name}"

    print("  -> Mamba Gradient Check PASSED: All parameters have non-NaN, non-Inf gradients.")

    # 6. JEPA Forward Pass & Consistency Loss
    print(f"\n[Test 4] JEPA Latent World Model Forward Pass & Consistency Loss")
    jepa_batch = 4
    obs_t = torch.randn(jepa_batch, hidden_dim, device=device, dtype=dtype, requires_grad=True)
    action_t = torch.randn(jepa_batch, config.jepa_action_dim, device=device, dtype=dtype)
    obs_next = torch.randn(jepa_batch, hidden_dim, device=device, dtype=dtype)

    z_pred, z_target, jepa_loss, metrics = jepa(obs_t, action_t, obs_next)

    print(f"  -> Predicted Latent Shape (z_hat) : {tuple(z_pred.shape)}")
    print(f"  -> Target Latent Shape (z_target) : {tuple(z_target.shape)}")
    print(f"  -> Composite JEPA Loss            : {jepa_loss.item():.6f}")
    print(f"  -> Smooth L1 Component            : {metrics['loss_smooth_l1'].item():.6f}")
    print(f"  -> Cosine Distance Component      : {metrics['loss_cos_dist'].item():.6f}")
    print(f"  -> Latent Cosine Similarity       : {metrics['cos_sim'].item():.6f}")

    assert z_pred.shape == (jepa_batch, config.jepa_latent_dim), "z_pred shape mismatch!"
    assert z_target.shape == (jepa_batch, config.jepa_latent_dim), "z_target shape mismatch!"
    assert not z_target.requires_grad, "z_target should have requires_grad=False!"
    assert jepa_loss.item() > 0, "JEPA loss must be strictly positive!"

    # 7. JEPA Discrete Action Conditioning
    print(f"\n[Test 5] JEPA Discrete Action Conditioning")
    discrete_actions = torch.randint(0, config.vocab_size, (jepa_batch,), device=device)
    z_pred_discrete = jepa.predict(jepa.encode(obs_t), discrete_actions)
    assert z_pred_discrete.shape == (jepa_batch, config.jepa_latent_dim)
    print("  -> Discrete Action Conditioning PASSED: successfully handled tokenized action IDs.")

    # 8. JEPA Backward Pass & Target Gradient Isolation
    print(f"\n[Test 6] JEPA Backward Pass & Target Network Gradient Isolation")
    jepa_loss.backward()

    assert obs_t.grad is not None, "obs_t gradient is None!"
    assert not torch.isnan(obs_t.grad).any(), "NaN in obs_t grad!"
    assert not torch.isinf(obs_t.grad).any(), "Inf in obs_t grad!"

    # Online encoder parameters must have valid gradients
    for name, param in jepa.encoder.named_parameters():
        if "vis" in name:
            continue
        assert param.grad is not None, f"Missing grad for online encoder param: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in encoder grad: {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in encoder grad: {name}"

    # Predictor parameters must have valid gradients
    for name, param in jepa.predictor.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN in predictor grad: {name}"
            assert not torch.isinf(param.grad).any(), f"Inf in predictor grad: {name}"

    # Target encoder parameters must strictly have NO gradients (isolated)
    for name, param in jepa.target_encoder.named_parameters():
        assert param.grad is None, f"Target encoder param {name} received unwanted gradient!"

    print("  -> JEPA Gradient Check PASSED: Gradients isolated cleanly from target network.")

    # 9. JEPA Momentum Target Update
    print(f"\n[Test 7] JEPA Target Network Momentum Update")
    target_sample_before = list(jepa.target_encoder.encoder.parameters())[0].clone()
    with torch.no_grad():
        list(jepa.encoder.parameters())[0].add_(torch.ones_like(target_sample_before))
    jepa.update_target_network()
    target_sample_after = list(jepa.target_encoder.encoder.parameters())[0]

    assert not torch.equal(target_sample_before, target_sample_after), "Target parameters did not update!"
    assert not torch.isnan(target_sample_after).any(), "NaN in target parameter after EMA update!"
    print(f"  -> Target Network EMA Update PASSED (tau={jepa.ema_decay}).")

    # 10. Memory Profiling on Hardware
    print("\n" + "-" * 38 + " Memory Profiling " + "-" * 38)
    if is_cuda:
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_vram_bytes / (1024**2)
        print(f"Peak GPU VRAM Allocated : {peak_vram_mb:.2f} MB ({peak_vram_bytes / (1024**3):.3f} GB)")
        print(f"Target Constraint       : < 6144.00 MB (6.0 GB)")
        assert peak_vram_mb < 6144.0, f"Peak VRAM ({peak_vram_mb:.2f} MB) exceeded 6.0GB limit!"
        print("  -> VRAM Constraint PASSED: Memory footprint well within hardware budget.")
    else:
        print("Running on CPU fallback. Forward, step, and backward passes executed cleanly.")
    print("-" * 94)

    print("\n" + "=" * 80)
    print("    Phase 2 Verification SUCCEEDED: Architecture Validated Successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
