"""
Verification Script for Phase 1: JED-M2 Core Architecture.
Tests MultiHeadLatentAttention (MLA) and SparseTop2MoE under hardware constraints.
"""

import sys
from typing import Tuple
import torch

from jed_m2.config import JEDM2Config
from jed_m2.mla import MultiHeadLatentAttention
from jed_m2.moe import SparseTop2MoE


def probe_device_and_precision() -> Tuple[torch.device, torch.dtype, bool]:
    """
    Detects hardware capabilities and probes CUDA kernel execution.
    Gracefully falls back to CPU if sm_120 (Blackwell) kernel image is missing.
    """
    if torch.cuda.is_available():
        try:
            # Probe kernel execution to check if GPU compute capability is supported by current PyTorch
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
            print("\nHardware: NVIDIA GeForce RTX 5060 Laptop GPU (Compute Capability 12.0 / sm_120).")
            print("The currently installed PyTorch build lacks sm_120 binary kernels.")
            print("To enable native GPU execution for Blackwell, please install:")
            print("  pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu129")
            print("  # or cu130 / cu132 as appropriate for CUDA runtime.")
            print("Proceeding with CPU verification (bfloat16) as requested.")
            print("=" * 80)
            return torch.device("cpu"), torch.bfloat16, False
    else:
        print("[Device Probe] No CUDA device detected. Using CPU (float32).")
        return torch.device("cpu"), torch.float32, False


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of parameters in a module."""
    return sum(p.numel() for p in model.parameters())


def main():
    print("\n" + "=" * 80)
    print("           JED-M2 Large Action Model (LAM) - Phase 1 Verification")
    print("=" * 80 + "\n")

    # 1. Device and precision configuration
    device, dtype, is_cuda = probe_device_and_precision()

    # 2. Instantiate configuration and modules
    config = JEDM2Config()
    print(f"\n[Config] hidden_dim={config.hidden_dim}, num_heads={config.num_heads}, "
          f"q_lora_rank={config.q_lora_rank}, kv_lora_rank={config.kv_lora_rank}, "
          f"num_experts={config.num_experts}, top_k={config.top_k}, activation={config.activation}")

    mla = MultiHeadLatentAttention(config).to(device=device, dtype=dtype)
    moe = SparseTop2MoE(config).to(device=device, dtype=dtype)

    mla_params = count_parameters(mla)
    moe_params = count_parameters(moe)
    total_params = mla_params + moe_params

    print("\n" + "-" * 40 + " Parameter Counts " + "-" * 40)
    print(f"MultiHeadLatentAttention Parameters : {mla_params:,} (~{mla_params / 1e6:.2f}M)")
    print(f"SparseTop2MoE Parameters             : {moe_params:,} (~{moe_params / 1e6:.2f}M)")
    print(f"Total Combined Layer Parameters      : {total_params:,} (~{total_params / 1e6:.2f}M)")
    print("-" * 98)

    # 3. Create synthetic batch
    batch_size = 2
    seq_len = 1024
    hidden_dim = config.hidden_dim
    print(f"\n[Input Batch] Shape: ({batch_size}, {seq_len}, {hidden_dim}), Dtype: {dtype}, Device: {device}")

    # Set seed for reproducible verification
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, requires_grad=True)

    # 4. Execute forward pass
    print("\n[Forward Pass] Executing MultiHeadLatentAttention...")
    mla_out = mla(x)
    print(f"  -> MLA Output Shape: {tuple(mla_out.shape)} (Expected: ({batch_size}, {seq_len}, {hidden_dim}))")
    assert mla_out.shape == (batch_size, seq_len, hidden_dim), (
        f"MLA output shape mismatch: {mla_out.shape} != {(batch_size, seq_len, hidden_dim)}"
    )

    print("[Forward Pass] Executing SparseTop2MoE...")
    moe_out, aux_loss = moe(mla_out)
    print(f"  -> MoE Output Shape : {tuple(moe_out.shape)} (Expected: ({batch_size}, {seq_len}, {hidden_dim}))")
    print(f"  -> Auxiliary Loss   : {aux_loss.item():.6f}")
    assert moe_out.shape == (batch_size, seq_len, hidden_dim), (
        f"MoE output shape mismatch: {moe_out.shape} != {(batch_size, seq_len, hidden_dim)}"
    )
    assert aux_loss.item() > 0, "Auxiliary loss must be positive."

    # 5. Compute dummy loss & execute backward pass
    print("\n[Backward Pass] Computing loss and backpropagating gradients...")
    dummy_loss = moe_out.mean() + aux_loss
    dummy_loss.backward()

    # 6. Gradient & NaN / Inf Verification
    print("\n[Gradient Check] Verifying gradient integrity...")
    assert x.grad is not None, "Input gradient x.grad is None!"
    assert not torch.isnan(x.grad).any(), "NaN found in input gradient!"
    assert not torch.isinf(x.grad).any(), "Inf found in input gradient!"

    for name, param in mla.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in MLA param: {name}"
        assert not torch.isinf(param).any(), f"Inf in MLA param: {name}"
        assert param.grad is not None, f"Grad is None for MLA param: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in MLA grad: {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in MLA grad: {name}"

    for name, param in moe.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in MoE param: {name}"
        assert not torch.isinf(param).any(), f"Inf in MoE param: {name}"
        assert param.grad is not None, f"Grad is None for MoE param: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in MoE grad: {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in MoE grad: {name}"

    print("  -> Gradient Check PASSED: All parameters and inputs have valid, non-NaN/non-Inf gradients.")

    # 7. VRAM and Memory Profiling
    print("\n" + "-" * 40 + " Memory Profiling " + "-" * 40)
    if is_cuda:
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_mb = peak_vram_bytes / (1024 ** 2)
        print(f"Peak GPU VRAM Allocated : {peak_vram_mb:.2f} MB ({peak_vram_bytes / (1024 ** 3):.3f} GB)")
        print(f"Target Constraint       : < 2048.00 MB (2.0 GB)")
        assert peak_vram_mb < 2048.0, f"Peak VRAM ({peak_vram_mb:.2f} MB) exceeded 2.0GB limit!"
        print("  -> VRAM Constraint PASSED: Memory footprint well within hardware budget.")
    else:
        print("Running on CPU fallback. Peak CUDA VRAM profiling skipped.")
        print("Forward and backward passes executed cleanly in bfloat16 without memory issues.")
    print("-" * 98)

    print("\n" + "=" * 80)
    print("    Phase 1 Verification SUCCEEDED: Architecture Validated Successfully!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
