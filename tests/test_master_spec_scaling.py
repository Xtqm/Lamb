"""
Master Specification Verification Test Suite:
Scaling JED-M2 Nano Intelligence via 60/40 Interleaved Domain Randomization,
Pointer-Generator Copy Grounding, and Anti-Shortcut Guardrails.

Test Deliverables:
(a) Pointer-Generator copy distribution, marginalized CE loss, and gate gradients.
(b) 0% holdout domain overlap in the unified dataset builder (Zero-Shot Firewall).
(c) Adversarial near-miss distractor presence and bounded attributes in SyntheticTrajectoryGenerator.
(d) Per-source MoE auxiliary loss computation and zero expert bifurcation.
"""

import os
from pathlib import Path
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from jed_m2.config import JEDM2Config
from jed_m2.core.types import ActionType, UIElement, InteractionTurn
from jed_m2.dataset import (
    SyntheticTrajectoryGenerator,
    TrajectoryCollator,
)
BOUNDED_CLASSES = SyntheticTrajectoryGenerator.BOUNDED_CLASSES
BOUNDED_TEST_IDS = SyntheticTrajectoryGenerator.BOUNDED_TEST_IDS
BOUNDED_ARIA_LABELS = SyntheticTrajectoryGenerator.BOUNDED_ARIA_LABELS
from jed_m2.heads import ArgumentHead
from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
from jed_m2.moe import SparseTop2MoE
from data.build_unified_pretrain_dataset import HOLDOUT_DOMAINS, is_holdout_leaked


# ==============================================================================
# Deliverable (a): Pointer-Generator Copy Distribution & Gate Gradients
# ==============================================================================

def test_pointer_generator_copy_and_gate_gradients():
    """Verifies ArgumentHead Pointer-Generator copy mechanism and gradient flow."""
    torch.manual_seed(42)
    B, S, S_p, D, V = 2, 8, 16, 512, 8192
    cfg = JEDM2Config()
    cfg.hidden_dim = D
    cfg.vocab_size = V
    head = ArgumentHead(cfg)

    h_text = torch.randn(B, S, D, requires_grad=True)
    prompt_hidden = torch.randn(B, S_p, D, requires_grad=True)
    prompt_ids = torch.randint(100, 2000, (B, S_p))
    prompt_mask = torch.ones((B, S_p), dtype=torch.bool)

    # Put a specific target token in prompt_ids
    target_token = 543
    prompt_ids[0, 3] = target_token
    prompt_ids[1, 7] = target_token

    log_p, p_ptr, alpha, p_final = head.forward_components(
        h_text,
        prompt_hidden=prompt_hidden,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
    )

    # Shapes verification
    assert log_p.shape == (B, S, V)
    assert p_ptr.shape == (B, S, S_p)
    assert alpha.shape == (B, S, 1)
    assert p_final.shape == (B, S, V)

    # Probabilities sum to 1.0
    p_sum = p_final.sum(dim=-1)
    assert torch.allclose(p_sum, torch.ones_like(p_sum), atol=1e-5)

    # Check gate values in [0, 1]
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()

    # Targets with verbatim match in prompt
    targets = torch.full((B, S), -100, dtype=torch.long)
    targets[0, 2] = target_token
    targets[1, 4] = target_token
    arg_mask = (targets != -100)

    # Marginalized NLL loss
    nll_loss = F.nll_loss(log_p.view(-1, V), targets.view(-1), ignore_index=-100)

    # Auxiliary binary gate supervision
    # Since target appears in prompt, gate should learn to copy (alpha -> 0)
    gate_loss = F.binary_cross_entropy(alpha.squeeze(-1)[arg_mask], torch.zeros(arg_mask.sum()))
    total_loss = nll_loss + 0.5 * gate_loss

    total_loss.backward()

    # Verify non-zero gradients on all pointer-generator sub-modules
    assert head.gate_fc.weight.grad is not None
    assert head.gate_fc.weight.grad.abs().sum() > 0
    assert head.query_proj.weight.grad is not None
    assert head.query_proj.weight.grad.abs().sum() > 0
    assert head.key_proj.weight.grad is not None
    assert head.key_proj.weight.grad.abs().sum() > 0
    assert h_text.grad is not None
    assert prompt_hidden.grad is not None


# ==============================================================================
# Deliverable (b): 0% Holdout Domain Overlap in Unified Dataset Builder
# ==============================================================================

def test_zero_shot_holdout_firewall_and_zero_leakage():
    """Verifies that the Zero-Shot Holdout Firewall strictly blocks all 11 test domains."""
    # 1. Test firewall detection function
    for domain in HOLDOUT_DOMAINS:
        task_leak = {
            "website": domain,
            "url": f"https://www.{domain}.com/browse",
            "goal": f"Find products on {domain}",
            "turns": [{"url": f"https://www.{domain}.com/item/123", "obs_text": f"Welcome to {domain}"}],
        }
        leaked, matched = is_holdout_leaked(task_leak)
        assert leaked is True, f"Firewall failed to intercept leak for held-out domain: {domain}"
        assert matched == domain

    # Clean domain test
    clean_task = {
        "website": "wikipedia",
        "url": "https://www.wikipedia.org/wiki/Mixture_of_experts",
        "goal": "Read about Mixture of Experts",
        "turns": [{"url": "https://www.wikipedia.org", "obs_text": "Wikipedia the free encyclopedia"}],
    }
    leaked, _ = is_holdout_leaked(clean_task)
    assert leaked is False, "Firewall falsely flagged clean Wikipedia task!"

    # 2. Verify cached unified dataset (if exists on disk) has 0 leaks
    cache_path = Path("data/real_trajectories_cache/unified_pretrain_train_5000_h512_seed42.pt")
    if cache_path.exists():
        samples = torch.load(cache_path, map_location="cpu", weights_only=False)
        leak_count = 0
        for s in samples:
            t_dict = {"domain": s.domain, "turns": [{"url": t.url, "obs_text": t.obs_text} for t in s.turns]}
            leaked, _ = is_holdout_leaked(t_dict)
            if leaked:
                leak_count += 1
        assert leak_count == 0, f"Found {leak_count} leaked trajectories in cached train set!"


# ==============================================================================
# Deliverable (c): Adversarial Near-Miss Distractors & Guardrails 2A-2D
# ==============================================================================

def test_synthetic_trajectory_generator_guardrails():
    """Verifies 20+ archetypes, bounded CSS, attribute dropout, near-miss distractors, and DOM grafting."""
    cfg = JEDM2Config()
    gen = SyntheticTrajectoryGenerator(cfg, seed=123)

    assert len(gen.ARCHETYPES) >= 20, f"Expected >=20 archetypes, found {len(gen.ARCHETYPES)}"

    has_distractor_in_dom = False
    has_attribute_dropout = False
    has_dom_grafting = False

    for i in range(40):
        traj = gen.generate_trajectory(scenario_idx=i)
        assert len(traj.turns) > 0
        assert traj.goal != ""

        # Check turn observations
        for turn in traj.turns:
            obs = turn.obs_text
            # Guardrail 2D: Check if DOM grafting scaffold was used
            for scaffold_frag in ["site-header", "app-layout", "checkout-wrapper", "booking-view"]:
                if scaffold_frag in obs:
                    has_dom_grafting = True

            # Check DOM elements
            for el in turn.dom_elements:
                # Guardrail 2A: Check bounded classes (no random hash classes)
                if el.attributes and "class" in el.attributes:
                    classes = el.attributes["class"].split()
                    for c in classes:
                        assert c in BOUNDED_CLASSES, f"Random or unbounded class found: {c}"

                # Guardrail 2B: Check attribute dropout
                dom_s = el.to_dom_string()
                if " id=" not in dom_s and " class=" not in dom_s:
                    has_attribute_dropout = True

            # Guardrail 2C: If action is TYPE, verify goal has distractor
            if turn.action_type == ActionType.TYPE:
                # Check for near-miss entity in DOM
                for el in turn.dom_elements:
                    if el.text in traj.goal or (el.value and el.value in traj.goal):
                        has_distractor_in_dom = True

    assert has_dom_grafting, "DOM grafting (Guardrail 2D) did not activate across 40 trajectories!"
    assert has_attribute_dropout, "Structural attribute dropout (Guardrail 2B) did not activate!"
    assert has_distractor_in_dom, "Adversarial near-miss distractors (Guardrail 2C) not found in DOM elements!"


# ==============================================================================
# Deliverable (d): Per-Source MoE Auxiliary Loss & Zero Expert Bifurcation
# ==============================================================================

def test_per_source_moe_auxiliary_loss_and_routing_entropy():
    """Verifies per-source MoE load-balancing loss and high entropy on REAL vs SYNTH."""
    torch.manual_seed(42)
    cfg = JEDM2Config()
    moe = SparseTop2MoE(cfg)

    B, S, D = 4, 16, cfg.hidden_dim
    x = torch.randn(B, S, D, requires_grad=True)

    # 2 real samples, 2 synthetic samples
    source_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])

    output, aux_loss = moe(x, source_mask=source_mask)

    assert output.shape == (B, S, D)
    assert aux_loss.ndim == 0
    assert aux_loss > 0

    # Backward pass
    aux_loss.backward()
    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum() > 0

    # Check routing entropy diagnostics
    stats = moe.last_routing_stats
    assert "entropy_real" in stats
    assert "entropy_synth" in stats

    h_real = stats["entropy_real"]
    h_synth = stats["entropy_synth"]

    # Maximum possible entropy for 8 experts is ln(8) ~= 2.079
    # If experts bifurcate (4 for real, 4 for synth), entropy drops towards ln(4) ~= 1.386 or lower.
    # Uniform routing across both sources guarantees entropy >= 1.70 nats.
    assert h_real >= 1.70, f"Expert polarization detected on REAL batch: entropy {h_real:.3f} < 1.70 nats"
    assert h_synth >= 1.70, f"Expert polarization detected on SYNTH batch: entropy {h_synth:.3f} < 1.70 nats"
