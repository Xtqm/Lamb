"""
Unit tests for calibrated argument loss weighting and multi-token argument generation.
"""

import pytest
import torch

from jed_m2.agent import JEDM2Agent
from jed_m2.config import JEDM2Config
from jed_m2.dataset import SimpleLAMTokenizer, TrajectoryBatch
from jed_m2.heads import ActionType
from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
from jed_m2.model import JEDM2Model


def test_argument_loss_calibration_and_gradient_flow():
    cfg = JEDM2Config(hidden_dim=64, num_heads=2, max_seq_len=32, num_layers=1, num_action_types=7)
    model = JEDM2Model(cfg)
    loss_cfg = JEDM2LossConfig()
    assert loss_cfg.arg_weight == 8.0

    loss_fn = JEDM2CompositeLoss(cfg, loss_cfg)

    b_sz = 2
    seq_len = 16
    vocab_size = cfg.vocab_size

    input_ids = torch.randint(11, vocab_size, (b_sz, seq_len))
    labels = input_ids.clone()
    attn_mask = torch.zeros((b_sz, 1, seq_len, seq_len), dtype=torch.float32)

    # Masks
    reason_mask = torch.zeros((b_sz, seq_len), dtype=torch.bool)
    reason_mask[:, 1:4] = True

    arg_mask = torch.zeros((b_sz, seq_len), dtype=torch.bool)
    arg_mask[:, 8:12] = True  # active argument tokens

    action_indices = torch.tensor([[5], [5]], dtype=torch.long)
    action_type_targets = torch.tensor([[1], [1]], dtype=torch.long)  # TYPE
    action_type_mask = torch.tensor([[True], [True]], dtype=torch.bool)

    coord_targets = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]], dtype=torch.float32)
    coord_loss_mask = torch.tensor([[True], [True]], dtype=torch.bool)

    jepa_obs_t = torch.zeros((b_sz, cfg.hidden_dim), dtype=torch.float32)
    jepa_action_t = torch.zeros((b_sz,), dtype=torch.long)
    jepa_obs_next = torch.zeros((b_sz, cfg.hidden_dim), dtype=torch.float32)

    batch = TrajectoryBatch(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=labels,
        reason_loss_mask=reason_mask,
        action_indices=action_indices,
        action_type_targets=action_type_targets,
        action_type_mask=action_type_mask,
        coord_targets=coord_targets,
        coord_loss_mask=coord_loss_mask,
        arg_targets=input_ids,
        arg_loss_mask=arg_mask,
        jepa_obs_t=jepa_obs_t,
        jepa_action_t=jepa_action_t,
        jepa_obs_next=jepa_obs_next,
    )

    out = model(input_ids=input_ids, attention_mask=attn_mask)
    total_loss, metrics = loss_fn(out, batch)

    assert torch.isfinite(total_loss)
    assert metrics["loss_arg"] > 0.0
    assert "loss_total" in metrics

    # Verify backpropagation produces non-zero gradients for argument head
    total_loss.backward()
    grad = model.action_head.argument_head.proj.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0.0


def test_agent_multi_token_argument_emission():
    cfg = JEDM2Config(hidden_dim=64, num_heads=2, max_seq_len=64, num_layers=1, num_action_types=7)
    model = JEDM2Model(cfg)
    agent = JEDM2Agent(model, cfg, max_thought_tokens=2)

    # Act on observation with typing intent
    step_out = agent.act(
        obs_text="Search input box",
        snap_coordinates=True,
    )

    # Verify return types and non-crash
    assert isinstance(step_out.argument, str)
    assert hasattr(step_out, "raw_coordinates")
    assert hasattr(step_out, "coordinates")


def test_finish_action_weight_scaling():
    cfg = JEDM2Config(hidden_dim=64, num_heads=2, max_seq_len=32, num_layers=1, num_action_types=7)
    model = JEDM2Model(cfg)

    # Config with standard finish weight 2.0 vs 10.0
    loss_cfg_1 = JEDM2LossConfig(finish_action_weight=2.0)
    loss_fn_1 = JEDM2CompositeLoss(cfg, loss_cfg_1)

    loss_cfg_2 = JEDM2LossConfig(finish_action_weight=10.0)
    loss_fn_2 = JEDM2CompositeLoss(cfg, loss_cfg_2)

    assert loss_cfg_1.finish_action_weight == 2.0
    assert loss_cfg_2.finish_action_weight == 10.0

    b_sz = 2
    seq_len = 8
    input_ids = torch.randint(11, cfg.vocab_size, (b_sz, seq_len))
    attn_mask = torch.zeros((b_sz, 1, seq_len, seq_len), dtype=torch.float32)

    # Batch with mixed action targets: sample 0 has CLICK (0), sample 1 has FINISH (5)
    action_indices = torch.tensor([[4], [4]], dtype=torch.long)
    action_type_targets = torch.tensor([[0], [5]], dtype=torch.long)
    action_type_mask = torch.tensor([[True], [True]], dtype=torch.bool)

    batch = TrajectoryBatch(
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=input_ids,
        reason_loss_mask=torch.zeros((b_sz, seq_len), dtype=torch.bool),
        action_indices=action_indices,
        action_type_targets=action_type_targets,
        action_type_mask=action_type_mask,
        coord_targets=torch.zeros((b_sz, 1, 2), dtype=torch.float32),
        coord_loss_mask=torch.zeros((b_sz, 1), dtype=torch.bool),
        arg_targets=input_ids,
        arg_loss_mask=torch.zeros((b_sz, seq_len), dtype=torch.bool),
        jepa_obs_t=torch.zeros((b_sz, cfg.hidden_dim), dtype=torch.float32),
        jepa_action_t=torch.zeros((b_sz,), dtype=torch.long),
        jepa_obs_next=torch.zeros((b_sz, cfg.hidden_dim), dtype=torch.float32),
    )

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        # Artificially set higher loss for FINISH (class 5) than CLICK (class 0)
        out.action_output.action_type_logits[0, 4, :] = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # high prob on CLICK -> low loss
        out.action_output.action_type_logits[1, 4, :] = torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0, -5.0, 0.0])  # low prob on FINISH -> high loss

        _, m1 = loss_fn_1(out, batch)
        _, m2 = loss_fn_2(out, batch)

    # Weighting FINISH at 10.0 shifts the weighted cross-entropy higher towards FINISH loss
    assert m2["loss_action_type"] > m1["loss_action_type"]

