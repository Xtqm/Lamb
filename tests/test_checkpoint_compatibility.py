"""
Unit tests for 7-action primitive defaults and legacy 6-class checkpoint compatibility.
"""

from pathlib import Path
import pytest
import torch

from jed_m2.config import JEDM2Config
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model, adapt_state_dict_for_action_classes


def test_default_config_has_seven_actions():
    cfg = JEDM2Config()
    assert cfg.num_action_types == 7
    assert len(ActionType) == 7
    assert ActionType.SELECT.value == 6

    model = JEDM2Model(cfg)
    assert model.action_head.action_type_head.num_action_types == 7
    fc2 = model.action_head.action_type_head.fc2
    assert fc2.weight.shape[0] == 7
    assert fc2.bias.shape[0] == 7


def test_adapt_state_dict_for_action_classes():
    # Simulate a 6-class state dict
    old_weight = torch.randn(6, 256)
    old_bias = torch.randn(6)
    state_dict = {
        "action_head.action_type_head.fc2.weight": old_weight,
        "action_head.action_type_head.fc2.bias": old_bias,
        "tok_embeddings.weight": torch.randn(8192, 512),
    }

    adapted = adapt_state_dict_for_action_classes(state_dict, target_classes=7)
    new_w = adapted["action_head.action_type_head.fc2.weight"]
    new_b = adapted["action_head.action_type_head.fc2.bias"]

    assert new_w.shape == (7, 256)
    assert new_b.shape == (7,)
    assert torch.equal(new_w[:6], old_weight)
    assert torch.equal(new_b[:6], old_bias)
    assert torch.all(new_w[6] == 0.0)
    assert new_b[6] == 0.0


def test_legacy_six_class_state_dict_loads_into_seven_class_model():
    cfg_6 = JEDM2Config(num_action_types=6, max_seq_len=64, hidden_dim=64, num_layers=1)
    model_6 = JEDM2Model(cfg_6)
    state_dict_6 = model_6.state_dict()

    cfg_7 = JEDM2Config(num_action_types=7, max_seq_len=64, hidden_dim=64, num_layers=1)
    model_7 = JEDM2Model(cfg_7)

    # Directly loading 6-class state dict into 7-class model should succeed via zero-padding
    model_7.load_state_dict(state_dict_6, strict=True)
    fc2_7 = model_7.action_head.action_type_head.fc2
    assert fc2_7.weight.shape[0] == 7
    assert fc2_7.bias.shape[0] == 7
    assert torch.equal(fc2_7.weight[:6], model_6.action_head.action_type_head.fc2.weight)


def test_legacy_saved_checkpoint_loads_cleanly_if_present():
    ckpt_path = Path("checkpoints/step-19500.pt")
    if not ckpt_path.is_file():
        pytest.skip("Legacy checkpoint step-19500.pt not found on disk")

    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Instantiate 7-class model
    cfg_7 = JEDM2Config(num_action_types=7, max_seq_len=saved["model_config"].get("max_seq_len", 512))
    model = JEDM2Model(cfg_7)

    # Should load cleanly without shape mismatch
    model.load_state_dict(saved["model_state_dict"], strict=True)
    assert model.action_head.action_type_head.fc2.weight.shape[0] == 7
