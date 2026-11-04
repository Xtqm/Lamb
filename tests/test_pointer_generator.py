"""
Tests for Pointer-Generator Copy Mechanism in ArgumentHead and NativeActionHead.
Validates:
- Forward pass shapes in 2D and 3D with and without prompt context.
- Probability normalization: sum(P(word)) == 1.0.
- Duplicate token probability accumulation via scatter_add.
- Gradient flow through query/key projections, generation gate, and vocab projection.
- Backward compatibility when prompt inputs are absent.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.heads import ArgumentHead, NativeActionHead, ActionOutput, ActionType


@pytest.fixture
def test_config():
    return JEDM2Config(
        hidden_dim=128,
        vocab_size=256,
        num_layers=2,
        num_heads=4,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
        num_experts=4,
        top_k=2,
        moe_intermediate_dim=256,
    )


def test_argument_head_without_prompt(test_config):
    head = ArgumentHead(test_config)
    
    # 2D case: [B, D]
    x_2d = torch.randn(2, test_config.hidden_dim)
    logits_2d = head(x_2d)
    assert logits_2d.shape == (2, test_config.vocab_size)
    assert not torch.isnan(logits_2d).any()

    # 3D case: [B, S, D]
    x_3d = torch.randn(2, 5, test_config.hidden_dim)
    logits_3d = head(x_3d)
    assert logits_3d.shape == (2, 5, test_config.vocab_size)
    assert not torch.isnan(logits_3d).any()


def test_argument_head_with_prompt_2d(test_config):
    head = ArgumentHead(test_config)
    b_sz = 2
    s_p = 8
    
    x = torch.randn(b_sz, test_config.hidden_dim)
    prompt_hidden = torch.randn(b_sz, s_p, test_config.hidden_dim)
    prompt_ids = torch.randint(0, test_config.vocab_size, (b_sz, s_p))
    
    log_p, p_pointer, alpha, p_final = head.forward_components(
        x, prompt_hidden=prompt_hidden, prompt_ids=prompt_ids
    )
    
    assert log_p.shape == (b_sz, test_config.vocab_size)
    assert p_pointer.shape == (b_sz, s_p)
    assert alpha.shape == (b_sz, 1)
    assert p_final.shape == (b_sz, test_config.vocab_size)
    
    # Verify probability normalization
    probs_sum = p_final.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)
    
    # Verify alpha bounded in [0, 1]
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()
    # Verify pointer distribution normalized
    pointer_sum = p_pointer.sum(dim=-1)
    assert torch.allclose(pointer_sum, torch.ones_like(pointer_sum), atol=1e-4)


def test_argument_head_with_prompt_3d(test_config):
    head = ArgumentHead(test_config)
    b_sz = 2
    seq_len = 4
    s_p = 8
    
    x = torch.randn(b_sz, seq_len, test_config.hidden_dim)
    prompt_hidden = torch.randn(b_sz, s_p, test_config.hidden_dim)
    prompt_ids = torch.randint(0, test_config.vocab_size, (b_sz, s_p))
    
    log_p, p_pointer, alpha, p_final = head.forward_components(
        x, prompt_hidden=prompt_hidden, prompt_ids=prompt_ids
    )
    
    assert log_p.shape == (b_sz, seq_len, test_config.vocab_size)
    assert p_pointer.shape == (b_sz, seq_len, s_p)
    assert alpha.shape == (b_sz, seq_len, 1)
    assert p_final.shape == (b_sz, seq_len, test_config.vocab_size)
    
    # Verify probability normalization across sequence
    probs_sum = p_final.sum(dim=-1)
    assert torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-4)


def test_pointer_accumulation_on_duplicate_tokens(test_config):
    head = ArgumentHead(test_config)
    b_sz = 1
    s_p = 4
    
    x = torch.randn(b_sz, test_config.hidden_dim)
    prompt_hidden = torch.randn(b_sz, s_p, test_config.hidden_dim)
    # Token 50 appears at indices 0 and 2
    prompt_ids = torch.tensor([[50, 10, 50, 20]], dtype=torch.long)
    
    log_p, p_pointer, alpha, p_final = head.forward_components(
        x, prompt_hidden=prompt_hidden, prompt_ids=prompt_ids
    )
    
    # Probability of token 50 should include contributions from index 0 and 2:
    # (1 - alpha) * (p_pointer[0, 0] + p_pointer[0, 2]) + alpha * p_vocab[50]
    expected_copy_contrib = (1.0 - alpha[0, 0]) * (p_pointer[0, 0] + p_pointer[0, 2])
    assert p_final[0, 50] >= expected_copy_contrib


def test_pointer_generator_gradients(test_config):
    head = ArgumentHead(test_config)
    b_sz = 2
    s_p = 6
    
    x = torch.randn(b_sz, test_config.hidden_dim, requires_grad=True)
    prompt_hidden = torch.randn(b_sz, s_p, test_config.hidden_dim, requires_grad=True)
    prompt_ids = torch.randint(0, test_config.vocab_size, (b_sz, s_p))
    target_ids = prompt_ids[:, 0]  # Target appears in prompt
    
    log_p = head(x, prompt_hidden=prompt_hidden, prompt_ids=prompt_ids)
    loss = F.nll_loss(log_p, target_ids)
    loss.backward()
    
    # Verify all components received non-zero gradients
    assert head.vocab_proj.weight.grad is not None
    assert head.query_proj.weight.grad is not None
    assert head.key_proj.weight.grad is not None
    assert head.gate_fc.weight.grad is not None
    assert head.gate_fc.bias.grad is not None
    assert not torch.isnan(head.query_proj.weight.grad).any()
    assert not torch.isnan(head.gate_fc.weight.grad).any()


def test_native_action_head_pointer_integration(test_config):
    native_head = NativeActionHead(test_config)
    b_sz = 2
    s_p = 6
    
    x = torch.randn(b_sz, test_config.hidden_dim)
    prompt_hidden = torch.randn(b_sz, s_p, test_config.hidden_dim)
    prompt_ids = torch.randint(0, test_config.vocab_size, (b_sz, s_p))
    
    # Forward pass with prompt
    out = native_head(x, prompt_hidden=prompt_hidden, prompt_ids=prompt_ids)
    assert isinstance(out, ActionOutput)
    assert out.pointer_probs is not None
    assert out.alpha is not None
    assert out.argument_probs is not None
    assert out.argument_logits.shape == (b_sz, test_config.vocab_size)
    
    # Forward pass without prompt
    out_no_prompt = native_head(x)
    assert out_no_prompt.pointer_probs is None
    assert out_no_prompt.alpha is None
    assert out_no_prompt.argument_probs is None
    assert out_no_prompt.argument_logits.shape == (b_sz, test_config.vocab_size)


def test_composite_loss_with_pointer_generator(test_config):
    from jed_m2.losses import JEDM2CompositeLoss, JEDM2LossConfig
    from jed_m2.model import JEDM2Model
    from jed_m2.dataset import TrajectoryBatch

    model = JEDM2Model(test_config)
    loss_cfg = JEDM2LossConfig()
    loss_fn = JEDM2CompositeLoss(test_config, loss_cfg)

    b_sz = 2
    seq_len = 12
    vocab_size = test_config.vocab_size

    input_ids = torch.randint(11, vocab_size, (b_sz, seq_len))
    labels = input_ids.clone()
    attn_mask = torch.zeros((b_sz, 1, seq_len, seq_len), dtype=torch.float32)

    reason_mask = torch.zeros((b_sz, seq_len), dtype=torch.bool)
    reason_mask[:, 1:3] = True

    arg_mask = torch.zeros((b_sz, seq_len), dtype=torch.bool)
    arg_mask[:, 6:9] = True

    action_indices = torch.tensor([[4], [4]], dtype=torch.long)
    action_type_targets = torch.tensor([[1], [1]], dtype=torch.long)
    action_type_mask = torch.tensor([[True], [True]], dtype=torch.bool)

    coord_targets = torch.tensor([[[0.5, 0.5]], [[0.5, 0.5]]], dtype=torch.float32)
    coord_loss_mask = torch.tensor([[True], [True]], dtype=torch.bool)

    jepa_obs_t = torch.zeros((b_sz, test_config.hidden_dim), dtype=torch.float32)
    jepa_action_t = torch.zeros((b_sz,), dtype=torch.long)
    jepa_obs_next = torch.zeros((b_sz, test_config.hidden_dim), dtype=torch.float32)

    prompt_ids = torch.randint(0, vocab_size, (b_sz, 8))
    prompt_mask = torch.ones((b_sz, 8), dtype=torch.bool)

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
        prompt_token_ids=prompt_ids,
        prompt_mask=prompt_mask,
    )

    out = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
        prompt_ids=prompt_ids,
        prompt_hidden=torch.randn(b_sz, 8, test_config.hidden_dim),
    )
    total_loss, metrics = loss_fn(out, batch)

    assert torch.isfinite(total_loss)
    assert metrics["loss_arg"] > 0.0
    total_loss.backward()
    assert model.action_head.argument_head.query_proj.weight.grad is not None
    assert model.action_head.argument_head.vocab_proj.weight.grad is not None


def test_trajectory_collator_prompt_tokens(test_config):
    from jed_m2.dataset import TrajectoryCollator, TrajectorySample, InteractionTurn

    collator = TrajectoryCollator(test_config)
    sample = TrajectorySample(
        sample_id="task_1",
        goal="Search for shoes",
        turns=[
            InteractionTurn(
                turn_id=0,
                obs_text="Search input box",
                obs_features=torch.randn(test_config.hidden_dim),
                thought="Looking for search box",
                action_type=ActionType.TYPE,
                coordinates=(0.3, 0.4),
                argument="shoes",
            )
        ],
    )
    batch = collator([sample])
    assert batch.prompt_token_ids is not None
    assert batch.prompt_mask is not None
    assert batch.prompt_token_ids.shape[0] == 1
    assert batch.prompt_token_ids.shape[1] > 0


def test_agent_argument_copy_decoding():
    from jed_m2.agent import JEDM2Agent
    from jed_m2.model import JEDM2Model
    from jed_m2.heads import ActionType

    config = JEDM2Config(
        hidden_dim=128,
        vocab_size=8192,
        num_layers=2,
        num_heads=4,
        kv_lora_rank=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        v_head_dim=16,
        num_experts=4,
        top_k=2,
        moe_intermediate_dim=256,
    )
    model = JEDM2Model(config)
    agent = JEDM2Agent(model, config=config)

    # Bias pointer-generator head towards copy: alpha -> 0 (copy mode)
    with torch.no_grad():
        agent.model.action_head.argument_head.gate_fc.bias.fill_(-10.0)
        # Bias action type head to emit TYPE
        agent.model.action_head.action_type_head.fc2.bias.data.fill_(-5.0)
        agent.model.action_head.action_type_head.fc2.bias.data[ActionType.TYPE.value] = 10.0
        # Bias action gate to commit immediately
        agent.model.action_head.gate.proj.bias.data.fill_(10.0)

    obs_text = "Search box input with placeholder: 'mechanical keyboard'"
    res = agent.act(obs_text=obs_text)

    assert res.action_name == "TYPE"
    assert res.argument != ""
    assert agent.cached_prompt_ids is not None
    cached_ids_set = set(agent.cached_prompt_ids[0].tolist())
    decoded_arg_tokens = agent.tokenizer.encode(res.argument)
    # Verify that argument tokens decoded by the agent were copied from the prompt tokens
    copied_tokens = set(decoded_arg_tokens).intersection(cached_ids_set)
    assert len(copied_tokens) > 0

