"""
Unit tests for JEDM2Agent with Differential DOM Delta Ingestion.

Verifies:
1. Ingestion of <obs full> and <obs delta> observations.
2. Recurrent state persistence across delta turns.
3. TurnRecord delta tracking and token savings metrics.
4. Multi-turn closed-loop execution with PlaywrightEnvironment.
"""

import pytest
import torch

from jed_m2.config import JEDM2Config
from jed_m2.model import JEDM2Model
from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.agent import JEDM2Agent
from jed_m2.browser import PlaywrightEnvironment


def _create_mini_agent():
    """Constructs lightweight JEDM2Agent instance for testing."""
    config = JEDM2Config(
        vocab_size=8192,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=32,
        num_experts=4,
        top_k=2,
        moe_intermediate_dim=256,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_dt_rank=8,
    )
    model = JEDM2Model(config)
    tokenizer = ActionBPETokenizer(vocab_size=config.vocab_size)
    agent = JEDM2Agent(model=model, config=config, tokenizer=tokenizer, device="cpu", dtype=torch.float32)
    return agent


def test_agent_delta_prompt_ingestion():
    """Verifies that JEDM2Agent ingests full and delta observations."""
    agent = _create_mini_agent()
    agent.reset()

    # Turn 0: Full snapshot
    full_obs = "<obs full> URL: https://example.com | DOM: <button id=1 role='button' bbox='[0.50,0.50]'>Search</button> </obs>"
    out0 = agent.act(obs_text=full_obs)

    assert out0.is_delta is False
    assert out0.obs_token_count > 0
    assert agent.recurrent_states is not None
    pos_after_0 = agent.current_pos
    assert pos_after_0 > 0

    # Turn 1: Differential delta
    delta_obs = "<obs delta> ~ id=1 text='Searching...' </obs>"
    out1 = agent.act(obs_text=delta_obs)

    assert out1.is_delta is True
    assert out1.obs_token_count > 0
    assert out1.obs_token_count < out0.obs_token_count
    assert agent.current_pos > pos_after_0


def test_agent_turn_record_metrics():
    """Verifies TurnRecord tracks delta token metrics."""
    agent = _create_mini_agent()
    env = PlaywrightEnvironment(mock_mode=True, use_differential_obs=True)

    agent.reset()
    env.reset()

    # Step 1
    agent_out1, step_res1 = agent.step_environment(env)
    assert len(agent.history) == 1
    rec1 = agent.history[0]
    assert rec1.turn_id == 1

    # Step 2
    agent_out2, step_res2 = agent.step_environment(env)
    assert len(agent.history) == 2
    rec2 = agent.history[1]
    assert rec2.turn_id == 2
    assert rec2.delta_tokens_count > 0


def test_multi_turn_rollout_memory_coherence():
    """Verifies 4-turn closed-loop episode without NaNs or state reset."""
    agent = _create_mini_agent()
    env = PlaywrightEnvironment(mock_mode=True, use_differential_obs=True, max_steps=5)

    records = agent.run_episode(env, max_turns=4)
    assert len(records) > 0
    for r in records:
        assert not torch.isnan(torch.tensor(r.coordinates[0]))
        assert not torch.isnan(torch.tensor(r.coordinates[1]))
        assert r.turn_id > 0
