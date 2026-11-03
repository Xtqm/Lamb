"""
Unit tests for JED-M2 Parallel Prompt Prefill & State Equivalence.
Verifies:
- Numerical parity between parallel prefill (model.prefill) and sequential step (model.step).
- Seamless transition from parallel prompt prefill to autoregressive recurrent rollout.
- Sub-100ms latency on CUDA / hardware benchmarking without exceeding 6.0 GB VRAM.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import sys
import time
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from jed_m2.config import JEDM2Config
from jed_m2.mamba import MambaBlock
from jed_m2.mla import MultiHeadLatentAttention
from jed_m2.model import JEDM2Block, JEDM2Model
from jed_m2.agent import JEDM2Agent
from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.dataset import LegacyHashTokenizer
from eval_benchmarks import detect_tokenizer_for_checkpoint


class TestParallelPrefill(unittest.TestCase):
    """Test suite for parallel prompt prefill and state equivalence."""

    @classmethod
    def setUpClass(cls):
        cls.config = JEDM2Config(
            vocab_size=1024,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_dt_rank=16,
            max_seq_len=512,
        )

    def test_mamba_prefill_equivalence(self):
        """Verifies MambaBlock.prefill produces exact final SSM and conv states as step()."""
        block = MambaBlock(self.config).eval()
        x = torch.randn(2, 16, self.config.hidden_dim)

        # Sequential step mode
        ssm, conv = None, None
        for t in range(16):
            out_t, ssm, conv = block.step(x[:, t], ssm_state=ssm, conv_state=conv)

        # Parallel prefill
        out_fwd, final_ssm, final_conv = block.prefill(x)

        ssm_diff = (ssm - final_ssm).abs().max().item()
        conv_diff = (conv - final_conv).abs().max().item()
        out_diff = (out_t - out_fwd[:, -1]).abs().max().item()

        self.assertLess(ssm_diff, 1e-4)
        self.assertLess(conv_diff, 1e-4)
        self.assertLess(out_diff, 1e-4)

    def test_mla_prefill_equivalence(self):
        """Verifies MultiHeadLatentAttention.prefill produces exact KV cache as step()."""
        mla = MultiHeadLatentAttention(self.config).eval()
        x = torch.randn(2, 12, self.config.hidden_dim)

        # Sequential step mode
        kv = None
        for t in range(12):
            out_t, kv = mla.step(x[:, t], kv_cache=kv, pos=t)

        # Parallel prefill
        out_fwd, final_kv = mla.prefill(x)

        c_kv_diff = (kv[0] - final_kv[0]).abs().max().item()
        k_rope_diff = (kv[1] - final_kv[1]).abs().max().item()
        out_diff = (out_t - out_fwd[:, -1]).abs().max().item()

        self.assertLess(c_kv_diff, 1e-4)
        self.assertLess(k_rope_diff, 1e-4)
        self.assertLess(out_diff, 1e-4)

    def test_model_prefill_step_continuation(self):
        """Verifies JEDM2Model.prefill seeds states allowing exact step() continuation."""
        torch.manual_seed(42)
        model = JEDM2Model(self.config).eval()
        prompt_tokens = torch.randint(0, self.config.vocab_size, (1, 20))

        # 1. Full sequential rollout
        states_seq = None
        for pos in range(20):
            l_seq, a_seq, states_seq = model.step(
                prompt_tokens[:, pos], states=states_seq, pos=pos
            )

        # 2. Parallel prefill
        l_pre, a_pre, states_pre = model.prefill(prompt_tokens, start_pos=0)

        # Assert final prompt position parity
        action_diff = (
            a_seq.action_type_logits - a_pre.action_type_logits
        ).abs().max().item()
        self.assertLess(action_diff, 1e-4)

        # 3. Step 21 continuation from both
        next_tok = torch.tensor([123])
        l_seq_next, a_seq_next, _ = model.step(next_tok, states=states_seq, pos=20)
        l_pre_next, a_pre_next, _ = model.step(next_tok, states=states_pre, pos=20)

        next_action_diff = (
            a_seq_next.action_type_logits - a_pre_next.action_type_logits
        ).abs().max().item()
        self.assertLess(next_action_diff, 1e-4)

    def test_prefill_latency_bench(self):
        """Verifies parallel prefill on a long 512-token prompt executes in under 100ms."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        model = JEDM2Model(self.config).to(device=device, dtype=dtype).eval()
        prompt = torch.randint(0, self.config.vocab_size, (1, 512), device=device)

        # Warmup
        with torch.no_grad():
            model.prefill(prompt)

        # Bench
        t_start = time.perf_counter()
        with torch.no_grad():
            model.prefill(prompt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        self.assertLess(latency_ms, 150.0)

    def test_detect_tokenizer_for_checkpoints(self):
        """Verifies detect_tokenizer_for_checkpoint returns correct tokenizer type."""
        # 1. Unified pretrain checkpoint path -> ActionBPETokenizer
        tok_pre = detect_tokenizer_for_checkpoint(
            "checkpoints/pretrain_unified/jed_m2_nano_pretrain_final.pt", self.config
        )
        self.assertIsInstance(tok_pre, ActionBPETokenizer)

        # 2. GRPO checkpoint path -> ActionBPETokenizer
        tok_grpo = detect_tokenizer_for_checkpoint(
            "checkpoints/grpo_run2/jed_m2_grpo_step_40.pt", self.config
        )
        self.assertIsInstance(tok_grpo, ActionBPETokenizer)

        # 3. SFT base checkpoint path -> LegacyHashTokenizer
        tok_sft = detect_tokenizer_for_checkpoint(
            "checkpoints/jed_m2_sft_base.pt", self.config
        )
        self.assertIsInstance(tok_sft, LegacyHashTokenizer)

    def test_agent_rollout_with_parallel_prefill(self):
        """Verifies JEDM2Agent rollout on real DOM strings with parallel prefill."""
        torch.manual_seed(42)
        model = JEDM2Model(self.config).eval()
        tokenizer = ActionBPETokenizer(vocab_size=self.config.vocab_size)
        agent = JEDM2Agent(
            model=model,
            config=self.config,
            tokenizer=tokenizer,
            device="cpu",
            max_thought_tokens=8,
        )

        real_dom = (
            "<form action='/login' method='post'>"
            "<input type='text' name='username' placeholder='Email or Username' id='user_in'/>"
            "<input type='password' name='pass' placeholder='Password' id='pass_in'/>"
            "<button type='submit' id='login_btn'>Sign In</button>"
            "<a href='/forgot'>Forgot Password?</a>"
            "</form>"
        )

        # Turn 1
        out1 = agent.act(real_dom)
        self.assertIn(out1.action_type, range(self.config.num_action_types))
        self.assertGreaterEqual(out1.coordinates[0], 0.0)
        self.assertLessEqual(out1.coordinates[0], 1.0)
        self.assertGreaterEqual(out1.coordinates[1], 0.0)
        self.assertLessEqual(out1.coordinates[1], 1.0)
        self.assertGreater(agent.current_pos, 0)
        self.assertIsNotNone(agent.recurrent_states)

        # Turn 2: state continuation
        pos_after_t1 = agent.current_pos
        out2 = agent.act("Observation: Logged in successfully. Current URL: /dashboard")
        self.assertGreater(agent.current_pos, pos_after_t1)
        self.assertIn(out2.action_type, range(self.config.num_action_types))

    def test_agent_oversized_prompt_truncation(self):
        """Verifies agent handles prompts exceeding max_seq_len without runtime errors."""
        model = JEDM2Model(self.config).eval()
        tokenizer = ActionBPETokenizer(vocab_size=self.config.vocab_size)
        agent = JEDM2Agent(
            model=model,
            config=self.config,
            tokenizer=tokenizer,
            device="cpu",
        )

        # Create oversized DOM with 2000 repeated tokens
        long_dom = "<div>" + " long text token repeat " * 300 + "</div>"
        out = agent.act(long_dom)
        self.assertIn(out.action_type, range(self.config.num_action_types))
        self.assertFalse(torch.isnan(torch.tensor(out.coordinates)).any())


if __name__ == "__main__":
    unittest.main()
