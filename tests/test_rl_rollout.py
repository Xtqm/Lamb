"""
Unit tests for JED-M2 Real-Task GRPO Rollout Sampling and Collation.
Verifies:
- Group rollout collection across authentic real web tasks (G=4).
- Group advantage normalization and continuous coordinate targets collation.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import os
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from jed_m2.config import JEDM2Config
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.model import JEDM2Model
from jed_m2.real_environment import RealTaskDOMEnvironment
from jed_m2.rl_pipeline import TrajectoryRolloutSampler


class TestRLRollout(unittest.TestCase):
    """Test suite for real-task rollout sampling and group advantage collation."""

    @classmethod
    def setUpClass(cls):
        from jed_m2.model import create_jed_m2_nano
        cls.model = create_jed_m2_nano(vocab_size=256)
        cls.config = cls.model.config
        cls.config.max_seq_len = 256
        cls.device = torch.device("cpu")
        cls.model.to(cls.device)
        cls.model.eval()

        cls.tokenizer = SimpleLAMTokenizer(vocab_size=cls.config.vocab_size)
        cls.sampler = TrajectoryRolloutSampler(
            model=cls.model,
            config=cls.config,
            tokenizer=cls.tokenizer,
            device=cls.device,
            dtype=torch.float32,
            max_thought_tokens=4,
        )

    def test_real_task_rollout_sampling(self):
        """Verifies group rollout generation on RealTaskDOMEnvironment with advantage normalization."""
        # Create environment with mock tasks for ultra-fast unit testing
        env = RealTaskDOMEnvironment(
            raw_mind2web_dir=None,
            raw_webarena_path=None,
            max_elements=16,
            hidden_dim=self.config.hidden_dim,
            seed=42,
        )

        group_size = 4
        group = self.sampler.sample_group_rollouts(
            scenario_id="mock_ecommerce_001",
            group_size=group_size,
            temperature=0.7,
            max_turns=2,
            env_type="real_web",
            browser_env=env,
        )

        self.assertEqual(len(group), group_size, "Group must contain exactly G trajectories.")
        for traj in group:
            self.assertEqual(traj.scenario_id, "mock_ecommerce_001")
            self.assertGreater(len(traj.full_token_ids), 0)

        # Verify collation for GRPO
        batch = self.sampler.collate_rollouts(group, reward_mode="shaped")
        self.assertIn("input_ids", batch)
        self.assertIn("targets", batch)
        self.assertIn("advantages", batch)
        self.assertIn("target_coords", batch)

        self.assertEqual(batch["input_ids"].shape[0], group_size)
        self.assertEqual(batch["advantages"].shape[0], group_size)
        self.assertEqual(batch["target_coords"].shape[0], group_size)
        self.assertEqual(batch["target_coords"].shape[2], 2)

        # Advantage mean across group should be approximately 0.0
        adv_mean = batch["advantages"].mean().item()
        self.assertAlmostEqual(adv_mean, 0.0, places=4)
        print("PASS: test_real_task_rollout_sampling - Group advantages:", batch["advantages"].tolist())

    def test_rollout_scaffold_referee_termination(self):
        """Verifies that TrajectoryRolloutSampler uses ExecutionScaffold and CompletionReferee."""
        from jed_m2.environment import WebOSEnvironment
        env = WebOSEnvironment(scenario_id="ecommerce")

        traj = self.sampler.sample_trajectory(
            env=env,
            temperature=0.0,
            max_turns=3,
        )

        self.assertIsNotNone(traj)
        self.assertGreater(len(traj.turns), 0)
        self.assertTrue(hasattr(self.sampler, "scaffold"))
        self.assertTrue(hasattr(self.sampler, "referee"))
        print("PASS: test_rollout_scaffold_referee_termination - Trajectory num steps:", traj.num_steps)


if __name__ == "__main__":
    unittest.main()
