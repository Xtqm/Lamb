"""
JED-M2 JEPA Regression Test Suite
Validates:
1. End-to-end JEPA gradient propagation into shared backbone parameters.
2. Target encoder parameter gradient isolation (requires_grad=False, grad is None).
3. Target encoder Polyak EMA momentum update math and parameter divergence.
4. JEDM2Trainer step execution triggers automatic target network updates.
5. Semantic observation embedding determinism and temporal divergence (s_t != s_{t+1}).
"""

import unittest
import torch
import torch.nn as nn

from jed_m2.config import JEDM2Config
from jed_m2.model import JEDM2Model
from jed_m2.dataset import AgentTrajectoryDataset, TrajectoryCollator
from jed_m2.trainer import JEDM2Trainer, JEDM2TrainerConfig
from jed_m2.losses import JEDM2LossConfig


class TestJEPARegression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = JEDM2Config(
            hidden_dim=256,
            num_layers=2,
            vocab_size=1024,
            jepa_latent_dim=128,
            jepa_action_dim=32,
            num_experts=4,
            top_k=2,
            moe_intermediate_dim=512,
            max_seq_len=64,
        )

    def test_01_jepa_gradient_propagation_to_backbone(self):
        """Verifies JEPA loss propagates finite, non-zero gradients into backbone."""
        model = JEDM2Model(self.config)
        model.train()

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(11, 1000, (batch_size, seq_len))
        obs_t = torch.randn(batch_size, self.config.hidden_dim)
        action_t = torch.randint(0, 10, (batch_size,))
        obs_next = torch.randn(batch_size, self.config.hidden_dim)

        output = model(
            input_ids=input_ids,
            obs_t=obs_t,
            action_t=action_t,
            obs_next=obs_next,
        )

        self.assertIsNotNone(output.jepa_loss, "JEPA loss must not be None when action_t and obs_next are provided")
        output.jepa_loss.backward()

        # Check backbone blocks gradient
        block0_weight = model.blocks[0].norm_mamba.weight
        self.assertIsNotNone(block0_weight.grad, "Block 0 norm_mamba must have gradients from JEPA loss")
        grad_norm = block0_weight.grad.norm().item()
        self.assertFalse(torch.isnan(block0_weight.grad).any(), "Block 0 gradient contains NaNs")
        self.assertGreater(grad_norm, 0.0, f"Block 0 gradient norm must be > 0.0, got {grad_norm}")

        # Check token embedding gradient
        self.assertIsNotNone(model.tok_embeddings.weight.grad, "tok_embeddings must have gradients from JEPA loss")
        tok_grad_norm = model.tok_embeddings.weight.grad.norm().item()
        self.assertGreater(tok_grad_norm, 0.0, f"Token embeddings gradient norm must be > 0.0, got {tok_grad_norm}")

    def test_02_target_encoder_gradient_isolation(self):
        """Verifies EMA target encoder parameters have requires_grad=False and grad=None."""
        model = JEDM2Model(self.config)
        model.train()

        target_enc = model.jepa.target_encoder
        for name, param in target_enc.named_parameters():
            self.assertFalse(param.requires_grad, f"Target encoder param {name} should have requires_grad=False")

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(11, 1000, (batch_size, seq_len))
        action_t = torch.randint(0, 10, (batch_size,))
        obs_next = torch.randn(batch_size, self.config.hidden_dim)

        output = model(input_ids=input_ids, action_t=action_t, obs_next=obs_next)
        output.jepa_loss.backward()

        for name, param in target_enc.named_parameters():
            self.assertIsNone(param.grad, f"Target encoder param {name} should have grad=None after backward()")

    def test_03_target_encoder_ema_update_math(self):
        """Verifies Polyak EMA update mathematically alters target encoder parameters."""
        model = JEDM2Model(self.config)
        jepa = model.jepa
        decay = 0.90  # 10% update from online
        jepa.ema_decay = decay

        # Get a target parameter copy
        target_param = next(jepa.target_encoder.encoder.parameters())
        online_param = next(jepa.encoder.parameters())
        initial_target_val = target_param.clone()

        # Perturb online parameter
        with torch.no_grad():
            online_param.add_(10.0)

        expected_val = decay * initial_target_val + (1.0 - decay) * online_param

        # Execute EMA update
        jepa.update_target_network()

        updated_target_val = next(jepa.target_encoder.encoder.parameters())
        diff = (updated_target_val - initial_target_val).abs().max().item()
        self.assertGreater(diff, 0.1, "Target parameter should have changed after update_target_network()")
        self.assertTrue(torch.allclose(updated_target_val, expected_val, atol=1e-5), "EMA math mismatch")

    def test_04_trainer_updates_target_network(self):
        """Verifies JEDM2Trainer.training_step() triggers target network EMA update."""
        model = JEDM2Model(self.config)
        trainer_config = JEDM2TrainerConfig(
            lr=1e-2,
            max_steps=5,
            grad_accum_steps=1,
            mixed_precision="no",
        )
        trainer = JEDM2Trainer(
            model=model,
            config=trainer_config,
            loss_config=JEDM2LossConfig(),
            device=torch.device("cpu"),
        )

        dataset = AgentTrajectoryDataset(config=self.config, virtual_size=4, seed=42)
        collator = TrajectoryCollator(config=self.config, max_seq_len=32)
        batch = collator([dataset[0]])

        # Record initial target encoder parameter
        initial_target_weight = next(model.jepa.target_encoder.encoder.parameters()).clone()

        # Execute single training step (grad_accum_steps = 1)
        loss_val, metrics = trainer.training_step(batch)

        # Check updated target encoder parameter
        updated_target_weight = next(model.jepa.target_encoder.encoder.parameters())
        weight_diff = (updated_target_weight - initial_target_weight).abs().max().item()
        self.assertGreater(
            weight_diff,
            1e-8,
            f"Trainer training_step() failed to update JEPA target network! weight_diff={weight_diff}",
        )

    def test_05_semantic_observation_divergence(self):
        """Verifies observation features are semantically grounded and not identical across turns."""
        dataset = AgentTrajectoryDataset(config=self.config, virtual_size=2, seed=42)
        sample = dataset[0]
        self.assertGreaterEqual(len(sample.turns), 2, "Sample must have at least 2 turns")

        turn0_feat = sample.turns[0].obs_features
        turn1_feat = sample.turns[1].obs_features

        # Check normalization (norm ~ 1.0)
        self.assertAlmostEqual(turn0_feat.norm(p=2).item(), 1.0, places=4)
        self.assertAlmostEqual(turn1_feat.norm(p=2).item(), 1.0, places=4)

        # Check cosine similarity is strictly < 1.0 (they are distinct states)
        cos_sim = torch.nn.functional.cosine_similarity(turn0_feat.unsqueeze(0), turn1_feat.unsqueeze(0)).item()
        self.assertLess(cos_sim, 0.99, f"Observations between turns should differ, got cos_sim={cos_sim}")


if __name__ == "__main__":
    unittest.main()
