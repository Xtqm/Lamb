"""
JED-M2 Real-World Reinforcement Learning Test Suite.
Validates:
1. RealWebPreferenceDataset: Expert trajectory extraction, spatial coordinate perturbation (|Delta x| > 0.15, |Delta y| > 0.15),
   and action primitive substitution.
2. RealPreferenceCollator: Paired batch collation for DPO, tensor shapes, shift alignment, and action loss masks.
3. DPOTrainer: Backward pass gradient flow with calculate_dpo_loss() on paired real-web traces.
4. PlaywrightWebEnvironment: Interface compliance (reset, step), local HTML fixture resolution, and native browser dispatch.
5. WebTaskOracle: Declarative programmatic verification engine, milestone reward shaping, and terminal outcome evaluation.
6. GRPOTrainer: Dual-modality Group Relative Policy Optimization updating discrete token and continuous spatial coordinate heads (lambda_coord).
7. TrajectoryRolloutSampler: Group rollout generation on browser environment sandboxes and batched collation.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import math
import os
import sys
from pathlib import Path
import unittest

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure mock browser execution under sandboxed environments
if "JED_M2_MOCK_BROWSER" not in os.environ:
    os.environ["JED_M2_MOCK_BROWSER"] = "1"

import torch
import torch.nn as nn

from jed_m2.config import JEDM2Config
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.heads import ActionType
from jed_m2.model import JEDM2Model
from jed_m2.playwright_env import (
    PlaywrightWebEnvironment,
    WebTaskOracle,
    get_default_web_oracles,
    get_fixture_path,
)
from jed_m2.real_preference_dataset import (
    PreferenceTrajectory,
    PreferenceTurn,
    RealPreferenceCollator,
    RealPreferencePair,
    RealWebPreferenceDataset,
    inject_action_primitive_substitution,
    inject_spatial_coordinate_perturbation,
    tokenize_preference_trajectory,
)
from jed_m2.rl import (
    DPOConfig,
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
    calculate_dpo_loss,
    calculate_grpo_loss,
    compute_coordinate_log_probs,
    compute_token_log_probs,
)
from jed_m2.rl_pipeline import (
    RolloutTrajectory,
    RolloutTurn,
    TrajectoryRolloutSampler,
)


class TestRealWorldRL(unittest.TestCase):
    """Unit and integration test suite for real-world web reinforcement learning."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.config = JEDM2Config(
            hidden_dim=128,
            num_layers=2,
            vocab_size=1024,
            max_seq_len=64,
            jepa_latent_dim=64,
            jepa_action_dim=16,
            num_experts=2,
            top_k=1,
            moe_intermediate_dim=256,
        )
        self.tokenizer = SimpleLAMTokenizer(self.config.vocab_size)

    # --------------------------------------------------------------------------
    # Track A: Preference Dataset & DPO Tests
    # --------------------------------------------------------------------------

    def test_01_spatial_perturbation_displacement(self):
        """Verifies spatial coordinate perturbation satisfies |Delta x| >= 0.15 and |Delta y| >= 0.15."""
        coords = (0.50, 0.50)
        pert_coords, delta = inject_spatial_coordinate_perturbation(coords, min_displacement=0.15, max_displacement=0.30)

        self.assertGreaterEqual(abs(delta[0]), 0.15, f"Delta x must be >= 0.15, got {delta[0]}")
        self.assertGreaterEqual(abs(delta[1]), 0.15, f"Delta y must be >= 0.15, got {delta[1]}")
        self.assertTrue(0.0 <= pert_coords[0] <= 1.0, f"Perturbed x must be in [0, 1], got {pert_coords[0]}")
        self.assertTrue(0.0 <= pert_coords[1] <= 1.0, f"Perturbed y must be in [0, 1], got {pert_coords[1]}")

        # Edge coordinate perturbation
        edge_coords = (0.05, 0.95)
        pert_edge, delta_edge = inject_spatial_coordinate_perturbation(edge_coords, min_displacement=0.15)
        self.assertGreaterEqual(abs(delta_edge[0]), 0.15)
        self.assertGreaterEqual(abs(delta_edge[1]), 0.15)
        self.assertTrue(0.0 <= pert_edge[0] <= 1.0)
        self.assertTrue(0.0 <= pert_edge[1] <= 1.0)

    def test_02_action_primitive_substitution(self):
        """Verifies action primitive substitution degrades optimal actions."""
        orig_click = ActionType.CLICK
        sub_click = inject_action_primitive_substitution(orig_click)
        self.assertNotEqual(sub_click, orig_click, "Substituted action must differ from CLICK")

        orig_finish = ActionType.FINISH
        sub_finish = inject_action_primitive_substitution(orig_finish)
        self.assertEqual(sub_finish, ActionType.SCROLL, "Substituted FINISH should be SCROLL")

    def test_03_preference_pair_generation(self):
        """Verifies RealWebPreferenceDataset constructs winning and losing trajectory pairs."""
        dataset = RealWebPreferenceDataset(
            config=self.config,
            split="train",
            dataset_size=4,
            force_rebuild=True,
            seed=42,
        )
        self.assertEqual(len(dataset), 4)

        pair: RealPreferencePair = dataset[0]
        self.assertTrue(pair.chosen.is_winning)
        self.assertFalse(pair.rejected.is_winning)
        self.assertIn(pair.perturbation_type, ["spatial", "action_substitution"])
        self.assertGreater(len(pair.chosen_tokens), 0)
        self.assertGreater(len(pair.rejected_tokens), 0)
        self.assertEqual(len(pair.chosen_tokens), len(pair.chosen_mask))
        self.assertEqual(len(pair.rejected_tokens), len(pair.rejected_mask))

    def test_04_preference_collator_tensor_shapes(self):
        """Verifies RealPreferenceCollator produces correctly shaped and aligned DPO tensors."""
        dataset = RealWebPreferenceDataset(
            config=self.config,
            split="train",
            dataset_size=4,
            force_rebuild=False,
            seed=42,
        )
        collator = RealPreferenceCollator(config=self.config, tokenizer=self.tokenizer, device="cpu")
        batch = collator([dataset[0], dataset[1]])

        self.assertIn("chosen_input_ids", batch)
        self.assertIn("chosen_targets", batch)
        self.assertIn("chosen_mask", batch)
        self.assertIn("rejected_input_ids", batch)
        self.assertIn("rejected_targets", batch)
        self.assertIn("rejected_mask", batch)

        b_size = 2
        self.assertEqual(batch["chosen_input_ids"].shape[0], b_size)
        self.assertEqual(batch["rejected_input_ids"].shape[0], b_size)
        self.assertEqual(batch["chosen_input_ids"].shape, batch["chosen_targets"].shape)
        self.assertEqual(batch["chosen_input_ids"].shape, batch["chosen_mask"].shape)
        self.assertEqual(batch["rejected_input_ids"].shape, batch["rejected_targets"].shape)
        self.assertEqual(batch["rejected_input_ids"].shape, batch["rejected_mask"].shape)

    def test_05_dpo_gradient_backward_pass(self):
        """Verifies DPOTrainer computes finite loss and propagates gradients to active policy."""
        model = JEDM2Model(self.config)
        ref_model = JEDM2Model(self.config)
        ref_model.load_state_dict(model.state_dict())
        for p in ref_model.parameters():
            p.requires_grad = False

        trainer = DPOTrainer(model=model, ref_model=ref_model, config=DPOConfig(beta=0.1))

        # Synthetic paired input
        b, s = 2, 16
        chosen_input = torch.randint(11, 500, (b, s))
        chosen_target = torch.randint(11, 500, (b, s))
        chosen_mask = torch.ones((b, s), dtype=torch.bool)
        rejected_input = torch.randint(11, 500, (b, s))
        rejected_target = torch.randint(11, 500, (b, s))
        rejected_mask = torch.ones((b, s), dtype=torch.bool)

        loss, metrics = trainer.compute_loss(
            chosen_input_ids=chosen_input,
            chosen_targets=chosen_target,
            chosen_mask=chosen_mask,
            rejected_input_ids=rejected_input,
            rejected_targets=rejected_target,
            rejected_mask=rejected_mask,
        )

        self.assertTrue(torch.isfinite(loss), "DPO loss must be finite")
        self.assertIn("loss_dpo", metrics)
        self.assertIn("reward_margin", metrics)
        self.assertIn("dpo_accuracy", metrics)

        loss.backward()
        # Verify finite gradients on model parameters
        first_param = next(model.parameters())
        self.assertIsNotNone(first_param.grad)
        self.assertTrue(torch.isfinite(first_param.grad).all())

    # --------------------------------------------------------------------------
    # Track B: Playwright Browser Environment & Oracle Tests
    # --------------------------------------------------------------------------

    def test_06_local_fixture_paths(self):
        """Verifies local HTML benchmark fixture files exist on disk."""
        ecom_path = get_fixture_path("ecommerce")
        form_path = get_fixture_path("form_portal")
        travel_path = get_fixture_path("travel")

        self.assertIsNotNone(ecom_path, "ecommerce fixture must exist")
        self.assertTrue(ecom_path.exists())
        self.assertIsNotNone(form_path, "form_portal fixture must exist")
        self.assertTrue(form_path.exists())
        self.assertIsNotNone(travel_path, "travel fixture must exist")
        self.assertTrue(travel_path.exists())

    def test_07_programmatic_oracle_evaluation(self):
        """Verifies WebTaskOracle evaluates URL regex, selector matches, and terminal rewards."""
        oracle = WebTaskOracle(
            scenario_id="ecommerce",
            url_pattern=r"file:///.*ecommerce\.html",
            required_selectors=["#cart-badge"],
            text_assertions=[("#cart-badge", "1")],
            milestone_selectors={"#cart-status": 0.5},
            terminal_reward=1.0,
        )

        # 1. Unfinished state: no cart item
        dom_text_empty = "Saleor Benchmark Store | Cart Items: 0"
        r_empty, sat_empty, _ = oracle.evaluate_page(
            page=None,
            current_url="file:///ecommerce.html",
            current_dom_text=dom_text_empty,
        )
        self.assertFalse(sat_empty)
        self.assertEqual(r_empty, 0.0)

        # 2. Finished state: cart item present
        dom_text_done = "Saleor Benchmark Store | #cart-badge 1 | Item added to cart"
        r_done, sat_done, details = oracle.evaluate_page(
            page=None,
            current_url="file:///ecommerce.html",
            current_dom_text=dom_text_done,
        )
        self.assertTrue(sat_done)
        self.assertGreaterEqual(r_done, 1.0)
        self.assertTrue(len(details["checks"]) > 0)

    def test_08_playwright_env_interface_conformance(self):
        """Verifies PlaywrightWebEnvironment adheres to WebOSEnvironment interface."""
        env = PlaywrightWebEnvironment(
            config=self.config,
            scenario_id="ecommerce",
            headless=True,
            device="cpu",
            dtype=torch.float32,
        )
        res = env.reset()
        self.assertIsNotNone(res.obs_text)
        self.assertIsNotNone(res.obs_features)
        self.assertEqual(res.obs_features.shape[0], self.config.hidden_dim)
        self.assertFalse(res.done)
        self.assertFalse(env.goal_satisfied)

        # Step CLICK
        step1 = env.step(ActionType.CLICK, coordinates=(0.5, 0.5))
        self.assertIsInstance(step1.reward, float)
        self.assertIn("step", step1.info)

        # Step FINISH
        step2 = env.step(ActionType.FINISH)
        self.assertTrue(step2.done)
        env.close()

    # --------------------------------------------------------------------------
    # Track C: Dual-Modality GRPO Tests
    # --------------------------------------------------------------------------

    def test_09_grpo_dual_modality_loss_computation(self):
        """Verifies calculate_grpo_loss incorporates discrete token and continuous coordinate logps."""
        cfg = GRPOConfig(group_size=2, lambda_coord=0.5, clip_eps=0.2, beta_kl=0.04)
        b, s = 2, 8

        policy_token_logps = torch.randn(b, s)
        old_token_logps = policy_token_logps - 0.1
        ref_token_logps = policy_token_logps - 0.05
        rewards = torch.tensor([1.0, 0.0])
        mask = torch.ones(b, s, dtype=torch.bool)

        policy_coord_logps = torch.randn(b, s)
        old_coord_logps = policy_coord_logps - 0.2
        ref_coord_logps = policy_coord_logps - 0.1

        loss, metrics = calculate_grpo_loss(
            policy_token_logps=policy_token_logps,
            old_token_logps=old_token_logps,
            ref_token_logps=ref_token_logps,
            rewards=rewards,
            mask=mask,
            policy_coord_logps=policy_coord_logps,
            old_coord_logps=old_coord_logps,
            ref_coord_logps=ref_coord_logps,
            config=cfg,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_grpo", metrics)
        self.assertIn("mean_advantage", metrics)
        self.assertIn("mean_kl", metrics)
        self.assertIn("clip_fraction", metrics)

    def test_10_trajectory_rollout_sampler_browser(self):
        """Verifies TrajectoryRolloutSampler executes rollouts and collates training tensors."""
        model = JEDM2Model(self.config)
        sampler = TrajectoryRolloutSampler(
            model=model,
            config=self.config,
            tokenizer=self.tokenizer,
            device="cpu",
            dtype=torch.float32,
        )

        env = PlaywrightWebEnvironment(
            config=self.config,
            scenario_id="ecommerce",
            headless=True,
            device="cpu",
            dtype=torch.float32,
        )

        try:
            group = sampler.sample_group_rollouts(
                scenario_id="ecommerce",
                group_size=2,
                temperature=0.7,
                max_turns=2,
                env_type="browser",
                browser_env=env,
            )
            self.assertEqual(len(group), 2)
            self.assertIsInstance(group[0], RolloutTrajectory)

            batch = sampler.collate_rollouts(group, reward_mode="shaped")
            self.assertIn("input_ids", batch)
            self.assertIn("targets", batch)
            self.assertIn("mask", batch)
            self.assertIn("old_token_logps", batch)
            self.assertIn("old_coord_logps", batch)
            self.assertIn("target_coords", batch)
            self.assertEqual(batch["input_ids"].shape[0], 2)
            self.assertEqual(batch["target_coords"].shape[-1], 2)
        finally:
            env.close()

    def test_11_grpo_trainer_backward_pass(self):
        """Verifies GRPOTrainer computes joint dual-modality gradient updates on active model."""
        model = JEDM2Model(self.config)
        ref_model = JEDM2Model(self.config)
        ref_model.load_state_dict(model.state_dict())
        for p in ref_model.parameters():
            p.requires_grad = False

        grpo_cfg = GRPOConfig(group_size=2, lambda_coord=0.5)
        trainer = GRPOTrainer(model=model, ref_model=ref_model, config=grpo_cfg)

        b, s = 2, 16
        input_ids = torch.randint(11, 500, (b, s))
        targets = torch.randint(11, 500, (b, s))
        mask = torch.ones((b, s), dtype=torch.bool)
        old_token_logps = torch.randn(b, s)
        rewards = torch.tensor([1.5, 0.2])
        old_coord_logps = torch.randn(b, s)
        target_coords = torch.rand(b, s, 2)

        loss, metrics = trainer.compute_loss(
            input_ids=input_ids,
            targets=targets,
            mask=mask,
            old_token_logps=old_token_logps,
            rewards=rewards,
            old_coord_logps=old_coord_logps,
            target_coords=target_coords,
        )

        self.assertTrue(torch.isfinite(loss), "GRPO loss must be finite")
        loss.backward()

        # Verify gradient flow into backbone
        first_param = next(model.parameters())
        self.assertIsNotNone(first_param.grad)
        self.assertTrue(torch.isfinite(first_param.grad).all())


if __name__ == "__main__":
    unittest.main()
