"""
Unit tests for JED-M2 RealTaskDOMEnvironment and RealTaskOracle.
Verifies:
- Task initialization, HTML DOM tree parsing, and interactive candidate extraction.
- Programmatic multi-attribute oracle scoring across discrete action types, spatial IoU,
  continuous Gaussian coordinate proximity, and textual arguments.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import os
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from jed_m2.heads import ActionType
from jed_m2.real_environment import RealTaskDOMEnvironment, RealTaskOracle, RealTaskTurn


class TestRealEnvironment(unittest.TestCase):
    """Test suite for RealTaskDOMEnvironment and RealTaskOracle."""

    def setUp(self):
        self.env = RealTaskDOMEnvironment(
            raw_mind2web_dir="data/raw_mind2web",
            raw_webarena_path="data/raw_webarena/test.raw.json",
            max_elements=32,
            hidden_dim=512,
            max_tasks=20,
            seed=42,
        )

    def test_task_init(self):
        """Verifies that RealTaskDOMEnvironment initializes tasks and parses initial DOM state."""
        self.assertGreater(len(self.env.tasks), 0, "Environment must load at least one real web task.")

        step_res = self.env.reset(task_idx=0)
        self.assertIsNotNone(step_res.obs_text)
        self.assertTrue(step_res.obs_text.startswith("URL:"))
        self.assertIn("DOM:", step_res.obs_text)
        self.assertEqual(step_res.obs_features.shape, (512,))
        self.assertFalse(step_res.done)
        self.assertGreater(len(self.env.active_elements), 0)
        self.assertIsNotNone(self.env.current_goal)
        print("PASS: test_task_init - Task initialized with goal:", self.env.current_goal)

    def test_oracle_rewards(self):
        """Verifies multi-attribute programmatic oracle reward computation."""
        oracle = RealTaskOracle(sigma=0.10)
        target_box = (0.20, 0.30, 0.40, 0.50)  # center = (0.30, 0.40)

        turn = RealTaskTurn(
            turn_id=0,
            url="https://www.example.com",
            html="<html><body><button id='target-btn' bbox='[200,300,400,500]'>Confirm</button></body></html>",
            thought="Click confirm button",
            action_type=ActionType.CLICK,
            target_id=1,
            target_bbox=target_box,
            argument="",
        )

        elements = self.env.active_elements

        # 1. Exact center click with correct action type
        center_coords = (0.30, 0.40)
        rew_perfect, info_perfect = oracle.compute_step_reward(
            action_type=ActionType.CLICK,
            coordinates=center_coords,
            argument="",
            expected_turn=turn,
            available_elements=elements,
        )
        self.assertTrue(info_perfect["step_success"])
        self.assertTrue(info_perfect["is_inside_target"])
        self.assertGreater(rew_perfect, 2.0)

        # 2. Wrong action type (TYPE instead of CLICK) and off-target click
        off_coords = (0.85, 0.90)
        rew_wrong, info_wrong = oracle.compute_step_reward(
            action_type=ActionType.TYPE,
            coordinates=off_coords,
            argument="wrong",
            expected_turn=turn,
            available_elements=elements,
        )
        self.assertFalse(info_wrong["step_success"])
        self.assertFalse(info_wrong["is_inside_target"])
        self.assertLess(rew_wrong, rew_perfect)
        print("PASS: test_oracle_rewards - Perfect reward:", rew_perfect, "| Wrong reward:", rew_wrong)

    def test_environment_goal_criteria(self):
        """Verifies referee-queryable goal criteria on WebOSEnvironment and RealTaskDOMEnvironment."""
        from jed_m2.environment import WebOSEnvironment
        webos = WebOSEnvironment(scenario_id="ecommerce")
        webos.reset()
        self.assertFalse(webos.check_goal_criteria())

        # Step to results
        submit_btn = [el for el in webos.current_state.elements if el.role == "submit_btn"][0]
        webos.step(ActionType.CLICK, coordinates=submit_btn.center)
        self.assertFalse(webos.check_goal_criteria())

        # Step to details
        item_card = [el for el in webos.current_state.elements if el.role in ("first_product_link", "item_card")][0]
        webos.step(ActionType.CLICK, coordinates=item_card.center)
        self.assertFalse(webos.check_goal_criteria())

        # Step to checkout
        checkout_btn = [el for el in webos.current_state.elements if el.role in ("checkout_btn", "proceed_checkout")][0]
        res = webos.step(ActionType.CLICK, coordinates=checkout_btn.center)
        self.assertTrue(webos.check_goal_criteria())
        self.assertTrue(res.info["goal_criteria_met"])

        # RealTaskDOMEnvironment check
        self.env.reset(task_idx=0)
        self.assertFalse(self.env.check_goal_criteria())
        print("PASS: test_environment_goal_criteria - Verified goal criteria queries")

    def test_manifest_driven_loading(self):
        """Verifies RealTaskDOMEnvironment loads authentic tasks directly from manifest split."""
        manifest_env = RealTaskDOMEnvironment(
            manifest_path="data/manifests/real_mind2web_v1/manifest.json",
            split="test",
            real_only=True,
            max_tasks=10,
        )
        self.assertGreater(len(manifest_env.tasks), 0)
        self.assertLessEqual(len(manifest_env.tasks), 10)
        for task in manifest_env.tasks:
            self.assertEqual(task.dataset_source, "mind2web")
            self.assertFalse(task.task_id.startswith("mock_"))
            self.assertTrue(len(task.turns) > 0)
        print("PASS: test_manifest_driven_loading - Loaded genuine test tasks from manifest")

    def test_real_only_zero_mock_fallback(self):
        """Verifies RealTaskDOMEnvironment strictly raises errors and rejects mock fixtures in real_only mode."""
        # 1. Missing manifest raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            RealTaskDOMEnvironment(
                manifest_path="nonexistent_manifest.json",
                real_only=True,
            )

        # 2. Missing raw directory raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            RealTaskDOMEnvironment(
                manifest_path="data/manifests/real_mind2web_v1/manifest.json",
                raw_mind2web_dir="nonexistent_raw_dir",
                real_only=True,
            )

        # 3. Procedural mock tasks loaded only when real_only=False
        fallback_env = RealTaskDOMEnvironment(
            manifest_path=None,
            raw_mind2web_dir=None,
            raw_webarena_path=None,
            real_only=False,
        )
        self.assertGreater(len(fallback_env.tasks), 0)
        self.assertTrue(fallback_env.tasks[0].task_id.startswith("mock_"))
        print("PASS: test_real_only_zero_mock_fallback - Enforced zero mock fallback in real mode")


if __name__ == "__main__":
    unittest.main()
