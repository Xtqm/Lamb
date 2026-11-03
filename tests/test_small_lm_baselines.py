"""
Unit tests for JED-M2 Small Language Model Baselines (T5-Small & DistilGPT-2).
Verifies:
- DOMActionSequenceFormatter serialization of goals, DOM elements, and targets.
- parse_action_string extraction of thoughts, discrete ActionTypes, coordinates, and arguments.
- T5SmallActionAgent and DistilGPT2ActionAgent forward loss and action generation.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from jed_m2.core.types import UIElement
from jed_m2.heads import ActionType
from jed_m2.small_lm_baselines import (
    DOMActionSequenceFormatter,
    DistilGPT2ActionAgent,
    ParsedAction,
    T5SmallActionAgent,
    parse_action_string,
)


class TestSmallLMBaselines(unittest.TestCase):
    """Test suite for T5-Small and DistilGPT-2 web agent baselines."""

    @classmethod
    def setUpClass(cls):
        cls.formatter = DOMActionSequenceFormatter()
        cls.sample_elements = [
            UIElement(
                element_id=0,
                tag="input",
                role="searchbox",
                text="Search items...",
                bbox=(0.10, 0.20, 0.40, 0.25),
                is_input=True,
            ),
            UIElement(
                element_id=1,
                tag="button",
                role="submit_btn",
                text="Search",
                bbox=(0.42, 0.20, 0.50, 0.25),
                is_clickable=True,
            ),
            UIElement(
                element_id=2,
                tag="a",
                role="link",
                text="Special Deals",
                bbox=(0.60, 0.05, 0.75, 0.10),
                is_clickable=True,
            ),
        ]

    def test_sequence_formatter(self):
        """Verifies DOMActionSequenceFormatter formats inputs and target outputs cleanly."""
        prompt = self.formatter.format_turn_prompt(
            goal="Find wireless noise cancelling headphones",
            elements=self.sample_elements,
            history=["CLICK (0.100, 0.100)"],
        )
        self.assertIn("[GOAL] Find wireless noise cancelling headphones", prompt)
        self.assertIn("[OBS] [ID 0] input role=\"searchbox\"", prompt)
        self.assertIn("[ID 1] button role=\"submit_btn\"", prompt)
        self.assertIn("[HISTORY] CLICK (0.100, 0.100)", prompt)

        target = self.formatter.format_target_action(
            action_type=ActionType.TYPE,
            coordinates=(0.250, 0.225),
            argument="Sony WH-1000XM5",
            thought="Type product name into the search box",
        )
        self.assertIn("<thought> Type product name into the search box </thought>", target)
        self.assertIn("[ACTION] TYPE (0.250, 0.225)", target)
        self.assertIn("[ARG] Sony WH-1000XM5 [/ARG]", target)

    def test_parse_action_string(self):
        """Verifies parse_action_string extracts discrete and continuous components."""
        # 1. Perfectly formatted string
        text1 = (
            "<thought> Click search button </thought> "
            "[ACTION] CLICK (0.460, 0.225) [ARG]  [/ARG]"
        )
        act1 = parse_action_string(text1, self.sample_elements)
        self.assertEqual(act1.action_type, ActionType.CLICK)
        self.assertAlmostEqual(act1.coordinates[0], 0.460, places=3)
        self.assertAlmostEqual(act1.coordinates[1], 0.225, places=3)
        self.assertEqual(act1.thought, "Click search button")
        self.assertEqual(act1.argument, "")
        self.assertTrue(act1.is_valid)

        # 2. TYPE action with argument
        text2 = "[ACTION] TYPE (0.250, 0.225) [ARG] running shoes [/ARG]"
        act2 = parse_action_string(text2, self.sample_elements)
        self.assertEqual(act2.action_type, ActionType.TYPE)
        self.assertEqual(act2.argument, "running shoes")
        self.assertTrue(act2.is_valid)

        # 3. FINISH action
        text3 = "<thought> Task is done </thought> [ACTION] FINISH (0.000, 0.000)"
        act3 = parse_action_string(text3, self.sample_elements)
        self.assertEqual(act3.action_type, ActionType.FINISH)
        self.assertTrue(act3.is_valid)

        # 4. Fallback on missing coordinates uses first candidate element center
        text4 = "I will click the button"
        act4 = parse_action_string(text4, self.sample_elements)
        self.assertEqual(act4.action_type, ActionType.CLICK)
        self.assertEqual(act4.coordinates, self.sample_elements[0].center)

    def test_t5_small_agent(self):
        """Verifies T5SmallActionAgent initialization, parameter count, loss, and inference."""
        agent = T5SmallActionAgent(local_files_only=True)
        # Verify parameter count is within ~60M - ~65M range
        self.assertGreater(agent.total_parameters, 55_000_000)
        self.assertLess(agent.total_parameters, 70_000_000)

        # Forward loss
        prompts = [
            self.formatter.format_turn_prompt("Search items", self.sample_elements),
            self.formatter.format_turn_prompt("Checkout cart", self.sample_elements),
        ]
        targets = [
            self.formatter.format_target_action(ActionType.CLICK, (0.45, 0.22)),
            self.formatter.format_target_action(ActionType.FINISH, (0.0, 0.0)),
        ]
        loss = agent.compute_loss(prompts, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        # Inference act
        parsed = agent.act(prompts[0], available_elements=self.sample_elements, max_new_tokens=16)
        self.assertIsInstance(parsed, ParsedAction)
        self.assertIsInstance(parsed.action_type, ActionType)
        self.assertIsInstance(parsed.coordinates, tuple)
        self.assertEqual(len(parsed.coordinates), 2)

    def test_distilgpt2_agent(self):
        """Verifies DistilGPT2ActionAgent initialization, parameter count, loss, and inference."""
        agent = DistilGPT2ActionAgent(local_files_only=True)
        # Verify parameter count is within ~80M - ~85M range
        self.assertGreater(agent.total_parameters, 75_000_000)
        self.assertLess(agent.total_parameters, 90_000_000)

        # Forward loss
        prompts = [
            self.formatter.format_turn_prompt("Search items", self.sample_elements),
        ]
        targets = [
            self.formatter.format_target_action(ActionType.CLICK, (0.45, 0.22)),
        ]
        loss = agent.compute_loss(prompts, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0.0)

        # Inference act
        parsed = agent.act(prompts[0], available_elements=self.sample_elements, max_new_tokens=16)
        self.assertIsInstance(parsed, ParsedAction)
        self.assertIsInstance(parsed.action_type, ActionType)
        self.assertIsInstance(parsed.coordinates, tuple)
        self.assertEqual(len(parsed.coordinates), 2)

    def test_small_lm_agent_wrapper(self):
        """Verifies SmallLMAgentWrapper conforms to JEDM2Agent contract."""
        from jed_m2.agent import AgentStepOutput
        from jed_m2.small_lm_baselines import SmallLMAgentWrapper

        base_agent = T5SmallActionAgent(local_files_only=True)
        wrapper = SmallLMAgentWrapper(base_agent, device=torch.device("cpu"), dtype=torch.float32)

        wrapper.reset()
        self.assertEqual(len(wrapper.history), 0)

        step_out = wrapper.act(
            obs_text="[GOAL] Find item [OBS] [ID 0] button role=\"submit\"",
            candidate_elements=self.sample_elements,
        )
        self.assertIsInstance(step_out, AgentStepOutput)
        self.assertIsInstance(step_out.action_type, ActionType)
        self.assertIsInstance(step_out.action_type_id, int)
        self.assertIsInstance(step_out.action_name, str)
        self.assertIsInstance(step_out.coordinates, tuple)
        self.assertEqual(len(step_out.coordinates), 2)
        self.assertEqual(step_out.gate_prob, 1.0)
        self.assertGreaterEqual(step_out.num_thought_tokens, 0)
        self.assertEqual(len(wrapper.history), 1)

    def test_eval_benchmarks_baseline_integration(self):
        """Verifies eval_benchmarks.load_model correctly identifies and instantiates baselines."""
        from eval_benchmarks import is_distilgpt2_baseline, is_t5_baseline, load_model

        self.assertTrue(is_t5_baseline("t5_small"))
        self.assertTrue(is_t5_baseline("t5-small"))
        self.assertTrue(is_distilgpt2_baseline("distilgpt2"))
        self.assertTrue(is_distilgpt2_baseline("distilgpt-2"))

        t5_model, t5_label = load_model("t5_small", device=torch.device("cpu"), dtype=torch.float32)
        self.assertIsInstance(t5_model, T5SmallActionAgent)
        self.assertIn("T5-Small", t5_label)

        gpt2_model, gpt2_label = load_model("distilgpt2", device=torch.device("cpu"), dtype=torch.float32)
        self.assertIsInstance(gpt2_model, DistilGPT2ActionAgent)
        self.assertIn("DistilGPT-2", gpt2_label)

    def test_offline_replay_with_baseline_wrapper(self):
        """Verifies evaluate_offline_dataset executes with SmallLMAgentWrapper without errors."""
        from jed_m2.benchmark import TaskEvaluator, evaluate_offline_dataset
        from jed_m2.core.types import InteractionTurn
        from jed_m2.dataset import TrajectorySample
        from jed_m2.small_lm_baselines import SmallLMAgentWrapper

        base_agent = T5SmallActionAgent(local_files_only=True)
        wrapper = SmallLMAgentWrapper(base_agent, device=torch.device("cpu"), dtype=torch.float32)

        sample = TrajectorySample(
            sample_id="test_sample_0",
            goal="Click the search button",
            turns=[
                InteractionTurn(
                    turn_id=0,
                    obs_text="[GOAL] Click search [OBS] [ID 0] button role=\"search\" bbox=[0.42, 0.20, 0.50, 0.25]",
                    thought="Click button",
                    action_type=ActionType.CLICK,
                    coordinates=(0.46, 0.22),
                    argument="",
                    dom_elements=self.sample_elements,
                ),
            ],
        )

        evaluator = TaskEvaluator(tolerance_radius=0.05)
        summary = evaluate_offline_dataset(
            agent=wrapper,
            trajectories=[sample],
            evaluator=evaluator,
            benchmark_name="test_baseline_replay",
            checkpoint_name="T5-Small-Test",
            max_episodes=1,
            max_steps=1,
        )

        self.assertEqual(summary.num_episodes, 1)
        self.assertGreaterEqual(summary.mean_step_success_rate, 0.0)
        self.assertGreater(summary.mean_decision_latency_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
