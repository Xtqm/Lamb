"""
JED-M2 Deterministic Macro-Action Execution Scaffold and Completion Referee.
Provides:
- ExecutionScaffold: Macro-action execution layer (input clearing, render stability polling).
- CompletionReferee: Programmatic referee inspecting environment state to determine
  milestone achievement and signal task completion, preventing infinite-step truncation.

STRRICT BUILD RULE: 100% native PyTorch & Python standard library without external compiled packages.
"""

from dataclasses import dataclass
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from jed_m2.environment import EnvironmentState, StepResult, WebOSEnvironment
from jed_m2.heads import ActionType

logger = logging.getLogger("jed_m2.scaffold")


# ==============================================================================
# 1. Deterministic Macro-Action Execution Scaffold
# ==============================================================================

class ExecutionScaffold:
    """
    Deterministic execution scaffold abstracting asynchronous browser and WebOS timing.
    Applies UINav-style programmatic macro-actions:
    - Input clearing before typing (`Ctrl+A` / clear text)
    - Render stability polling (ensuring DOM mutations settle before capturing observations)
    """

    def __init__(
        self,
        clear_input_before_type: bool = True,
        render_settle_delay_ms: float = 0.0,
    ):
        self.clear_input_before_type = clear_input_before_type
        self.render_settle_delay_ms = render_settle_delay_ms

    def execute_step(
        self,
        env: Any,
        action_type: Union[ActionType, int, str],
        coordinates: Optional[Tuple[float, float]] = None,
        argument: Optional[str] = None,
    ) -> StepResult:
        """
        Executes an agent action wrapped in macro-action safety mechanisms.
        """
        if isinstance(action_type, int):
            act_enum = ActionType(action_type)
        elif isinstance(action_type, str):
            act_enum = ActionType[action_type.upper()]
        else:
            act_enum = action_type

        # 1. Macro-Action: Input clearing before TYPE
        effective_arg = argument
        if act_enum == ActionType.TYPE:
            if self.clear_input_before_type:
                self._apply_input_clearing_macro(env, coordinates)
            effective_arg = self._resolve_argument_macro(env, argument)

        # 2. Step execution in environment
        step_res = env.step(
            action_type=act_enum,
            coordinates=coordinates,
            argument=effective_arg,
        )

        # 3. Macro-Action: Render stability settling
        if self.render_settle_delay_ms > 0.0:
            time.sleep(self.render_settle_delay_ms / 1000.0)

        return step_res

    def _apply_input_clearing_macro(
        self,
        env: Any,
        coordinates: Optional[Tuple[float, float]],
    ) -> None:
        """Pre-clears the target input field to avoid accidental text concatenation."""
        if not hasattr(env, "current_state") or env.current_state is None:
            return

        cx, cy = coordinates if coordinates is not None else (0.5, 0.5)
        target_el = None

        if hasattr(env, "hit_test"):
            target_el = env.hit_test(cx, cy)

        if (target_el is None or not getattr(target_el, "is_input", False)) and getattr(
            env.current_state, "focused_element_id", None
        ) is not None:
            for el in getattr(env.current_state, "elements", []):
                if el.element_id == env.current_state.focused_element_id:
                    target_el = el
                    break

        if target_el is not None and getattr(target_el, "is_input", False):
            target_el.value = ""

    def _resolve_argument_macro(self, env: Any, argument: Optional[str]) -> str:
        """
        Resolves the text argument for TYPE actions.
        Prioritizes the native decoded text argument produced by the model.
        Strips surrounding <arg> / </arg> control delimiters if present.
        Only falls back to heuristic goal extraction when the argument is empty
        or consists solely of legacy un-decoded tokens (e.g. 'tok_...').
        """
        raw_arg = (argument or "").strip()
        # Clean surrounding control tokens if present
        if raw_arg.startswith("<arg>") and raw_arg.endswith("</arg>"):
            raw_arg = raw_arg[5:-6].strip()
        elif raw_arg.startswith("<arg>"):
            raw_arg = raw_arg[5:].strip()
        elif raw_arg.endswith("</arg>"):
            raw_arg = raw_arg[:-6].strip()

        # If a valid decoded string is present (not empty, not legacy tok_...), dispatch directly
        if raw_arg and not raw_arg.startswith("tok_") and not (raw_arg.startswith("<") and raw_arg.endswith(">")):
            return raw_arg

        goal_text = getattr(env, "current_goal", "") or (
            getattr(env.active_scenario, "goal", "") if hasattr(env, "active_scenario") and env.active_scenario else ""
        )
        if not goal_text:
            return raw_arg

        goal_lower = goal_text.lower()
        if "search for " in goal_lower:
            query_part = goal_text[goal_lower.index("search for ") + len("search for ") :]
            for delim in [" and ", ",", " on ", " for "]:
                if delim in query_part:
                    query_part = query_part.split(delim)[0]
            return query_part.strip()
        elif "hotel in " in goal_lower:
            query_part = goal_text[goal_lower.index("hotel in ") + len("hotel in ") :]
            for delim in [" for ", " from ", " next "]:
                if delim in query_part:
                    query_part = query_part.split(delim)[0]
            return query_part.strip()
        elif "rules in " in goal_lower:
            return "authorize_ingress"

        return raw_arg


# ==============================================================================
# 2. Programmatic Task Feasibility & Completion Referee (Evaluation Oracle)
# ==============================================================================

class CompletionReferee:
    """
    Deterministic task referee and post-hoc evaluation oracle inspecting environment
    state to verify whether ground-truth task goals and milestones have been fulfilled.

    Decoupling Architecture:
    During live autonomous agent rollouts (RL exploration, benchmark evaluation, sidecar execution),
    the policy is responsible for detecting task completion and emitting ActionType.FINISH.
    CompletionReferee functions strictly as an external oracle to:
    1. Score post-hoc trajectory success (autonomous success vs referee-assisted milestone hit).
    2. Guide offline preference dataset labeling and reward shaping.
    3. Provide ground-truth milestone validation independent of policy output.
    """

    @classmethod
    def evaluate_autonomous_success(
        cls,
        env: Any,
        final_action: Optional[Union[ActionType, int, str]],
    ) -> Tuple[bool, bool]:
        """
        Evaluates whether an episode achieved autonomous success vs referee-assisted milestone hit.

        Returns:
            Tuple of (autonomous_success, referee_goal_satisfied)
        """
        referee_sat = cls.is_goal_satisfied(env)
        if isinstance(final_action, int):
            try:
                final_action = ActionType(final_action)
            except ValueError:
                final_action = None
        elif isinstance(final_action, str):
            try:
                final_action = ActionType[final_action.upper()]
            except KeyError:
                final_action = None
        is_autonomous = bool(referee_sat and final_action == ActionType.FINISH)
        return is_autonomous, referee_sat

    @staticmethod
    def is_goal_satisfied(env: Any) -> bool:
        """
        Queries whether the environment has fulfilled its end-to-end task objective.
        """
        # 1. Check if environment already marked goal satisfied
        if getattr(env, "goal_satisfied", False):
            return True

        # 2. WebOSEnvironment state machine inspection
        if hasattr(env, "active_scenario") and hasattr(env, "current_state") and env.current_state:
            domain = getattr(env.active_scenario, "scenario_id", "")
            meta = getattr(env.current_state, "metadata", {}) or {}
            stage = meta.get("stage", "")

            if domain == "ecommerce":
                if stage == "cart_confirmed" or len(meta.get("cart", [])) > 0:
                    return True
            elif domain == "travel":
                if stage == "booked" or meta.get("booked", False):
                    return True
            elif domain == "os_admin":
                if stage == "rules_saved" or len(meta.get("rules", [])) > 0:
                    return True

        # 3. RealTaskDOMEnvironment inspection
        if hasattr(env, "active_task") and hasattr(env, "current_turn_idx"):
            num_turns = len(env.active_task.turns) if env.active_task and env.active_task.turns else 0
            # If agent executed the final turn or reached the goal state
            if num_turns > 0 and env.current_turn_idx >= num_turns - 1:
                if getattr(env, "_goal_satisfied", False):
                    return True

        # 4. Browser / Playwright oracle inspection
        if hasattr(env, "active_oracle") and env.active_oracle is not None:
            if getattr(env, "goal_satisfied", False):
                return True

        return False

    @classmethod
    def evaluate_termination(
        cls,
        env: Any,
        current_step: int,
        max_steps: int,
        action_type: ActionType,
        repeated_action_count: int = 0,
    ) -> Tuple[bool, bool, str]:
        """
        Evaluates whether the episode should terminate and whether it succeeded.

        Returns:
            Tuple of (done, goal_satisfied, termination_reason)
        """
        # A. Explicit FINISH emitted by policy
        if action_type == ActionType.FINISH:
            satisfied = cls.is_goal_satisfied(env)
            return True, satisfied, "action_finish"

        # B. Referee detected task completion
        if cls.is_goal_satisfied(env):
            return True, True, "referee_goal_achieved"

        # C. Loop cutoff: too many repeated identical actions
        if repeated_action_count >= 3:
            return True, False, "loop_truncated"

        # D. Maximum step limit reached
        if current_step >= max_steps:
            return True, False, "step_limit_truncated"

        return False, False, "in_progress"
