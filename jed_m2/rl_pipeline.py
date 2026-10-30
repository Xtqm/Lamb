"""
JED-M2 Interactive RL Pipeline.
Provides:
- RolloutTurn: Record of a single turn in an interactive environment rollout.
- RolloutTrajectory: Multi-turn trajectory capturing tokens, log-probabilities, actions,
  coordinates, arguments, step rewards, and goal satisfaction for RL optimization.
- TrajectoryRolloutSampler: Executes stochastic rollouts in WebOSEnvironment with
  temperature sampling, collects groups of G rollouts per instruction goal for GRPO,
  pairs winning and losing trajectories for DPO, and serializes trajectories into
  batched training tensors.

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.environment import ScenarioDefinition, StepResult, WebOSEnvironment, get_default_scenarios
from jed_m2.heads import ActionOutput, ActionType
from jed_m2.model import JEDM2Model
from jed_m2.scaffold import CompletionReferee, ExecutionScaffold
from jed_m2.rl import (
    compute_action_type_log_probs,
    compute_coordinate_log_probs,
    compute_point_box_iou,
    compute_shaped_reward,
    compute_token_log_probs,
)


# ==============================================================================
# 1. Rollout Data Containers
# ==============================================================================

@dataclass
class RolloutTurn:
    """Record of a single interaction turn within an agent rollout."""

    turn_id: int
    obs_text: str
    thought_tokens: List[int]
    thought_text: str
    action_type: ActionType
    action_type_id: int
    coordinates: Tuple[float, float]
    argument: str
    step_tokens: List[int]
    step_mask: List[bool]
    reward: float
    done: bool
    gate_prob: float
    coord_logp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes turn record to dictionary."""
        return {
            "turn_id": self.turn_id,
            "obs_text": self.obs_text,
            "thought_tokens": self.thought_tokens,
            "thought_text": self.thought_text,
            "action_type": self.action_type.name,
            "action_type_id": self.action_type_id,
            "coordinates": self.coordinates,
            "coord_logp": self.coord_logp,
            "argument": self.argument,
            "reward": self.reward,
            "done": self.done,
            "gate_prob": self.gate_prob,
        }


@dataclass
class RolloutTrajectory:
    """Complete multi-turn agent rollout trajectory for RL policy optimization."""

    scenario_id: str
    goal: str
    turns: List[RolloutTurn]
    full_token_ids: List[int]
    trainable_mask: List[bool]
    step_rewards: List[float]
    cumulative_reward: float
    terminal_reward: float
    success: bool
    num_steps: int

    def to_dict(self) -> Dict[str, Any]:
        """Serializes complete trajectory record to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "goal": self.goal,
            "turns": [turn.to_dict() for turn in self.turns],
            "full_token_ids": self.full_token_ids,
            "trainable_mask": self.trainable_mask,
            "step_rewards": self.step_rewards,
            "cumulative_reward": self.cumulative_reward,
            "terminal_reward": self.terminal_reward,
            "success": self.success,
            "num_steps": self.num_steps,
        }


# ==============================================================================
# 2. Interactive Trajectory Rollout Sampler
# ==============================================================================

class TrajectoryRolloutSampler:
    """
    Executes parallel or batched interactive rollouts in WebOSEnvironment with temperature sampling.
    Collects groups of G rollouts per instruction goal for Group Relative Policy Optimization (GRPO).
    Pairs winning (success=1.0) and losing (success=0.0) trajectories for Direct Policy Optimization (DPO).
    """

    def __init__(
        self,
        model: JEDM2Model,
        config: Optional[JEDM2Config] = None,
        tokenizer: Optional[SimpleLAMTokenizer] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
        max_thought_tokens: int = 16,
        action_gate_threshold: float = 0.5,
    ):
        self.model = model
        self.config = config if config is not None else getattr(model, "config", JEDM2Config())
        self.tokenizer = tokenizer if tokenizer is not None else SimpleLAMTokenizer(self.config.vocab_size)

        if device is None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype
        self.max_thought_tokens = max_thought_tokens
        self.action_gate_threshold = action_gate_threshold
        self.scenarios = get_default_scenarios()
        self.scaffold = ExecutionScaffold(clear_input_before_type=True)
        self.referee = CompletionReferee()

    def sample_trajectory(
        self,
        env: Any,
        temperature: float = 0.7,
        max_turns: int = 8,
        scenario_id: Optional[str] = None,
    ) -> RolloutTrajectory:
        """
        Executes a single interactive episode in WebOSEnvironment with temperature-scaled exploration.

        Args:
            env: WebOSEnvironment or PlaywrightEnvironment instance.
            temperature: Sampling temperature for System 2 rationale tokens and System 1 action primitives.
            max_turns: Maximum allowed interaction turns before truncation.
            scenario_id: Optional scenario identifier for scenario/fixture routing.

        Returns:
            RolloutTrajectory with token sequence, loss masks, and reward signals.
        """
        self.model.eval()

        # Reset environment with optional scenario routing
        init_res = env.reset(scenario_id=scenario_id) if scenario_id else env.reset()
        obs_text = init_res.obs_text

        # Recurrent state buffers for O(1) step inference
        recurrent_states = None
        current_pos = 0

        # Accumulated trajectory tokens and loss masks
        all_tokens: List[int] = [self.tokenizer.BOS_TOKEN_ID]
        all_mask: List[bool] = [False]  # BOS is not trainable
        turns: List[RolloutTurn] = []
        step_rewards: List[float] = []

        # Goal conditioning matching pretraining grammar
        goal_text = getattr(env, "current_goal", None) or (
            getattr(env.active_scenario, "goal", "") if hasattr(env, "active_scenario") and env.active_scenario else ""
        )
        if goal_text:
            goal_prompt = f"Goal: {goal_text}"
            goal_tokens = self.tokenizer.encode(goal_prompt)[:64]
            all_tokens.extend(goal_tokens)
            all_mask.extend([False] * len(goal_tokens))
            goal_tensor = torch.tensor(goal_tokens, device=self.device, dtype=torch.long)
            for i in range(len(goal_tokens)):
                with torch.no_grad():
                    last_lm_logits, last_action_out, recurrent_states = self.model.step(
                        goal_tensor[i : i + 1],
                        states=recurrent_states,
                        pos=current_pos,
                    )
                current_pos += 1

        end_thought_id = self.tokenizer.special_tokens["</thought>"]

        for turn_idx in range(max_turns):
            if current_pos >= self.config.max_seq_len - 64:
                break
            turn_tokens: List[int] = []
            turn_mask: List[bool] = []

            # ------------------------------------------------------------------
            # 1. Observation Prompt Ingestion (budget up to 256 tokens)
            # ------------------------------------------------------------------
            obs_prompt = f"<obs> {obs_text} </obs> <thought>"
            obs_tokens = self.tokenizer.encode(obs_prompt)[:256]

            all_tokens.extend(obs_tokens)
            all_mask.extend([False] * len(obs_tokens))
            turn_tokens.extend(obs_tokens)
            turn_mask.extend([False] * len(obs_tokens))

            # Step model recurrently through observation tokens
            last_lm_logits = None
            last_action_out = None
            obs_tensor = torch.tensor(obs_tokens, device=self.device, dtype=torch.long)
            for i in range(len(obs_tokens)):
                with torch.no_grad():
                    last_lm_logits, last_action_out, recurrent_states = self.model.step(
                        obs_tensor[i : i + 1],
                        states=recurrent_states,
                        pos=current_pos,
                    )
                current_pos += 1

            lm_logits = last_lm_logits
            action_output = last_action_out

            # ------------------------------------------------------------------
            # 2. System 2: Autoregressive Latent Reasoning with Temperature
            # ------------------------------------------------------------------
            thought_tokens: List[int] = []
            gate_prob = float(torch.sigmoid(action_output.gate_logits).item())

            while (
                gate_prob < self.action_gate_threshold
                and len(thought_tokens) < self.max_thought_tokens
            ):
                # Sample next thought token
                if temperature > 0.0:
                    probs = F.softmax(lm_logits / temperature, dim=-1)
                    next_token_id = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    next_token_id = int(torch.argmax(lm_logits, dim=-1).item())

                if next_token_id == end_thought_id:
                    break

                thought_tokens.append(next_token_id)
                all_tokens.append(next_token_id)
                all_mask.append(True)  # Trainable policy token
                turn_tokens.append(next_token_id)
                turn_mask.append(True)

                token_tensor = torch.tensor([next_token_id], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    lm_logits, action_output, recurrent_states = self.model.step(
                        token_tensor,
                        states=recurrent_states,
                        pos=current_pos,
                    )
                current_pos += 1
                gate_prob = float(torch.sigmoid(action_output.gate_logits).item())

            # ------------------------------------------------------------------
            # 3. System 1: Action Primitive, Coordinates & Argument Decoding
            # ------------------------------------------------------------------
            # Advance through </thought> (if needed) and step into <action> trigger token
            # where the System 1 action heads were trained to emit decisions
            if not thought_tokens or thought_tokens[-1] != end_thought_id:
                token_tensor = torch.tensor([end_thought_id], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    lm_logits, action_output, recurrent_states = self.model.step(
                        token_tensor, states=recurrent_states, pos=current_pos
                    )
                current_pos += 1
                all_tokens.append(end_thought_id)
                all_mask.append(False)
                turn_tokens.append(end_thought_id)
                turn_mask.append(False)

            action_token_id = self.tokenizer.special_tokens["<action>"]
            token_tensor = torch.tensor([action_token_id], device=self.device, dtype=torch.long)
            with torch.no_grad():
                lm_logits, action_output, recurrent_states = self.model.step(
                    token_tensor, states=recurrent_states, pos=current_pos
                )
            current_pos += 1
            all_tokens.append(action_token_id)
            all_mask.append(False)
            turn_tokens.append(action_token_id)
            turn_mask.append(False)

            sigma = 0.05 * max(0.2, temperature)
            if temperature > 0.0:
                act_probs = F.softmax(action_output.action_type_logits / temperature, dim=-1)
                action_type_id = int(torch.multinomial(act_probs, num_samples=1).item())
                action_type = ActionType(action_type_id)

                raw_coords = action_output.coordinates.squeeze(0)  # [2]
                coord_noise = torch.randn(2, device=self.device) * sigma
                coord_x = float((raw_coords[0] + coord_noise[0]).clamp(0.0, 1.0).item())
                coord_y = float((raw_coords[1] + coord_noise[1]).clamp(0.0, 1.0).item())

                # Continuous Gaussian spatial policy log-density
                var = max(1e-6, sigma ** 2)
                sq_err = (coord_x - raw_coords[0].item()) ** 2 + (coord_y - raw_coords[1].item()) ** 2
                coord_logp = float(-0.5 * (sq_err / var + 2.0 * math.log(2.0 * math.pi * var)))

                arg_probs = F.softmax(action_output.argument_logits / temperature, dim=-1)
                arg_token_id = int(torch.multinomial(arg_probs, num_samples=1).item())
                arg_str = self.tokenizer.decode([arg_token_id]).strip()
            else:
                action_type_id = int(torch.argmax(action_output.action_type_logits, dim=-1).item())
                action_type = ActionType(action_type_id)

                raw_coords = action_output.coordinates.squeeze(0)  # [2]
                coord_x = float(raw_coords[0].clamp(0.0, 1.0).item())
                coord_y = float(raw_coords[1].clamp(0.0, 1.0).item())
                coord_logp = 0.0

                arg_token_id = int(torch.argmax(action_output.argument_logits, dim=-1).item())
                arg_str = self.tokenizer.decode([arg_token_id]).strip()

            coords = (coord_x, coord_y)
            cand_list = None
            if hasattr(env, "current_state") and env.current_state is not None:
                cand_list = getattr(env.current_state, "elements", None)
            if cand_list is None and hasattr(env, "active_elements"):
                cand_list = getattr(env, "active_elements", None)
            if cand_list and action_type in (ActionType.CLICK, ActionType.SELECT, ActionType.TYPE):
                valid_cands = []
                for e in cand_list:
                    if getattr(e, "has_bbox", True):
                        if hasattr(e, "center"):
                            valid_cands.append((float(e.center[0]), float(e.center[1])))
                        elif hasattr(e, "bbox") and len(e.bbox) >= 4:
                            valid_cands.append(((float(e.bbox[0]) + float(e.bbox[2])) / 2.0, (float(e.bbox[1]) + float(e.bbox[3])) / 2.0))
                if valid_cands:
                    best_c = min(valid_cands, key=lambda c: (c[0] - coord_x) ** 2 + (c[1] - coord_y) ** 2)
                    if (best_c[0] - coord_x) ** 2 + (best_c[1] - coord_y) ** 2 <= 0.20 ** 2:
                        coords = (float(best_c[0]), float(best_c[1]))

            # Decode thought string
            thought_text = self.tokenizer.decode(thought_tokens).strip()

            # Dispatch action to environment via deterministic macro-action scaffold
            step_res = self.scaffold.execute_step(
                env=env,
                action_type=action_type,
                coordinates=coords,
                argument=arg_str,
            )
            step_rewards.append(step_res.reward)

            # Check repeated actions
            repeated_count = 0
            if turns:
                prev_turn = turns[-1]
                if prev_turn.action_type == action_type:
                    dist = math.hypot(coords[0] - prev_turn.coordinates[0], coords[1] - prev_turn.coordinates[1])
                    if dist < 0.05:
                        repeated_count += 1

            # Evaluate termination and goal status via CompletionReferee
            referee_done, referee_goal_satisfied, term_reason = self.referee.evaluate_termination(
                env=env,
                current_step=turn_idx + 1,
                max_steps=max_turns,
                action_type=action_type,
                repeated_action_count=repeated_count,
            )
            is_done = step_res.done or referee_done
            if referee_goal_satisfied and hasattr(env, "goal_satisfied"):
                env.goal_satisfied = True

            # Action primitive token (trainable)
            act_tok = self.tokenizer.encode(action_type.name)[0]
            all_tokens.append(act_tok)
            all_mask.append(True)  # Trainable policy action token
            turn_tokens.append(act_tok)
            turn_mask.append(True)

            token_tensor = torch.tensor([act_tok], device=self.device, dtype=torch.long)
            with torch.no_grad():
                lm_logits, action_output, recurrent_states = self.model.step(
                    token_tensor,
                    states=recurrent_states,
                    pos=current_pos,
                )
            current_pos += 1

            act_end_tokens = self.tokenizer.encode("</action>")
            all_tokens.extend(act_end_tokens)
            all_mask.extend([False] * len(act_end_tokens))
            turn_tokens.extend(act_end_tokens)
            turn_mask.extend([False] * len(act_end_tokens))

            for tok in act_end_tokens:
                token_tensor = torch.tensor([tok], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    lm_logits, action_output, recurrent_states = self.model.step(
                        token_tensor,
                        states=recurrent_states,
                        pos=current_pos,
                    )
                current_pos += 1

            # Argument tokens if present
            if arg_str:
                arg_open_toks = self.tokenizer.encode("<arg>")
                all_tokens.extend(arg_open_toks)
                all_mask.extend([False] * len(arg_open_toks))
                turn_tokens.extend(arg_open_toks)
                turn_mask.extend([False] * len(arg_open_toks))

                for tok in arg_open_toks:
                    token_tensor = torch.tensor([tok], device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        lm_logits, action_output, recurrent_states = self.model.step(
                            token_tensor,
                            states=recurrent_states,
                            pos=current_pos,
                        )
                    current_pos += 1

                arg_tokens = self.tokenizer.encode(arg_str)[:8]
                all_tokens.extend(arg_tokens)
                all_mask.extend([True] * len(arg_tokens))  # Trainable argument tokens
                turn_tokens.extend(arg_tokens)
                turn_mask.extend([True] * len(arg_tokens))

                for tok in arg_tokens:
                    token_tensor = torch.tensor([tok], device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        lm_logits, action_output, recurrent_states = self.model.step(
                            token_tensor,
                            states=recurrent_states,
                            pos=current_pos,
                        )
                    current_pos += 1

                arg_close_toks = self.tokenizer.encode("</arg>")
                all_tokens.extend(arg_close_toks)
                all_mask.extend([False] * len(arg_close_toks))
                turn_tokens.extend(arg_close_toks)
                turn_mask.extend([False] * len(arg_close_toks))

                for tok in arg_close_toks:
                    token_tensor = torch.tensor([tok], device=self.device, dtype=torch.long)
                    with torch.no_grad():
                        lm_logits, action_output, recurrent_states = self.model.step(
                            token_tensor,
                            states=recurrent_states,
                            pos=current_pos,
                        )
                    current_pos += 1

            record = RolloutTurn(
                turn_id=turn_idx + 1,
                obs_text=obs_text,
                thought_tokens=thought_tokens,
                thought_text=thought_text,
                action_type=action_type,
                action_type_id=action_type_id,
                coordinates=coords,
                argument=arg_str,
                step_tokens=turn_tokens,
                step_mask=turn_mask,
                reward=step_res.reward,
                done=is_done,
                gate_prob=gate_prob,
                coord_logp=coord_logp,
            )
            turns.append(record)

            obs_text = step_res.obs_text
            if is_done:
                break

        # Append terminal EOS token
        all_tokens.append(self.tokenizer.EOS_TOKEN_ID)
        all_mask.append(False)

        is_success = bool(getattr(env, "goal_satisfied", False) or CompletionReferee.is_goal_satisfied(env))
        terminal_reward = 1.0 if is_success else 0.0
        cumulative_reward = float(env.current_state.cumulative_reward) if env.current_state else 0.0

        return RolloutTrajectory(
            scenario_id=env.active_scenario.scenario_id,
            goal=env.active_scenario.goal,
            turns=turns,
            full_token_ids=all_tokens,
            trainable_mask=all_mask,
            step_rewards=step_rewards,
            cumulative_reward=cumulative_reward,
            terminal_reward=terminal_reward,
            success=is_success,
            num_steps=len(turns),
        )

    def sample_group_rollouts(
        self,
        scenario_id: str = "ecommerce",
        group_size: int = 4,
        temperature: float = 0.7,
        max_turns: int = 8,
        generator: Optional[Any] = None,
        seed: Optional[int] = None,
        env_type: str = "webos",
        browser_env: Optional[Any] = None,
    ) -> List[RolloutTrajectory]:
        """
        Samples a group of G rollout trajectories for a single scenario goal for GRPO.
        Supports built-in benchmark scenarios, procedural environment synthesis, and
        headless Playwright browser sandboxes.

        Args:
            scenario_id: Identifier of benchmark or procedural domain.
            group_size: Number of rollouts G to sample per group.
            temperature: Exploration sampling temperature.
            max_turns: Maximum allowed turns per episode.
            generator: Optional DynamicScenarioGenerator instance.
            seed: Optional integer seed for procedural generation.
            env_type: Environment backend: 'webos' (procedural mock) or 'browser' (Playwright).
            browser_env: Optional reusable PlaywrightWebEnvironment instance.

        Returns:
            List of G RolloutTrajectory instances.
        """
        group: List[RolloutTrajectory] = []

        if env_type == "browser":
            from jed_m2.playwright_env import PlaywrightWebEnvironment
            owns_browser = False
            if browser_env is not None:
                active_env = browser_env
            else:
                active_env = PlaywrightWebEnvironment(
                    config=self.config,
                    scenario_id=scenario_id,
                    device=self.device,
                    dtype=self.dtype,
                )
                owns_browser = True

            try:
                for _ in range(group_size):
                    traj = self.sample_trajectory(
                        active_env,
                        temperature=temperature,
                        max_turns=max_turns,
                        scenario_id=scenario_id,
                    )
                    group.append(traj)
            finally:
                if owns_browser:
                    active_env.close()

            return group

        if env_type == "real_web":
            from jed_m2.real_environment import RealTaskDOMEnvironment
            if browser_env is not None and isinstance(browser_env, RealTaskDOMEnvironment):
                real_env = browser_env
            else:
                real_env = RealTaskDOMEnvironment(
                    hidden_dim=self.config.hidden_dim,
                    max_tasks=100,
                    seed=seed or 42,
                )

            # Determine task for this group rollout so all G trajectories explore identical initial DOM
            target_task_id = (
                scenario_id
                if scenario_id and scenario_id.lower() not in ("ecommerce", "form_portal", "travel", "real_web")
                else None
            )

            if target_task_id:
                real_env.reset(task_id=target_task_id)
            elif scenario_id and scenario_id.isdigit():
                real_env.reset(task_idx=int(scenario_id))
            else:
                real_env.reset()

            fixed_task_id = real_env.active_task.task_id if real_env.active_task else None

            for _ in range(group_size):
                traj = self.sample_trajectory(
                    real_env,
                    temperature=temperature,
                    max_turns=max_turns,
                    scenario_id=fixed_task_id,
                )
                group.append(traj)

            return group

        for _ in range(group_size):
            if generator is not None:
                env = generator.generate(domain=scenario_id, seed=seed)
            else:
                env = WebOSEnvironment(
                    config=self.config,
                    scenario_id=scenario_id,
                    device=self.device,
                    dtype=self.dtype,
                )
            traj = self.sample_trajectory(env, temperature=temperature, max_turns=max_turns)
            group.append(traj)

        return group

    def collate_rollouts(
        self,
        trajectories: List[RolloutTrajectory],
        reward_mode: str = "shaped",
        cpu_offload: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Collates a list of rollout trajectories into batched tensors formatted for GRPOTrainer,
        including continuous spatial coordinate targets and dense shaped rewards.

        Args:
            trajectories: List of B RolloutTrajectory instances.
            reward_mode: 'shaped' (dense multi-factor), 'cumulative', or 'terminal'.

        Returns:
            Dictionary containing:
            - input_ids: [B, S]
            - targets: [B, S]
            - mask: [B, S] (bool)
            - old_token_logps: [B, S]
            - old_coord_logps: [B, S]
            - target_coords: [B, S, 2]
            - rewards: [B]
            - advantages: [B] (group-normalized advantages)
        """
        batch_size = len(trajectories)
        if batch_size == 0:
            raise ValueError("Cannot collate empty trajectory list.")
        batch_device = torch.device("cpu") if cpu_offload else self.device

        # Determine sequence lengths (input length = full_tokens - 1)
        seq_lens = [max(1, len(t.full_token_ids) - 1) for t in trajectories]
        max_seq_len = min(max(seq_lens), self.config.max_seq_len)

        padded_input_ids = torch.full(
            (batch_size, max_seq_len), self.tokenizer.PAD_TOKEN_ID, dtype=torch.long, device=batch_device
        )
        padded_targets = torch.full(
            (batch_size, max_seq_len), -100, dtype=torch.long, device=batch_device
        )
        padded_mask = torch.zeros(
            (batch_size, max_seq_len), dtype=torch.bool, device=batch_device
        )
        padded_target_coords = torch.zeros(
            (batch_size, max_seq_len, 2), dtype=torch.float32, device=batch_device
        )
        action_type_targets = torch.zeros(
            (batch_size, max_seq_len), dtype=torch.long, device=batch_device
        )
        action_mask = torch.zeros(
            (batch_size, max_seq_len), dtype=torch.bool, device=batch_device
        )

        raw_rewards: List[float] = []

        for b, traj in enumerate(trajectories):
            tokens = traj.full_token_ids
            mask = traj.trainable_mask

            # input_ids: tokens[:-1], targets: tokens[1:], mask: mask[1:]
            s_len = min(len(tokens) - 1, max_seq_len)
            if s_len > 0:
                padded_input_ids[b, :s_len] = torch.tensor(
                    tokens[:s_len], dtype=torch.long, device=batch_device
                )
                padded_targets[b, :s_len] = torch.tensor(
                    tokens[1 : s_len + 1], dtype=torch.long, device=batch_device
                )
                padded_mask[b, :s_len] = torch.tensor(
                    mask[1 : s_len + 1], dtype=torch.bool, device=batch_device
                )

            # Assign coordinate targets from turns
            turn_pos = 0
            for turn in traj.turns:
                t_len = len(turn.step_tokens)
                if turn_pos < max_seq_len:
                    padded_target_coords[b, turn_pos : min(max_seq_len, turn_pos + t_len)] = torch.tensor(
                        turn.coordinates, dtype=torch.float32, device=batch_device
                    )
                turn_pos += t_len

            # Align action labels against the actual model inputs. The action-type head
            # predicts after the <action> token, where the next-token loss is active.
            action_trigger_id = self.tokenizer.special_tokens["<action>"]
            if s_len > 0:
                trigger_positions = torch.nonzero(
                    (padded_input_ids[b, :s_len] == action_trigger_id)
                    & padded_mask[b, :s_len],
                    as_tuple=False,
                ).flatten().tolist()
                for action_pos, turn in zip(trigger_positions, traj.turns):
                    action_type_targets[b, action_pos] = int(turn.action_type_id)
                    action_mask[b, action_pos] = True

            # Guard: ensure at least one active token in mask to prevent zero-sum division
            if not padded_mask[b, :s_len].any() and s_len > 0:
                padded_mask[b, s_len - 1] = True

            if reward_mode == "shaped":
                # Compute dense multi-factor reward shaping
                repeated_count = 0
                for t_i in range(1, len(traj.turns)):
                    prev_t = traj.turns[t_i - 1]
                    curr_t = traj.turns[t_i]
                    if prev_t.action_type == curr_t.action_type:
                        dist = math.hypot(
                            curr_t.coordinates[0] - prev_t.coordinates[0],
                            curr_t.coordinates[1] - prev_t.coordinates[1],
                        )
                        if dist < 0.05:
                            repeated_count += 1

                milestone_count = sum(1 for t in traj.turns if t.reward > 0.0 and not t.done)
                type_matches = sum(1 for t in traj.turns if t.reward >= 0.1)
                arg_matches = sum(1 for t in traj.turns if t.argument and t.reward > 0.0)
                iou_estimate = sum(0.2 for t in traj.turns if t.reward >= 0.2)
                shaped_r = compute_shaped_reward(
                    goal_satisfied=traj.success,
                    milestones_count=milestone_count,
                    turn_count=len(traj.turns),
                    iou_sum=iou_estimate + (0.5 if traj.success else 0.0),
                    repeated_action_count=repeated_count,
                    action_type_matches=type_matches,
                    argument_matches=arg_matches,
                )
                raw_rewards.append(shaped_r)
            elif reward_mode == "terminal":
                raw_rewards.append(traj.terminal_reward)
            else:
                raw_rewards.append(traj.cumulative_reward)

        rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32, device=batch_device)

        # Compute old logps under rollout policy using forward pass in torch.no_grad()
        self.model.eval()
        if cpu_offload:
            old_token_logps = torch.zeros_like(padded_targets, dtype=torch.float32)
            old_coord_logps = torch.zeros_like(padded_targets, dtype=torch.float32)
            old_action_logps = torch.zeros_like(padded_targets, dtype=torch.float32)
            microbatch_size = min(2, batch_size)
            with torch.no_grad():
                for start in range(0, batch_size, microbatch_size):
                    end = min(start + microbatch_size, batch_size)
                    rollout_input_ids = padded_input_ids[start:end].to(self.device)
                    rollout_targets = padded_targets[start:end].to(self.device)
                    rollout_mask = padded_mask[start:end].to(self.device)
                    rollout_coords = padded_target_coords[start:end].to(self.device)
                    rollout_action_targets = action_type_targets[start:end].to(self.device)
                    rollout_action_mask = action_mask[start:end].to(self.device)
                    rollout_out = self.model(input_ids=rollout_input_ids)
                    token_logps, _ = compute_token_log_probs(
                        rollout_out.lm_logits, rollout_targets, rollout_mask
                    )
                    coord_logps = compute_coordinate_log_probs(
                        rollout_out.action_output.coordinates, rollout_coords, sigma=0.05
                    )
                    action_logps = compute_action_type_log_probs(
                        rollout_out.action_output.action_type_logits,
                        rollout_action_targets,
                        rollout_action_mask,
                    )
                    old_token_logps[start:end] = token_logps.cpu()
                    old_coord_logps[start:end] = coord_logps.cpu()
                    old_action_logps[start:end] = action_logps.cpu()
        else:
            with torch.no_grad():
                rollout_out = self.model(input_ids=padded_input_ids)
                old_token_logps, _ = compute_token_log_probs(
                    rollout_out.lm_logits, padded_targets, padded_mask
                )
                old_coord_logps = compute_coordinate_log_probs(
                    rollout_out.action_output.coordinates, padded_target_coords, sigma=0.05
                )
                old_action_logps = compute_action_type_log_probs(
                    rollout_out.action_output.action_type_logits,
                    action_type_targets,
                    action_mask,
                )

        # Compute group-normalized advantages: A_i = (r_i - mean) / (std + 1e-8)
        reward_mean = rewards_tensor.mean()
        if rewards_tensor.numel() > 1:
            reward_std = rewards_tensor.std(unbiased=False).clamp(min=1e-8)
        else:
            reward_std = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        advantages = (rewards_tensor - reward_mean) / reward_std

        return {
            "input_ids": padded_input_ids,
            "targets": padded_targets,
            "mask": padded_mask,
            "old_token_logps": old_token_logps,
            "old_coord_logps": old_coord_logps,
            "action_type_targets": action_type_targets,
            "action_mask": action_mask,
            "old_action_logps": old_action_logps,
            "target_coords": padded_target_coords,
            "rewards": rewards_tensor,
            "advantages": advantages,
        }

    def create_preference_pairs(
        self,
        trajectories: List[RolloutTrajectory],
    ) -> List[Tuple[RolloutTrajectory, RolloutTrajectory]]:
        """
        Pairs winning (higher reward / success=1.0) and losing (lower reward / success=0.0)
        trajectories for Direct Policy Optimization (DPO).

        Args:
            trajectories: List of completed RolloutTrajectory instances.

        Returns:
            List of (chosen_trajectory, rejected_trajectory) pairs.
        """
        # Separate winning vs losing
        winners = [t for t in trajectories if t.success]
        losers = [t for t in trajectories if not t.success]

        pairs: List[Tuple[RolloutTrajectory, RolloutTrajectory]] = []

        if winners and losers:
            # Pair each winner with a loser
            for i in range(max(len(winners), len(losers))):
                chosen = winners[i % len(winners)]
                rejected = losers[i % len(losers)]
                pairs.append((chosen, rejected))
        else:
            # Fallback: sort by cumulative reward and pair highest with lowest if margin exists
            sorted_trajs = sorted(trajectories, key=lambda t: t.cumulative_reward, reverse=True)
            if len(sorted_trajs) >= 2 and sorted_trajs[0].cumulative_reward > sorted_trajs[-1].cumulative_reward:
                pairs.append((sorted_trajs[0], sorted_trajs[-1]))

        return pairs

    def collate_preference_pairs(
        self,
        pairs: List[Tuple[RolloutTrajectory, RolloutTrajectory]],
    ) -> Dict[str, torch.Tensor]:
        """
        Collates pairs of (chosen, rejected) trajectories into batched tensors formatted for DPOTrainer.

        Args:
            pairs: List of (chosen_traj, rejected_traj) tuples.

        Returns:
            Dictionary containing:
            - chosen_input_ids, chosen_targets, chosen_mask
            - rejected_input_ids, rejected_targets, rejected_mask
        """
        if not pairs:
            raise ValueError("Cannot collate empty preference pair list.")

        chosen_trajs = [p[0] for p in pairs]
        rejected_trajs = [p[1] for p in pairs]

        chosen_batch = self.collate_rollouts(chosen_trajs)
        rejected_batch = self.collate_rollouts(rejected_trajs)

        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_targets": chosen_batch["targets"],
            "chosen_mask": chosen_batch["mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_targets": rejected_batch["targets"],
            "rejected_mask": rejected_batch["mask"],
        }
