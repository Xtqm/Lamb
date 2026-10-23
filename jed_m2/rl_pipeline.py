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
from jed_m2.rl import (
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

    def sample_trajectory(
        self,
        env: WebOSEnvironment,
        temperature: float = 0.7,
        max_turns: int = 8,
    ) -> RolloutTrajectory:
        """
        Executes a single interactive episode in WebOSEnvironment with temperature-scaled exploration.

        Args:
            env: WebOSEnvironment instance.
            temperature: Sampling temperature for System 2 rationale tokens and System 1 action primitives.
            max_turns: Maximum allowed interaction turns before truncation.

        Returns:
            RolloutTrajectory with token sequence, loss masks, and reward signals.
        """
        self.model.eval()

        # Reset environment
        init_res = env.reset()
        obs_text = init_res.obs_text

        # Recurrent state buffers for O(1) step inference
        recurrent_states = None
        current_pos = 0

        # Accumulated trajectory tokens and loss masks
        all_tokens: List[int] = [self.tokenizer.BOS_TOKEN_ID]
        all_mask: List[bool] = [False]  # BOS is not trainable
        turns: List[RolloutTurn] = []
        step_rewards: List[float] = []

        end_thought_id = self.tokenizer.special_tokens["</thought>"]

        for turn_idx in range(max_turns):
            turn_tokens: List[int] = []
            turn_mask: List[bool] = []

            # ------------------------------------------------------------------
            # 1. Observation Prompt Ingestion
            # ------------------------------------------------------------------
            obs_prompt = f"<obs> {obs_text} </obs> <thought>"
            obs_tokens = self.tokenizer.encode(obs_prompt)[:24]

            all_tokens.extend(obs_tokens)
            all_mask.extend([False] * len(obs_tokens))
            turn_tokens.extend(obs_tokens)
            turn_mask.extend([False] * len(obs_tokens))

            # Step model recurrently through observation tokens
            last_lm_logits = None
            last_action_out = None
            for tok in obs_tokens:
                # [1] -> token tensor
                token_tensor = torch.tensor([tok], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    last_lm_logits, last_action_out, recurrent_states = self.model.step(
                        token_tensor,
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

            # Decode thought string
            thought_text = self.tokenizer.decode(thought_tokens).strip()

            # Dispatch action to environment
            step_res = env.step(
                action_type=action_type,
                coordinates=coords,
                argument=arg_str,
            )
            step_rewards.append(step_res.reward)

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
                done=step_res.done,
                gate_prob=gate_prob,
                coord_logp=coord_logp,
            )
            turns.append(record)

            obs_text = step_res.obs_text
            if step_res.done:
                break

        # Append terminal EOS token
        all_tokens.append(self.tokenizer.EOS_TOKEN_ID)
        all_mask.append(False)

        terminal_reward = 1.0 if env.goal_satisfied else 0.0
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
            success=env.goal_satisfied,
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
                    traj = self.sample_trajectory(active_env, temperature=temperature, max_turns=max_turns)
                    group.append(traj)
            finally:
                if owns_browser:
                    active_env.close()

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

        # Determine sequence lengths (input length = full_tokens - 1)
        seq_lens = [max(1, len(t.full_token_ids) - 1) for t in trajectories]
        max_seq_len = min(max(seq_lens), self.config.max_seq_len)

        padded_input_ids = torch.full(
            (batch_size, max_seq_len), self.tokenizer.PAD_TOKEN_ID, dtype=torch.long, device=self.device
        )
        padded_targets = torch.full(
            (batch_size, max_seq_len), -100, dtype=torch.long, device=self.device
        )
        padded_mask = torch.zeros(
            (batch_size, max_seq_len), dtype=torch.bool, device=self.device
        )
        padded_target_coords = torch.zeros(
            (batch_size, max_seq_len, 2), dtype=torch.float32, device=self.device
        )

        raw_rewards: List[float] = []

        for b, traj in enumerate(trajectories):
            tokens = traj.full_token_ids
            mask = traj.trainable_mask

            # input_ids: tokens[:-1], targets: tokens[1:], mask: mask[1:]
            s_len = min(len(tokens) - 1, max_seq_len)
            if s_len > 0:
                padded_input_ids[b, :s_len] = torch.tensor(
                    tokens[:s_len], dtype=torch.long, device=self.device
                )
                padded_targets[b, :s_len] = torch.tensor(
                    tokens[1 : s_len + 1], dtype=torch.long, device=self.device
                )
                padded_mask[b, :s_len] = torch.tensor(
                    mask[1 : s_len + 1], dtype=torch.bool, device=self.device
                )

            # Assign coordinate targets from turns
            turn_pos = 0
            for turn in traj.turns:
                t_len = len(turn.step_tokens)
                if turn_pos < max_seq_len:
                    padded_target_coords[b, turn_pos : min(max_seq_len, turn_pos + t_len)] = torch.tensor(
                        turn.coordinates, dtype=torch.float32, device=self.device
                    )
                turn_pos += t_len

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
                shaped_r = compute_shaped_reward(
                    goal_satisfied=traj.success,
                    milestones_count=milestone_count,
                    turn_count=len(traj.turns),
                    iou_sum=0.5 if traj.success else 0.0,
                    repeated_action_count=repeated_count,
                )
                raw_rewards.append(shaped_r)
            elif reward_mode == "terminal":
                raw_rewards.append(traj.terminal_reward)
            else:
                raw_rewards.append(traj.cumulative_reward)

        rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float32, device=self.device)

        # Compute old logps under rollout policy using forward pass in torch.no_grad()
        self.model.eval()
        with torch.no_grad():
            rollout_out = self.model(input_ids=padded_input_ids)
            old_token_logps, _ = compute_token_log_probs(
                rollout_out.lm_logits, padded_targets, padded_mask
            )
            old_coord_logps = compute_coordinate_log_probs(
                rollout_out.action_output.coordinates, padded_target_coords, sigma=0.05
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
