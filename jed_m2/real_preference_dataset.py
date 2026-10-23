"""
JED-M2 Real-World Web Preference Dataset & DPO Collator (Track A).
Constructs paired preference trajectories (tau_w, tau_l) directly from Mind2Web offline traces:
- Winning Trajectory (tau_w): Authentic human expert demonstration from data/raw_mind2web/ or cached real traces.
- Losing Trajectory (tau_l): Sub-optimal action primitive substitutions or ground-truth actions with
  injected spatial coordinate perturbations (|Delta x| > 0.15, |Delta y| > 0.15).
- Dynamic PyTorch Collator (RealPreferenceCollator): Batches paired sequences, attention masks,
  and action loss masks formatted for DPOTrainer.compute_loss().

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

from dataclasses import dataclass, field
import hashlib
import logging
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from jed_m2.config import JEDM2Config
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.heads import ActionType
from jed_m2.real_dataset import (
    InteractiveElement,
    RealWebInteractionTurn,
    RealWebTrajectoryDataset,
    RealWebTrajectorySample,
)

logger = logging.getLogger("jed_m2.real_preference_dataset")


# ==============================================================================
# 1. Preference Trajectory & Pair Containers
# ==============================================================================

@dataclass
class PreferenceTurn:
    """Represents a single turn step in a preference trajectory."""

    turn_id: int
    obs_text: str
    thought: str
    action_type: ActionType
    coordinates: Tuple[float, float]
    argument: str
    is_perturbed: bool = False


@dataclass
class PreferenceTrajectory:
    """Preference trajectory sequence containing multi-turn actions and tokens."""

    sample_id: str
    goal: str
    domain: str
    turns: List[PreferenceTurn]
    is_winning: bool = True
    perturbation_type: str = "none"  # "none", "spatial", "action_substitution"


@dataclass
class RealPreferencePair:
    """Paired winning (tau_w) and losing (tau_l) trajectories for Direct Policy Optimization."""

    sample_id: str
    goal: str
    domain: str
    chosen: PreferenceTrajectory
    rejected: PreferenceTrajectory
    chosen_tokens: List[int]
    chosen_mask: List[bool]
    chosen_target_coords: List[Tuple[float, float]]
    rejected_tokens: List[int]
    rejected_mask: List[bool]
    rejected_target_coords: List[Tuple[float, float]]
    perturbation_type: str
    perturbation_delta: Tuple[float, float]


# ==============================================================================
# 2. Perturbation Engine
# ==============================================================================

def inject_spatial_coordinate_perturbation(
    coords: Tuple[float, float],
    min_displacement: float = 0.15,
    max_displacement: float = 0.35,
    rng: Optional[random.Random] = None,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Injects spatial coordinate displacement (|Delta x| > 0.15, |Delta y| > 0.15) to force a miss
    outside target interactive element bounding boxes while clamping within [0, 1]^2.

    Returns:
        Tuple of ((perturbed_x, perturbed_y), (delta_x, delta_y)).
    """
    r = rng if rng is not None else random.Random()
    cx, cy = coords

    # Choose displacement direction pointing away from nearest edge or towards open viewport area
    dir_x = -1.0 if cx >= 0.5 else 1.0
    dir_y = -1.0 if cy >= 0.5 else 1.0

    disp_x = r.uniform(min_displacement, max_displacement)
    disp_y = r.uniform(min_displacement, max_displacement)

    # Compute raw perturbed coordinates
    pert_x = cx + dir_x * disp_x
    pert_y = cy + dir_y * disp_y

    # Clamp to [0.0, 1.0] while ensuring displacement magnitude remains >= min_displacement
    pert_x = max(0.0, min(1.0, pert_x))
    pert_y = max(0.0, min(1.0, pert_y))

    actual_dx = pert_x - cx
    actual_dy = pert_y - cy

    # If clamping compressed the delta below threshold, flip direction
    if abs(actual_dx) < min_displacement:
        pert_x = max(0.0, min(1.0, cx - dir_x * disp_x))
        actual_dx = pert_x - cx

    if abs(actual_dy) < min_displacement:
        pert_y = max(0.0, min(1.0, cy - dir_y * disp_y))
        actual_dy = pert_y - cy

    return (float(pert_x), float(pert_y)), (float(actual_dx), float(actual_dy))


def inject_action_primitive_substitution(
    action_type: ActionType,
    rng: Optional[random.Random] = None,
) -> ActionType:
    """
    Substitutes an optimal action primitive with a sub-optimal alternative (e.g., spurious SCROLL or misplaced CLICK).
    """
    r = rng if rng is not None else random.Random()
    if action_type == ActionType.CLICK:
        # Inadvertently scroll instead of clicking target
        return ActionType.SCROLL if r.random() < 0.6 else ActionType.NAVIGATE
    elif action_type == ActionType.TYPE:
        # Inadvertently click or scroll instead of typing input
        return ActionType.SCROLL if r.random() < 0.5 else ActionType.CLICK
    elif action_type == ActionType.FINISH:
        # Premature or redundant scroll instead of finishing task
        return ActionType.SCROLL
    elif action_type == ActionType.NAVIGATE:
        return ActionType.CLICK
    else:
        return ActionType.SCROLL


# ==============================================================================
# 3. Trajectory Tokenizer Helper
# ==============================================================================

def tokenize_preference_trajectory(
    traj: PreferenceTrajectory,
    tokenizer: SimpleLAMTokenizer,
    max_seq_len: int = 1024,
) -> Tuple[List[int], List[bool], List[Tuple[float, float]]]:
    """
    Serializes a PreferenceTrajectory into token IDs, trainable action/thought masks,
    and target continuous coordinate points.

    Returns:
        Tuple of (token_ids, trainable_mask, target_coords).
    """
    tokens: List[int] = [tokenizer.BOS_TOKEN_ID]
    mask: List[bool] = [False]
    coords: List[Tuple[float, float]] = [(0.5, 0.5)]

    # Encode task goal prompt (non-trainable)
    goal_str = f"Goal: {traj.goal}"
    goal_toks = tokenizer.encode(goal_str)
    tokens.extend(goal_toks)
    mask.extend([False] * len(goal_toks))
    coords.extend([(0.5, 0.5)] * len(goal_toks))

    for turn in traj.turns:
        # 1. Observation context (non-trainable)
        obs_str = f"<obs> {turn.obs_text} </obs>"
        obs_toks = tokenizer.encode(obs_str)[:32]
        tokens.extend(obs_toks)
        mask.extend([False] * len(obs_toks))
        coords.extend([(0.5, 0.5)] * len(obs_toks))

        # 2. System 2 Reasoning rationale (trainable tokens between tags)
        tokens.append(tokenizer.special_tokens["<thought>"])
        mask.append(False)
        coords.append((0.5, 0.5))

        thought_words = turn.thought.split()
        if thought_words:
            thought_str = " ".join(thought_words[:24])
            thought_toks = tokenizer.encode(thought_str)
            tokens.extend(thought_toks)
            mask.extend([True] * len(thought_toks))
            coords.extend([(0.5, 0.5)] * len(thought_toks))

        tokens.append(tokenizer.special_tokens["</thought>"])
        mask.append(False)
        coords.append((0.5, 0.5))

        # 3. System 1 Action Primitive (trainable token)
        tokens.append(tokenizer.special_tokens["<action>"])
        mask.append(False)
        coords.append((0.5, 0.5))

        act_toks = tokenizer.encode(turn.action_type.name)
        act_tok = act_toks[0] if act_toks else tokenizer.special_tokens["<action>"]
        tokens.append(act_tok)
        mask.append(True)
        coords.append(turn.coordinates)

        tokens.append(tokenizer.special_tokens["</action>"])
        mask.append(False)
        coords.append((0.5, 0.5))

        # 4. Optional Argument (trainable tokens)
        if turn.argument:
            tokens.append(tokenizer.special_tokens["<arg>"])
            mask.append(False)
            coords.append((0.5, 0.5))

            arg_toks = tokenizer.encode(turn.argument)[:8]
            tokens.extend(arg_toks)
            mask.extend([True] * len(arg_toks))
            coords.extend([turn.coordinates] * len(arg_toks))

            tokens.append(tokenizer.special_tokens["</arg>"])
            mask.append(False)
            coords.append((0.5, 0.5))

        if len(tokens) >= max_seq_len - 2:
            break

    tokens.append(tokenizer.EOS_TOKEN_ID)
    mask.append(False)
    coords.append((0.5, 0.5))

    return tokens[:max_seq_len], mask[:max_seq_len], coords[:max_seq_len]


# ==============================================================================
# 4. Real Web Preference Dataset
# ==============================================================================

class RealWebPreferenceDataset(Dataset):
    """
    Offline dataset generating paired preference trajectories (tau_w, tau_l) from authentic
    Mind2Web real-world web interaction traces for Direct Policy Optimization (DPO).

    Each sample contains:
    - Winning trajectory (tau_w): Authentic expert human demonstration.
    - Losing trajectory (tau_l): Sub-optimal action substitution or spatial perturbation (|Delta x| > 0.15, |Delta y| > 0.15).
    """

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        split: str = "train",
        dataset_size: int = 500,
        cache_dir: str = "data/real_preference_cache",
        raw_mind2web_dir: Optional[str] = "data/raw_mind2web",
        spatial_perturb_prob: float = 0.70,
        force_rebuild: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.config = config if config is not None else JEDM2Config()
        self.split = split
        self.dataset_size = dataset_size
        self.cache_dir = Path(cache_dir)
        self.raw_mind2web_dir = Path(raw_mind2web_dir) if raw_mind2web_dir else None
        self.spatial_perturb_prob = spatial_perturb_prob
        self.seed = seed
        self.tokenizer = SimpleLAMTokenizer(self.config.vocab_size)

        self.pairs: List[RealPreferencePair] = []
        self._init_dataset(force_rebuild=force_rebuild)

    def _get_cache_path(self) -> Path:
        os.makedirs(self.cache_dir, exist_ok=True)
        prefix = "mind2web" if (self.raw_mind2web_dir and self.raw_mind2web_dir.exists()) else "mock"
        return self.cache_dir / f"pref_{prefix}_{self.split}_{self.dataset_size}_seed{self.seed}.pt"

    def _init_dataset(self, force_rebuild: bool = False) -> None:
        cache_path = self._get_cache_path()
        if cache_path.exists() and not force_rebuild:
            try:
                cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)
                if isinstance(cached_data, list) and len(cached_data) == self.dataset_size:
                    self.pairs = cached_data
                    logger.info("[OK] Loaded %d preference pairs from cache: %s", len(self.pairs), cache_path)
                    return
            except Exception as exc:
                logger.warning("Cache load failed: %s. Rebuilding preference dataset...", exc)

        logger.info("Building RealWebPreferenceDataset (%s split, size=%d)...", self.split, self.dataset_size)
        self.pairs = self._build_pairs()

        try:
            torch.save(self.pairs, cache_path)
            logger.info("[OK] Persisted %d preference pairs to disk cache: %s", len(self.pairs), cache_path)
        except Exception as exc:
            logger.warning("Failed to save preference cache: %s", exc)

    def _build_pairs(self) -> List[RealPreferencePair]:
        # Ingest underlying real web trajectories
        base_dataset = RealWebTrajectoryDataset(
            config=self.config,
            split=self.split,
            dataset_size=self.dataset_size,
            raw_mind2web_dir=str(self.raw_mind2web_dir) if self.raw_mind2web_dir else None,
            force_rebuild=False,
            seed=self.seed,
        )

        pairs: List[RealPreferencePair] = []
        rng = random.Random(self.seed)

        for idx in range(len(base_dataset)):
            sample: RealWebTrajectorySample = base_dataset[idx]
            pair = self._create_preference_pair(sample, pair_idx=idx, rng=rng)
            pairs.append(pair)

        return pairs

    def _create_preference_pair(
        self,
        sample: RealWebTrajectorySample,
        pair_idx: int,
        rng: random.Random,
    ) -> RealPreferencePair:
        """Constructs paired (chosen, rejected) preference trajectories from a demonstration."""
        sample_id = f"pref_{sample.sample_id}"
        domain = sample.domain
        goal = sample.goal

        # 1. Build Winning Trajectory (tau_w) from expert human demonstration
        chosen_turns: List[PreferenceTurn] = []
        for t in sample.turns:
            chosen_turns.append(
                PreferenceTurn(
                    turn_id=t.turn_id,
                    obs_text=t.obs_text,
                    thought=t.thought,
                    action_type=t.action_type,
                    coordinates=t.coordinates,
                    argument=t.argument,
                    is_perturbed=False,
                )
            )

        chosen_traj = PreferenceTrajectory(
            sample_id=f"{sample_id}_w",
            goal=goal,
            domain=domain,
            turns=chosen_turns,
            is_winning=True,
            perturbation_type="none",
        )

        # 2. Build Losing Trajectory (tau_l) via Targeted Perturbation
        rejected_turns: List[PreferenceTurn] = []
        pert_type = "spatial" if rng.random() < self.spatial_perturb_prob else "action_substitution"
        pert_delta: Tuple[float, float] = (0.0, 0.0)

        # Select turn to perturb (prefer an interactive CLICK/TYPE turn if present)
        target_turn_idx = len(sample.turns) - 1
        for i_t, t in enumerate(sample.turns):
            if t.action_type in (ActionType.CLICK, ActionType.TYPE):
                target_turn_idx = i_t
                break

        for i_t, t in enumerate(sample.turns):
            if i_t == target_turn_idx:
                if pert_type == "spatial":
                    pert_coords, delta = inject_spatial_coordinate_perturbation(
                        coords=t.coordinates,
                        min_displacement=0.15,
                        max_displacement=0.35,
                        rng=rng,
                    )
                    pert_delta = delta
                    pert_thought = f"{t.thought} [perturbed coordinates: misclick outside element]"
                    rejected_turns.append(
                        PreferenceTurn(
                            turn_id=t.turn_id,
                            obs_text=t.obs_text,
                            thought=pert_thought,
                            action_type=t.action_type,
                            coordinates=pert_coords,
                            argument=t.argument,
                            is_perturbed=True,
                        )
                    )
                else:
                    sub_act = inject_action_primitive_substitution(t.action_type, rng=rng)
                    pert_thought = f"{t.thought} [sub-optimal action primitive]"
                    rejected_turns.append(
                        PreferenceTurn(
                            turn_id=t.turn_id,
                            obs_text=t.obs_text,
                            thought=pert_thought,
                            action_type=sub_act,
                            coordinates=t.coordinates,
                            argument=t.argument if sub_act != ActionType.SCROLL else "",
                            is_perturbed=True,
                        )
                    )
            else:
                rejected_turns.append(
                    PreferenceTurn(
                        turn_id=t.turn_id,
                        obs_text=t.obs_text,
                        thought=t.thought,
                        action_type=t.action_type,
                        coordinates=t.coordinates,
                        argument=t.argument,
                        is_perturbed=False,
                    )
                )

        rejected_traj = PreferenceTrajectory(
            sample_id=f"{sample_id}_l",
            goal=goal,
            domain=domain,
            turns=rejected_turns,
            is_winning=False,
            perturbation_type=pert_type,
        )

        # 3. Tokenize both trajectories
        ch_tokens, ch_mask, ch_coords = tokenize_preference_trajectory(
            chosen_traj, self.tokenizer, max_seq_len=self.config.max_seq_len
        )
        rej_tokens, rej_mask, rej_coords = tokenize_preference_trajectory(
            rejected_traj, self.tokenizer, max_seq_len=self.config.max_seq_len
        )

        return RealPreferencePair(
            sample_id=sample_id,
            goal=goal,
            domain=domain,
            chosen=chosen_traj,
            rejected=rejected_traj,
            chosen_tokens=ch_tokens,
            chosen_mask=ch_mask,
            chosen_target_coords=ch_coords,
            rejected_tokens=rej_tokens,
            rejected_mask=rej_mask,
            rejected_target_coords=rej_coords,
            perturbation_type=pert_type,
            perturbation_delta=pert_delta,
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> RealPreferencePair:
        return self.pairs[idx]


# ==============================================================================
# 5. Real Preference Collator
# ==============================================================================

class RealPreferenceCollator:
    """
    Dynamic PyTorch Collator assembling batches of RealPreferencePair items into
    padded tensor sequences formatted for DPOTrainer.compute_loss():
    - chosen_input_ids, chosen_targets, chosen_mask
    - rejected_input_ids, rejected_targets, rejected_mask
    """

    def __init__(
        self,
        config: Optional[JEDM2Config] = None,
        tokenizer: Optional[SimpleLAMTokenizer] = None,
        device: Union[str, torch.device] = "cpu",
    ):
        self.config = config if config is not None else JEDM2Config()
        self.tokenizer = tokenizer if tokenizer is not None else SimpleLAMTokenizer(self.config.vocab_size)
        self.device = torch.device(device)
        self.pad_id = self.tokenizer.PAD_TOKEN_ID

    def __call__(self, batch: List[RealPreferencePair]) -> Dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("Cannot collate empty preference batch.")

        batch_size = len(batch)

        # Lengths (input length = tokens - 1)
        chosen_lens = [max(1, len(p.chosen_tokens) - 1) for p in batch]
        rejected_lens = [max(1, len(p.rejected_tokens) - 1) for p in batch]

        max_chosen_len = min(max(chosen_lens), self.config.max_seq_len)
        max_rejected_len = min(max(rejected_lens), self.config.max_seq_len)

        # Allocate padded tensors
        chosen_input_ids = torch.full((batch_size, max_chosen_len), self.pad_id, dtype=torch.long, device=self.device)
        chosen_targets = torch.full((batch_size, max_chosen_len), -100, dtype=torch.long, device=self.device)
        chosen_mask = torch.zeros((batch_size, max_chosen_len), dtype=torch.bool, device=self.device)
        chosen_coords = torch.zeros((batch_size, max_chosen_len, 2), dtype=torch.float32, device=self.device)

        rejected_input_ids = torch.full((batch_size, max_rejected_len), self.pad_id, dtype=torch.long, device=self.device)
        rejected_targets = torch.full((batch_size, max_rejected_len), -100, dtype=torch.long, device=self.device)
        rejected_mask = torch.zeros((batch_size, max_rejected_len), dtype=torch.bool, device=self.device)
        rejected_coords = torch.zeros((batch_size, max_rejected_len, 2), dtype=torch.float32, device=self.device)

        for b, pair in enumerate(batch):
            # 1. Chosen trajectory shift alignment
            c_toks = pair.chosen_tokens
            c_msk = pair.chosen_mask
            c_cds = pair.chosen_target_coords
            s_c = min(len(c_toks) - 1, max_chosen_len)

            if s_c > 0:
                chosen_input_ids[b, :s_c] = torch.tensor(c_toks[:s_c], dtype=torch.long, device=self.device)
                chosen_targets[b, :s_c] = torch.tensor(c_toks[1 : s_c + 1], dtype=torch.long, device=self.device)
                chosen_mask[b, :s_c] = torch.tensor(c_msk[1 : s_c + 1], dtype=torch.bool, device=self.device)
                coords_t = torch.tensor(c_cds[1 : s_c + 1], dtype=torch.float32, device=self.device)
                chosen_coords[b, :s_c] = coords_t

            # Guard: ensure at least one active token in mask
            if not chosen_mask[b, :s_c].any() and s_c > 0:
                chosen_mask[b, s_c - 1] = True

            # 2. Rejected trajectory shift alignment
            r_toks = pair.rejected_tokens
            r_msk = pair.rejected_mask
            r_cds = pair.rejected_target_coords
            s_r = min(len(r_toks) - 1, max_rejected_len)

            if s_r > 0:
                rejected_input_ids[b, :s_r] = torch.tensor(r_toks[:s_r], dtype=torch.long, device=self.device)
                rejected_targets[b, :s_r] = torch.tensor(r_toks[1 : s_r + 1], dtype=torch.long, device=self.device)
                rejected_mask[b, :s_r] = torch.tensor(r_msk[1 : s_r + 1], dtype=torch.bool, device=self.device)
                coords_r = torch.tensor(r_cds[1 : s_r + 1], dtype=torch.float32, device=self.device)
                rejected_coords[b, :s_r] = coords_r

            if not rejected_mask[b, :s_r].any() and s_r > 0:
                rejected_mask[b, s_r - 1] = True

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_targets": chosen_targets,
            "chosen_mask": chosen_mask,
            "chosen_coords": chosen_coords,
            "rejected_input_ids": rejected_input_ids,
            "rejected_targets": rejected_targets,
            "rejected_mask": rejected_mask,
            "rejected_coords": rejected_coords,
        }
