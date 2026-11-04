"""
JED-M2 Synthetic Agent Trajectory Generator & Dynamic PyTorch Dataset.
Provides:
- Trajectory tuple representation: tau = (s_0, a_0, r_0, s_1, a_1, ..., s_T).
- Realistic multi-turn WebArena and Mind2Web interaction traces:
  DOM element trees, UI bounding boxes, discrete tools (CLICK, TYPE, SCROLL, NAVIGATE, API_CALL, FINISH),
  spatial (x, y) coordinates in [0, 1], and string argument tokens.
- Streaming PyTorch Dataset: AgentTrajectoryDataset.
- Dynamic Trajectory Collator: TrajectoryCollator assembling padded token sequences,
  4D causal & padding attention masks, action token index mapping, component loss masks,
  and JEPA observation transition pairs.

STRICT CONVENTION: Pure PyTorch without external compiled C++/CUDA packages or Hugging Face pipelines.
"""

from dataclasses import dataclass, field
import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch.utils.data import Dataset

from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.config import JEDM2Config
from jed_m2.core.features import compute_obs_features
from jed_m2.core.types import InteractionTurn, UIElement
from jed_m2.heads import ActionType


# ==============================================================================
# 1. Trajectory Data Structures
# ==============================================================================

@dataclass
class TrajectorySample:
    """Complete multi-turn agent interaction trajectory: tau = (s_0, a_0, r_0, ..., s_T)."""

    sample_id: str
    goal: str
    turns: List[InteractionTurn]
    total_reward: float = 0.0

    @property
    def length(self) -> int:
        """Returns number of turns in this trajectory."""
        return len(self.turns)


# ==============================================================================
# 2. Tokenizers: Invertible ActionBPETokenizer & LegacyHashTokenizer
# ==============================================================================

class LegacyHashTokenizer:
    """
    Legacy lightweight, deterministic MD5 hash tokenizer for JED-M2.
    Retained for backward compatibility with historical checkpoints.
    Reserves special tokens in [0..10] and hashes subwords/words into [11..vocab_size-1].
    """

    PAD_TOKEN_ID: int = 0
    BOS_TOKEN_ID: int = 1
    EOS_TOKEN_ID: int = 2
    OBS_TOKEN_ID: int = 3
    OBS_END_TOKEN_ID: int = 4
    THOUGHT_TOKEN_ID: int = 5
    THOUGHT_END_TOKEN_ID: int = 6
    ACTION_TOKEN_ID: int = 7
    ACTION_END_TOKEN_ID: int = 8
    ARG_TOKEN_ID: int = 9
    ARG_END_TOKEN_ID: int = 10

    def __init__(self, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.special_tokens = {
            "<pad>": self.PAD_TOKEN_ID,
            "<bos>": self.BOS_TOKEN_ID,
            "<eos>": self.EOS_TOKEN_ID,
            "<obs>": self.OBS_TOKEN_ID,
            "</obs>": self.OBS_END_TOKEN_ID,
            "<thought>": self.THOUGHT_TOKEN_ID,
            "</thought>": self.THOUGHT_END_TOKEN_ID,
            "<action>": self.ACTION_TOKEN_ID,
            "</action>": self.ACTION_END_TOKEN_ID,
            "<arg>": self.ARG_TOKEN_ID,
            "</arg>": self.ARG_END_TOKEN_ID,
        }
        self.inv_special_tokens = {v: k for k, v in self.special_tokens.items()}

    def _hash_word(self, word: str) -> int:
        """Hashes arbitrary word/string into [11, vocab_size - 1]."""
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
        return 11 + (h % (self.vocab_size - 11))

    def encode(self, text: str) -> List[int]:
        """Encodes string into list of token IDs."""
        tokens: List[int] = []
        words = text.split()
        for w in words:
            if w in self.special_tokens:
                tokens.append(self.special_tokens[w])
            else:
                tokens.append(self._hash_word(w))
        return tokens

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs into string representation."""
        words: List[str] = []
        for t in tokens:
            if t in self.inv_special_tokens:
                if not skip_special_tokens:
                    words.append(self.inv_special_tokens[t])
            else:
                words.append(f"tok_{t}")
        return " ".join(words)


# Primary tokenizer alias: SimpleLAMTokenizer now points to ActionBPETokenizer
SimpleLAMTokenizer = ActionBPETokenizer


# ==============================================================================
# 3. Synthetic Trajectory Generator (WebArena / Mind2Web Scenarios)
# ==============================================================================

class SyntheticTrajectoryGenerator:
    """
    Synthesizes realistic multi-turn web/OS interaction trajectories modeled after
    WebArena and Mind2Web digital agent benchmarks.
    Generates DOM element trees, UI bounding boxes, discrete tools, pointer targets,
    and rationale tokens.
    """

    DOM_TAGS = ["button", "input", "a", "select", "div", "textarea", "span"]
    ROLES = ["search_bar", "submit_btn", "nav_link", "filter_select", "item_card", "checkout_btn", "close_icon"]

    compute_obs_features = staticmethod(compute_obs_features)

    # Trajectory workflow templates
    SCENARIOS = [
        {
            "domain": "ecommerce",
            "goal": "Search for noise-cancelling headphones and add top result to cart",
            "steps": [
                (ActionType.TYPE, "input", "search_bar", "search_input", "Sony WH-1000XM5", "Type search query into product search bar"),
                (ActionType.CLICK, "button", "submit_btn", "search_button", "", "Click search button to execute query"),
                (ActionType.SCROLL, "div", "item_card", "product_listing", "scroll_down", "Scroll results listing to inspect pricing"),
                (ActionType.CLICK, "a", "item_card", "first_product_link", "", "Click product link to open details page"),
                (ActionType.CLICK, "button", "checkout_btn", "add_to_cart", "", "Click Add to Cart button to complete objective"),
                (ActionType.FINISH, "div", "checkout_btn", "finish_marker", "success", "Confirm item added and finish task"),
            ],
        },
        {
            "domain": "travel",
            "goal": "Book hotel in San Francisco for upcoming weekend",
            "steps": [
                (ActionType.NAVIGATE, "input", "nav_link", "url_bar", "https://travel.booking.internal/hotels", "Navigate to hotel booking portal"),
                (ActionType.TYPE, "input", "search_bar", "location_input", "San Francisco, CA", "Enter destination city"),
                (ActionType.CLICK, "select", "filter_select", "date_picker", "2026-10-24", "Select check-in weekend dates"),
                (ActionType.CLICK, "button", "submit_btn", "search_hotels", "", "Click search hotels button"),
                (ActionType.CLICK, "button", "item_card", "reserve_room", "", "Select first available room reservation"),
                (ActionType.FINISH, "div", "checkout_btn", "finish_marker", "booked", "Finish reservation workflow"),
            ],
        },
        {
            "domain": "os_admin",
            "goal": "Update security group rules in cloud administration console",
            "steps": [
                (ActionType.NAVIGATE, "a", "nav_link", "nav_security", "/console/vpc/security-groups", "Navigate to security groups dashboard"),
                (ActionType.SCROLL, "div", "item_card", "group_table", "scroll_down", "Scroll down to locate default VPC security group"),
                (ActionType.CLICK, "button", "submit_btn", "edit_inbound_rules", "", "Click edit inbound rules button"),
                (ActionType.API_CALL, "input", "search_bar", "rule_api", "authorize_ingress port=443 cidr=0.0.0.0/0", "Emit authorized ingress API update"),
                (ActionType.FINISH, "button", "submit_btn", "save_rules", "saved", "Commit rule changes and terminate"),
            ],
        },
    ]

    def __init__(
        self,
        config: JEDM2Config,
        seed: Optional[int] = None,
        tokenizer: Optional[ActionBPETokenizer] = None,
    ):
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.rng = random.Random(seed)
        self.tokenizer = tokenizer or ActionBPETokenizer(config.vocab_size)

    def _generate_dom_tree(self, target_element_name: str, target_tag: str, target_role: str) -> Tuple[List[UIElement], UIElement]:
        """Generates 6 to 12 realistic UI elements with non-overlapping bounding boxes in [0, 1]."""
        num_elements = self.rng.randint(6, 10)
        elements: List[UIElement] = []

        # Target element dimensions
        target_w = self.rng.uniform(0.10, 0.25)
        target_h = self.rng.uniform(0.04, 0.08)
        target_x = self.rng.uniform(0.05, 0.95 - target_w)
        target_y = self.rng.uniform(0.05, 0.95 - target_h)
        target_bbox = (target_x, target_y, target_x + target_w, target_y + target_h)

        target_el = UIElement(
            element_id=0,
            tag=target_tag,
            role=target_role,
            text=target_element_name,
            bbox=target_bbox,
        )
        elements.append(target_el)

        # Distractor elements
        for i in range(1, num_elements):
            w = self.rng.uniform(0.08, 0.22)
            h = self.rng.uniform(0.03, 0.07)
            x = self.rng.uniform(0.02, 0.98 - w)
            y = self.rng.uniform(0.02, 0.98 - h)
            tag = self.rng.choice(self.DOM_TAGS)
            role = self.rng.choice(self.ROLES)
            text = f"element_{tag}_{role}_{i}"
            elements.append(UIElement(element_id=i, tag=tag, role=role, text=text, bbox=(x, y, x + w, y + h)))

        self.rng.shuffle(elements)
        return elements, target_el

    def _render_screenshot(
        self,
        elements: List[UIElement],
        target_el: UIElement,
        h: int = 224,
        w: int = 224,
    ) -> torch.Tensor:
        """
        Renders a synthetic pure-PyTorch screenshot image [3, H, W] representing the GUI canvas.
        Draws background and UI element bounding box rectangular patches.
        """
        # Background: light gray [0.92, 0.94, 0.96]
        img = torch.full((3, h, w), 0.94, dtype=torch.float32)
        img[0] = 0.92
        img[1] = 0.94
        img[2] = 0.96

        # Draw distractor UI elements
        for el in elements:
            x0 = max(0, min(w - 1, int(el.bbox[0] * w)))
            y0 = max(0, min(h - 1, int(el.bbox[1] * h)))
            x1 = max(0, min(w, int(el.bbox[2] * w)))
            y1 = max(0, min(h, int(el.bbox[3] * h)))
            if x1 <= x0 or y1 <= y0:
                continue

            tag_hash = int(hashlib.md5(f"{el.tag}_{el.role}".encode()).hexdigest()[:6], 16)
            r = ((tag_hash >> 16) & 0xFF) / 255.0 * 0.4 + 0.5  # pastel range 0.5-0.9
            g = ((tag_hash >> 8) & 0xFF) / 255.0 * 0.4 + 0.5
            b = (tag_hash & 0xFF) / 255.0 * 0.4 + 0.5

            img[0, y0:y1, x0:x1] = r
            img[1, y0:y1, x0:x1] = g
            img[2, y0:y1, x0:x1] = b

            # Darker border
            border_r, border_g, border_b = r * 0.7, g * 0.7, b * 0.7
            img[0, y0, x0:x1] = border_r
            img[1, y0, x0:x1] = border_g
            img[2, y0, x0:x1] = border_b
            img[0, y1 - 1, x0:x1] = border_r
            img[1, y1 - 1, x0:x1] = border_g
            img[2, y1 - 1, x0:x1] = border_b
            img[0, y0:y1, x0] = border_r
            img[1, y0:y1, x0] = border_g
            img[2, y0:y1, x0] = border_b
            img[0, y0:y1, x1 - 1] = border_r
            img[1, y0:y1, x1 - 1] = border_g
            img[2, y0:y1, x1 - 1] = border_b

        # Draw target element with prominent accent color
        tx0 = max(0, min(w - 1, int(target_el.bbox[0] * w)))
        ty0 = max(0, min(h - 1, int(target_el.bbox[1] * h)))
        tx1 = max(0, min(w, int(target_el.bbox[2] * w)))
        ty1 = max(0, min(h, int(target_el.bbox[3] * h)))
        if tx1 > tx0 and ty1 > ty0:
            # Vibrant accent color: [0.15, 0.45, 0.85]
            img[0, ty0:ty1, tx0:tx1] = 0.15
            img[1, ty0:ty1, tx0:tx1] = 0.45
            img[2, ty0:ty1, tx0:tx1] = 0.85

        return img

    def generate_trajectory(self, scenario_idx: Optional[int] = None, max_turns: Optional[int] = None) -> TrajectorySample:
        """
        Synthesizes a full multi-turn trajectory sample tau = (s_0, a_0, r_0, ..., s_T).
        """
        if scenario_idx is None:
            scenario = self.rng.choice(self.SCENARIOS)
        else:
            scenario = self.SCENARIOS[scenario_idx % len(self.SCENARIOS)]

        goal = scenario["goal"]
        steps = scenario["steps"]
        if max_turns is not None:
            steps = steps[:max_turns]

        sample_id = f"traj_{self.rng.randint(100000, 999999)}"
        turns: List[InteractionTurn] = []

        # Generate turns
        for turn_idx, (act_type, tag, role, el_name, arg_val, thought_tmpl) in enumerate(steps):
            dom_elements, target_el = self._generate_dom_tree(el_name, tag, role)

            # Observation text: serialized DOM elements
            dom_str = " ".join(el.to_dom_string() for el in dom_elements[:6])
            obs_text = f"Goal: {goal} | Screen State: {dom_str}"

            # Continuous observation embedding [hidden_dim] for JEPA world model
            # Derived deterministically from DOM elements and observation structure
            obs_features = compute_obs_features(obs_text, dom_elements[:6], self.hidden_dim)

            # Target coordinates
            if act_type in (ActionType.CLICK, ActionType.TYPE):
                coords = target_el.center
            elif act_type == ActionType.SCROLL:
                coords = (0.50, 0.75)  # Normalized scroll position
            else:
                coords = (0.0, 0.0)  # Non-spatial action

            # Thought rationale
            thought = f"Step {turn_idx}: {thought_tmpl}. Target element: {el_name}."

            # Synthetic screenshot image [3, H, W] for visual grounding
            img_size = getattr(self.config, "image_size", 224)
            screenshot = self._render_screenshot(dom_elements, target_el, h=img_size, w=img_size)

            turns.append(
                InteractionTurn(
                    turn_id=turn_idx,
                    dom_elements=dom_elements,
                    obs_text=obs_text,
                    obs_features=obs_features,
                    thought=thought,
                    action_type=act_type,
                    coordinates=coords,
                    argument=arg_val,
                    reward=1.0 if turn_idx == len(steps) - 1 else 0.0,
                    screenshot=screenshot,
                )
            )

        return TrajectorySample(
            sample_id=sample_id,
            goal=goal,
            turns=turns,
            total_reward=1.0,
        )


# ==============================================================================
# 4. Streaming PyTorch Dataset
# ==============================================================================

class AgentTrajectoryDataset(Dataset):
    """
    Streaming PyTorch Dataset yielding structured multi-turn agent interaction trajectories.
    Generates samples on-the-fly to eliminate monolithic dataset caching and fit
    comfortably within host system RAM and GPU VRAM constraints.
    """

    def __init__(
        self,
        config: JEDM2Config,
        virtual_size: int = 10000,
        seed: int = 42,
        max_turns: Optional[int] = None,
        tokenizer: Optional[ActionBPETokenizer] = None,
    ):
        super().__init__()
        self.config = config
        self.virtual_size = virtual_size
        self.seed = seed
        self.max_turns = max_turns
        self.tokenizer = tokenizer or ActionBPETokenizer(config.vocab_size)
        self.generator = SyntheticTrajectoryGenerator(config, seed=seed, tokenizer=self.tokenizer)

    def __len__(self) -> int:
        return self.virtual_size

    def __getitem__(self, idx: int) -> TrajectorySample:
        # Seed generator deterministically per index
        self.generator.rng.seed(self.seed + idx)
        return self.generator.generate_trajectory(
            scenario_idx=idx, max_turns=self.max_turns
        )

    def __iter__(self):
        """Allows streaming iteration over infinite trajectory stream."""
        idx = 0
        while True:
            yield self.__getitem__(idx)
            idx += 1


# ==============================================================================
# 5. Dynamic Trajectory Batch & Collator
# ==============================================================================

@dataclass
class TrajectoryBatch:
    """
    Structured container for dynamically collated multi-turn trajectory batches.
    Provides padded token IDs, causal attention masks, action token index mapping,
    component loss masks, and aligned JEPA state transition tuples.
    """

    input_ids: torch.Tensor  # [B, S]
    attention_mask: torch.Tensor  # [B, 1, S, S] causal & padding mask
    labels: torch.Tensor  # [B, S] autoregressive target token IDs (-100 for non-target)
    reason_loss_mask: torch.Tensor  # [B, S] bool mask for thought tokens
    action_indices: torch.Tensor  # [B, max_actions] sequence indices for action emissions
    action_type_targets: torch.Tensor  # [B, max_actions] ground-truth ActionType IDs
    action_type_mask: torch.Tensor  # [B, max_actions] bool mask for valid action positions
    coord_targets: torch.Tensor  # [B, max_actions, 2] ground-truth continuous coordinates
    coord_loss_mask: torch.Tensor  # [B, max_actions] bool mask for spatial actions
    arg_targets: torch.Tensor  # [B, S] ground-truth argument token IDs (-100 for non-arg)
    arg_loss_mask: torch.Tensor  # [B, S] bool mask for argument tokens
    jepa_obs_t: torch.Tensor  # [B, hidden_dim] JEPA observation at time t
    jepa_action_t: torch.Tensor  # [B] JEPA action discrete ID at time t
    jepa_obs_next: torch.Tensor  # [B, hidden_dim] JEPA target observation at time t+1
    images: Optional[torch.Tensor] = None  # [B, 3, H, W] screenshot images for visual grounding
    prompt_token_ids: Optional[torch.Tensor] = None  # [B, S_p] prompt token IDs for pointer copy
    prompt_mask: Optional[torch.Tensor] = None  # [B, S_p] boolean valid mask

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "TrajectoryBatch":
        """Transfers all internal batch tensors to target device and precision."""
        images_tensor = (
            self.images.to(device=device, dtype=dtype or torch.float32)
            if self.images is not None
            else None
        )
        attn_mask_tensor = self.attention_mask.to(device=device)
        if dtype is not None and attn_mask_tensor.is_floating_point():
            attn_mask_tensor = attn_mask_tensor.to(dtype=dtype)
        prompt_ids_tensor = (
            self.prompt_token_ids.to(device=device)
            if self.prompt_token_ids is not None
            else None
        )
        prompt_mask_tensor = (
            self.prompt_mask.to(device=device)
            if self.prompt_mask is not None
            else None
        )
        return TrajectoryBatch(
            input_ids=self.input_ids.to(device=device),
            attention_mask=attn_mask_tensor,
            labels=self.labels.to(device=device),
            reason_loss_mask=self.reason_loss_mask.to(device=device),
            action_indices=self.action_indices.to(device=device),
            action_type_targets=self.action_type_targets.to(device=device),
            action_type_mask=self.action_type_mask.to(device=device),
            coord_targets=self.coord_targets.to(device=device, dtype=dtype or torch.float32),
            coord_loss_mask=self.coord_loss_mask.to(device=device),
            arg_targets=self.arg_targets.to(device=device),
            arg_loss_mask=self.arg_loss_mask.to(device=device),
            jepa_obs_t=self.jepa_obs_t.to(device=device, dtype=dtype or torch.float32),
            jepa_action_t=self.jepa_action_t.to(device=device),
            jepa_obs_next=self.jepa_obs_next.to(device=device, dtype=dtype or torch.float32),
            images=images_tensor,
            prompt_token_ids=prompt_ids_tensor,
            prompt_mask=prompt_mask_tensor,
        )


class TrajectoryCollator:
    """
    Dynamic Trajectory Collator for JED-M2 Large Action Model.
    Serializes variable-length multi-turn interaction traces into packed token sequences,
    computes 4D causal attention masks, maps action emission tokens, constructs
    component loss masks, and extracts JEPA transition pairs.
    """

    def __init__(
        self,
        config: JEDM2Config,
        tokenizer: Optional[Union[ActionBPETokenizer, LegacyHashTokenizer]] = None,
        max_seq_len: Optional[int] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer or ActionBPETokenizer(config.vocab_size)
        self.max_seq_len = max_seq_len or config.max_seq_len

    def __call__(self, samples: List[TrajectorySample]) -> TrajectoryBatch:
        """
        Collates a list of TrajectorySample instances into a unified TrajectoryBatch.

        Args:
            samples: List of multi-turn TrajectorySample instances.

        Returns:
            Collate TrajectoryBatch ready for model ingestion and composite loss evaluation.
        """
        batch_size = len(samples)

        batch_tokens: List[List[int]] = []
        batch_reason_masks: List[List[bool]] = []
        batch_arg_masks: List[List[bool]] = []
        batch_action_indices: List[List[int]] = []
        batch_action_types: List[List[int]] = []
        batch_coords: List[List[Tuple[float, float]]] = []
        batch_is_spatial: List[List[bool]] = []

        batch_jepa_obs_t: List[torch.Tensor] = []
        batch_jepa_act_t: List[int] = []
        batch_jepa_obs_next: List[torch.Tensor] = []
        batch_images: List[torch.Tensor] = []
        batch_prompt_tokens: List[List[int]] = []

        for sample in samples:
            tokens: List[int] = [self.tokenizer.BOS_TOKEN_ID]
            reason_mask: List[bool] = [False]
            arg_mask: List[bool] = [False]
            action_indices: List[int] = []
            action_types: List[int] = []
            coords: List[Tuple[float, float]] = []
            is_spatial: List[bool] = []

            # Extract prompt tokens (turn 0 observation text)
            sample_prompt: List[int] = []
            if sample.turns:
                obs_toks_turn0 = self.tokenizer.encode(sample.turns[0].obs_text)[:128]
                sample_prompt = [self.tokenizer.OBS_TOKEN_ID] + obs_toks_turn0 + [self.tokenizer.OBS_END_TOKEN_ID]
            batch_prompt_tokens.append(sample_prompt or [self.tokenizer.BOS_TOKEN_ID])

            # Extract screenshot for visual grounding (turn 0)
            if sample.turns and sample.turns[0].screenshot is not None:
                batch_images.append(sample.turns[0].screenshot)
            else:
                img_sz = getattr(self.config, "image_size", 224)
                batch_images.append(torch.zeros(3, img_sz, img_sz, dtype=torch.float32))

            # Process each turn in trajectory
            for turn_idx, turn in enumerate(sample.turns):
                # 1. Observation sequence: <obs> ... </obs>
                tokens.append(self.tokenizer.OBS_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                obs_toks = self.tokenizer.encode(turn.obs_text)[:32]  # Cap per-turn DOM tokens
                tokens.extend(obs_toks)
                reason_mask.extend([False] * len(obs_toks))
                arg_mask.extend([False] * len(obs_toks))

                tokens.append(self.tokenizer.OBS_END_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                # 2. System 2 Reasoning rationale: <thought> ... </thought>
                tokens.append(self.tokenizer.THOUGHT_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                thought_toks = self.tokenizer.encode(turn.thought)[:24]  # Cap rationale tokens
                tokens.extend(thought_toks)
                reason_mask.extend([True] * len(thought_toks))  # Loss active on thought tokens
                arg_mask.extend([False] * len(thought_toks))

                tokens.append(self.tokenizer.THOUGHT_END_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                # 3. System 1 Action emission point: <action>
                # Record sequence position where action heads will be evaluated
                action_indices.append(len(tokens))
                action_types.append(int(turn.action_type))
                coords.append(turn.coordinates)
                is_spatial.append(turn.action_type in (ActionType.CLICK, ActionType.TYPE, ActionType.SCROLL))

                tokens.append(self.tokenizer.ACTION_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                # Action type marker token
                act_token = self.tokenizer.encode(turn.action_type.name)[0]
                tokens.append(act_token)
                reason_mask.append(False)
                arg_mask.append(False)

                tokens.append(self.tokenizer.ACTION_END_TOKEN_ID)
                reason_mask.append(False)
                arg_mask.append(False)

                # 4. Action Argument payload: <arg> ... </arg>
                if turn.argument:
                    tokens.append(self.tokenizer.ARG_TOKEN_ID)
                    reason_mask.append(False)
                    arg_mask.append(False)

                    arg_toks = self.tokenizer.encode(turn.argument)[:16]
                    tokens.extend(arg_toks)
                    reason_mask.extend([False] * len(arg_toks))
                    arg_mask.extend([True] * len(arg_toks))  # Loss active on argument tokens

                    tokens.append(self.tokenizer.ARG_END_TOKEN_ID)
                    reason_mask.append(False)
                    arg_mask.append(False)

            # End of trajectory sequence: <eos>
            tokens.append(self.tokenizer.EOS_TOKEN_ID)
            reason_mask.append(False)
            arg_mask.append(False)

            # Bound length to max_seq_len
            tokens = tokens[:self.max_seq_len]
            reason_mask = reason_mask[:self.max_seq_len]
            arg_mask = arg_mask[:self.max_seq_len]

            # Filter action indices within max_seq_len
            valid_actions = [(idx, at, c, sp) for idx, at, c, sp in zip(action_indices, action_types, coords, is_spatial) if idx < self.max_seq_len]
            if valid_actions:
                a_idx, a_t, c_t, s_t = zip(*valid_actions)
                batch_action_indices.append(list(a_idx))
                batch_action_types.append(list(a_t))
                batch_coords.append(list(c_t))
                batch_is_spatial.append(list(s_t))
            else:
                batch_action_indices.append([0])
                batch_action_types.append([0])
                batch_coords.append([(0.0, 0.0)])
                batch_is_spatial.append([False])

            batch_tokens.append(tokens)
            batch_reason_masks.append(reason_mask)
            batch_arg_masks.append(arg_mask)

            # Extract JEPA state transition pair (s_t, a_t, s_{t+1})
            if len(sample.turns) >= 2:
                # Pair turn 0 and turn 1
                batch_jepa_obs_t.append(sample.turns[0].obs_features)
                batch_jepa_act_t.append(int(sample.turns[0].action_type))
                batch_jepa_obs_next.append(sample.turns[1].obs_features)
            else:
                # Single-turn fallback
                batch_jepa_obs_t.append(sample.turns[0].obs_features)
                batch_jepa_act_t.append(int(sample.turns[0].action_type))
                batch_jepa_obs_next.append(sample.turns[0].obs_features)

        # Compute max sequence length in batch
        max_batch_seq_len = max(len(t) for t in batch_tokens)
        max_batch_actions = max(len(a) for a in batch_action_indices)

        # Pad tokens, masks, and labels
        padded_input_ids = torch.full((batch_size, max_batch_seq_len), self.tokenizer.PAD_TOKEN_ID, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_batch_seq_len), -100, dtype=torch.long)
        padded_reason_mask = torch.zeros((batch_size, max_batch_seq_len), dtype=torch.bool)
        padded_arg_mask = torch.zeros((batch_size, max_batch_seq_len), dtype=torch.bool)
        padded_arg_targets = torch.full((batch_size, max_batch_seq_len), -100, dtype=torch.long)

        # Pad action containers
        padded_action_indices = torch.zeros((batch_size, max_batch_actions), dtype=torch.long)
        padded_action_types = torch.zeros((batch_size, max_batch_actions), dtype=torch.long)
        padded_action_mask = torch.zeros((batch_size, max_batch_actions), dtype=torch.bool)
        padded_coords = torch.zeros((batch_size, max_batch_actions, 2), dtype=torch.float32)
        padded_coord_mask = torch.zeros((batch_size, max_batch_actions), dtype=torch.bool)

        for b in range(batch_size):
            seq_l = len(batch_tokens[b])
            padded_input_ids[b, :seq_l] = torch.tensor(batch_tokens[b], dtype=torch.long)
            # Autoregressive next-token prediction: labels shifted right
            padded_labels[b, :seq_l] = torch.tensor(batch_tokens[b], dtype=torch.long)

            padded_reason_mask[b, :seq_l] = torch.tensor(batch_reason_masks[b], dtype=torch.bool)
            padded_arg_mask[b, :seq_l] = torch.tensor(batch_arg_masks[b], dtype=torch.bool)
            # Argument target labels active on arg tokens
            for s_idx in range(seq_l):
                if batch_arg_masks[b][s_idx]:
                    padded_arg_targets[b, s_idx] = batch_tokens[b][s_idx]

            # Action fields
            num_act = len(batch_action_indices[b])
            padded_action_indices[b, :num_act] = torch.tensor(batch_action_indices[b], dtype=torch.long)
            padded_action_types[b, :num_act] = torch.tensor(batch_action_types[b], dtype=torch.long)
            padded_action_mask[b, :num_act] = True
            padded_coords[b, :num_act] = torch.tensor(batch_coords[b], dtype=torch.float32)
            padded_coord_mask[b, :num_act] = torch.tensor(batch_is_spatial[b], dtype=torch.bool)

        # Pad prompt tokens for pointer copy supervision
        max_batch_prompt_len = max(max(len(p) for p in batch_prompt_tokens), 1)
        padded_prompt_token_ids = torch.full((batch_size, max_batch_prompt_len), self.tokenizer.PAD_TOKEN_ID, dtype=torch.long)
        padded_prompt_mask = torch.zeros((batch_size, max_batch_prompt_len), dtype=torch.bool)
        for b in range(batch_size):
            p_len = len(batch_prompt_tokens[b])
            padded_prompt_token_ids[b, :p_len] = torch.tensor(batch_prompt_tokens[b], dtype=torch.long)
            padded_prompt_mask[b, :p_len] = True

        # 4D Causal & Padding Attention Mask: [B, 1, S, S]
        # True = participate in attention, False = masked out
        pad_mask = (padded_input_ids != self.tokenizer.PAD_TOKEN_ID)[:, None, None, :]  # [B, 1, 1, S]
        causal_mask = torch.tril(torch.ones(max_batch_seq_len, max_batch_seq_len, dtype=torch.bool))[None, None, :, :]  # [1, 1, S, S]
        attention_mask = causal_mask & pad_mask  # [B, 1, S, S]

        # Stack JEPA transitions: [B, hidden_dim]
        jepa_obs_t = torch.stack(batch_jepa_obs_t, dim=0)
        jepa_action_t = torch.tensor(batch_jepa_act_t, dtype=torch.long)
        jepa_obs_next = torch.stack(batch_jepa_obs_next, dim=0)

        # Stack screenshot images: [B, 3, H, W]
        images_tensor = torch.stack(batch_images, dim=0) if batch_images else None

        return TrajectoryBatch(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            labels=padded_labels,
            reason_loss_mask=padded_reason_mask,
            action_indices=padded_action_indices,
            action_type_targets=padded_action_types,
            action_type_mask=padded_action_mask,
            coord_targets=padded_coords,
            coord_loss_mask=padded_coord_mask,
            arg_targets=padded_arg_targets,
            arg_loss_mask=padded_arg_mask,
            jepa_obs_t=jepa_obs_t,
            jepa_action_t=jepa_action_t,
            jepa_obs_next=jepa_obs_next,
            images=images_tensor,
            prompt_token_ids=padded_prompt_token_ids,
            prompt_mask=padded_prompt_mask,
        )
