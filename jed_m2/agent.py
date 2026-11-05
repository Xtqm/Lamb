"""
JED-M2 Interactive Agent Runtime.
Provides:
- AgentStepOutput: Structured container for decoded dual-system action decisions.
- TurnRecord: Multi-turn interaction history log entry.
- JEDM2Agent: Live execution engine wrapping JEDM2Model for closed-loop decision making:
  - O(1) constant-memory recurrent rollouts via model.step(...) preserving layer-wise
    Mamba SSM rolling buffers and MLA KV-cache across multiple interaction turns.
  - Decoupled System 1 / System 2 execution loop: Autoregressively generates latent
    rationale tokens (System 2) until ActionGate signals commitment, then decodes
    action primitive, continuous (x, y) coordinates in [0, 1], and argument payloads (System 1).
  - Observation parsing, environment dispatching, and turn history management.

STRICT BUILD RULE: 100% native PyTorch without external compiled C++/CUDA packages.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.config import JEDM2Config
from jed_m2.core.features import compute_obs_features
from jed_m2.dataset import SimpleLAMTokenizer
from jed_m2.environment import StepResult, WebOSEnvironment
from jed_m2.heads import ActionOutput, ActionType
from jed_m2.model import JEDM2Model


# ==============================================================================
# 1. Output & History Data Containers
# ==============================================================================

@dataclass
class AgentStepOutput:
    """Structured decision output emitted by JEDM2Agent on each turn."""

    action_type: ActionType
    action_type_id: int
    action_name: str
    coordinates: Tuple[float, float]
    argument: str
    thought: str
    gate_prob: float
    num_thought_tokens: int
    raw_coordinates: Tuple[float, float] = (0.0, 0.0)
    raw_action_output: Optional[ActionOutput] = None
    jepa_planning_scores: Optional[Dict[str, float]] = None
    is_terminal: bool = False
    is_delta: bool = False
    obs_token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Converts agent step output to dictionary."""
        res = {
            "action_type": self.action_name,
            "action_type_id": self.action_type_id,
            "coordinates": self.coordinates,
            "raw_coordinates": self.raw_coordinates,
            "argument": self.argument,
            "thought": self.thought,
            "gate_prob": self.gate_prob,
            "num_thought_tokens": self.num_thought_tokens,
            "is_terminal": self.is_terminal,
            "is_delta": self.is_delta,
            "obs_token_count": self.obs_token_count,
        }
        if self.jepa_planning_scores is not None:
            res["jepa_planning_scores"] = self.jepa_planning_scores
        return res

    def summary_str(self) -> str:
        """Returns concise single-line human-readable summary of the action decision."""
        cx, cy = self.coordinates
        arg_repr = f" arg='{self.argument}'" if self.argument else ""
        plan_repr = " [JEPA-planned]" if self.jepa_planning_scores else ""
        return (
            f"Action: {self.action_name} at ({cx:.2f}, {cy:.2f}){arg_repr}{plan_repr} "
            f"[Gate={self.gate_prob:.3f}, ThoughtTokens={self.num_thought_tokens}]"
        )


def _extract_element_center(element: Any) -> Optional[Tuple[float, float]]:
    """Extracts (center_x, center_y) normalized coordinates in [0, 1] from an element or dict."""
    if hasattr(element, "center") and element.center is not None:
        try:
            return (float(element.center[0]), float(element.center[1]))
        except (IndexError, TypeError):
            pass
    if hasattr(element, "bbox") and element.bbox is not None:
        try:
            b = element.bbox
            if len(b) >= 4:
                return (float(b[0] + b[2]) / 2.0, float(b[1] + b[3]) / 2.0)
        except (IndexError, TypeError):
            pass
    if isinstance(element, dict):
        if "center" in element and isinstance(element["center"], (list, tuple)) and len(element["center"]) >= 2:
            return (float(element["center"][0]), float(element["center"][1]))
        if "bbox" in element and isinstance(element["bbox"], (list, tuple)) and len(element["bbox"]) >= 4:
            b = element["bbox"]
            return (float(b[0] + b[2]) / 2.0, float(b[1] + b[3]) / 2.0)
    return None


@dataclass
class TurnRecord:
    """Historical record of an interaction turn."""

    turn_id: int
    obs_text: str
    thought: str
    action_name: str
    coordinates: Tuple[float, float]
    argument: str
    reward: float
    done: bool
    gate_prob: float
    is_delta: bool = False
    delta_tokens_count: int = 0
    prefill_savings: int = 0


# ==============================================================================
# 2. Interactive Agent Runtime Engine
# ==============================================================================

class JEDM2Agent:
    """
    Interactive LAM execution engine wrapping JEDM2Model for live closed-loop decision making.
    Maintains layer-wise Mamba SSM rolling buffers and MLA KV-cache for O(1) constant-memory
    autoregressive rollout across arbitrary multi-turn horizons.
    """

    def __init__(
        self,
        model: JEDM2Model,
        config: Optional[JEDM2Config] = None,
        tokenizer: Optional[ActionBPETokenizer] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.bfloat16,
        max_thought_tokens: int = 16,
        action_gate_threshold: float = 0.5,
        use_jepa_planning: bool = False,
        jepa_planning_lambda: float = 1.0,
        snap_coordinates_to_candidates: bool = True,
        max_snap_distance: Optional[float] = 0.35,
    ):
        self.model = model
        self.config = config if config is not None else getattr(model, "config", JEDM2Config())
        self.tokenizer = tokenizer if tokenizer is not None else ActionBPETokenizer(self.config.vocab_size)

        if device is None:
            # Infer device from model parameters
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        self.dtype = dtype
        self.max_thought_tokens = max_thought_tokens
        self.action_gate_threshold = action_gate_threshold
        self.use_jepa_planning = use_jepa_planning
        self.jepa_planning_lambda = jepa_planning_lambda
        self.snap_coordinates_to_candidates = snap_coordinates_to_candidates
        self.max_snap_distance = max_snap_distance

        # Best-of-N JEPA Test-Time Search Harness
        self.search_harness = None
        if getattr(model, "jepa", None) is not None:
            from jed_m2.jepa import JEPASearchHarness
            self.search_harness = JEPASearchHarness(
                jepa_model=model.jepa,
                plan_lambda=jepa_planning_lambda,
                device=self.device,
                dtype=self.dtype,
            )

        # Ensure model is in evaluation mode for live rollout
        self.model.eval()

        # Recurrent rollout states: List of (ssm_state, conv_state, kv_cache) per layer
        self.recurrent_states: Optional[List[Tuple[Any, Any, Any]]] = None
        self.current_pos: int = 0
        self.history: List[TurnRecord] = []
        self.turn_count: int = 0
        self.cached_prompt_hidden: Optional[torch.Tensor] = None
        self.cached_prompt_ids: Optional[torch.Tensor] = None
        self.latent_history: List[torch.Tensor] = []

    def reset(self) -> None:
        """
        Reinitializes agent runtime for a new episode.
        Clears recurrent state buffers, resets position counter, and flushes turn history.
        """
        self.recurrent_states = None
        self.current_pos = 0
        self.history.clear()
        self.latent_history.clear()
        self.turn_count = 0
        self.cached_prompt_hidden = None
        self.cached_prompt_ids = None

    def _feed_single_token(
        self,
        token_id: int,
        visual_tokens: Optional[torch.Tensor] = None,
        use_prompt_context: bool = False,
    ) -> Tuple[torch.Tensor, ActionOutput]:
        """
        Feeds a single token ID through the model's recurrent step method.
        Preserves and updates recurrent states and position counter.
        Supports visual patch tokens for grounded action decoding and prompt conditioning for pointer copy.

        Args:
            token_id: Integer token ID.
            visual_tokens: Optional visual patch tokens [1, N_v, hidden_dim].
            use_prompt_context: If True, supplies cached prompt representations for pointer copy.

        Returns:
            Tuple of:
            - lm_logits: Tensor of shape [1, vocab_size]
            - action_output: ActionOutput container
        """
        token_tensor = torch.tensor([token_id], device=self.device, dtype=torch.long)
        prompt_h = self.cached_prompt_hidden if use_prompt_context else None
        prompt_ids = self.cached_prompt_ids if use_prompt_context else None
        with torch.no_grad():
            lm_logits, action_output, self.recurrent_states = self.model.step(
                token_tensor,
                states=self.recurrent_states,
                pos=self.current_pos,
                visual_tokens=visual_tokens,
                prompt_hidden=prompt_h,
                prompt_ids=prompt_ids,
            )
        self.current_pos += 1
        return lm_logits, action_output

    def _feed_token_sequence(
        self,
        token_ids: List[int],
        visual_tokens: Optional[torch.Tensor] = None,
        cache_as_prompt: bool = True,
    ) -> Tuple[torch.Tensor, ActionOutput]:
        """
        Feeds a sequence of token IDs through parallel prompt prefill (model.prefill),
        advancing recurrent states and position counter in a single parallel tensor pass.
        Caches prompt hidden representations for subsequent pointer-generator copy decoding.

        Args:
            token_ids: List of integer token IDs.
            visual_tokens: Optional visual patch tokens [1, N_v, hidden_dim].
            cache_as_prompt: If True, preserves prompt representations in cache for pointer copying.

        Returns:
            Final step's (lm_logits, action_output).
        """
        if not token_ids:
            raise ValueError("Token sequence cannot be empty.")

        device = self.device
        max_prompt_len = max(16, self.config.max_seq_len - self.max_thought_tokens)
        if len(token_ids) > max_prompt_len:
            token_ids = token_ids[-max_prompt_len:]

        inp_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        start_pos = self.current_pos

        with torch.no_grad():
            last_lm_logits, last_action_out, new_states, prompt_h = self.model.prefill(
                inp_tensor,
                states=self.recurrent_states,
                start_pos=start_pos,
                visual_tokens=visual_tokens,
                return_prompt_hidden=True,
            )

        self.recurrent_states = new_states
        self.current_pos = start_pos + len(token_ids)
        if cache_as_prompt:
            self.cached_prompt_hidden = prompt_h
            self.cached_prompt_ids = inp_tensor
        return last_lm_logits, last_action_out

    def predict_next_latent_state(
        self,
        z_t: torch.Tensor,
        action: Union[int, ActionType, torch.Tensor],
        coords: Optional[Union[Tuple[float, float], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Predicts expected next abstract state z_hat_{t+1} using the JEPA world model predictor.

        Args:
            z_t: Current latent state tensor of shape [1, D_latent] or [D_latent].
            action: Integer action type ID, ActionType enum, or action tensor.
            coords: Optional continuous normalized screen coordinates (x, y) or tensor [B, 2].

        Returns:
            Predicted next latent state tensor z_hat_{t+1} of shape [1, D_latent].
        """
        if getattr(self.model, "jepa", None) is None:
            raise RuntimeError("Model does not have a configured JEPA world model.")

        if isinstance(action, ActionType):
            act_id = action.value
            act_tensor = torch.tensor([act_id], device=self.device, dtype=torch.long)
        elif isinstance(action, int):
            act_tensor = torch.tensor([action], device=self.device, dtype=torch.long)
        elif isinstance(action, torch.Tensor):
            act_tensor = action.to(device=self.device)
            if act_tensor.ndim == 0:
                act_tensor = act_tensor.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported action type: {type(action)}")

        coords_tensor: Optional[torch.Tensor] = None
        if coords is not None:
            if isinstance(coords, (tuple, list)):
                coords_tensor = torch.tensor([coords], device=self.device, dtype=self.dtype)
            elif isinstance(coords, torch.Tensor):
                coords_tensor = coords.to(device=self.device, dtype=self.dtype)
                if coords_tensor.ndim == 1:
                    coords_tensor = coords_tensor.unsqueeze(0)

        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            z_next = self.model.jepa.predict(z_t, act_tensor, coords=coords_tensor)
        return z_next

    def evaluate_action_candidates(
        self,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        candidate_actions: Optional[List[int]] = None,
    ) -> Dict[int, float]:
        """
        Evaluates candidate actions using JEPA predictive latent rollouts,
        computing cosine distance 1 - cos(z_hat_{t+1}, z_goal) for each candidate.

        Args:
            z_t: Current latent state tensor [1, D_latent].
            z_goal: Goal latent state tensor [1, D_latent].
            candidate_actions: List of candidate ActionType integer IDs.

        Returns:
            Dictionary mapping candidate ActionType ID to cosine distance (lower is closer to goal).
        """
        if candidate_actions is None:
            candidate_actions = [
                ActionType.CLICK.value,
                ActionType.TYPE.value,
                ActionType.SCROLL.value,
                ActionType.NAVIGATE.value,
            ]

        if z_goal.ndim == 1:
            z_goal = z_goal.unsqueeze(0)
        z_goal = z_goal.to(device=self.device, dtype=self.dtype)

        distances: Dict[int, float] = {}
        for act_id in candidate_actions:
            z_pred = self.predict_next_latent_state(z_t, act_id)
            sim = F.cosine_similarity(z_pred, z_goal, dim=-1).clamp(-1.0, 1.0).item()
            cos_dist = float(1.0 - sim)
            distances[act_id] = cos_dist

        return distances

    def evaluate_grounded_candidates(
        self,
        z_t: torch.Tensor,
        z_goal: torch.Tensor,
        action_logits: torch.Tensor,
        raw_coords: Tuple[float, float],
        candidate_elements: Optional[List[Any]] = None,
        candidate_actions: Optional[List[int]] = None,
        drift_threshold: float = 0.15,
        loop_threshold: float = 0.05,
        plan_lambda: float = 1.0,
        spatial_beta: float = 2.0,
    ) -> Tuple[int, Tuple[float, float], Dict[str, Any]]:
        """
        Evaluates candidate actions grounded on interactable DOM elements using 1-step JEPA lookahead,
        pruning dead-ends (goal drift) and cyclic navigation loops, and scoring candidates via MPC value fusion.

        Args:
            z_t: Current latent state [1, D_latent].
            z_goal: Task goal latent state [1, D_latent].
            action_logits: Unnormalized action primitive logits [num_action_types] from policy head.
            raw_coords: Predicted continuous coordinates (x, y) in [0, 1].
            candidate_elements: Optional list of interactable DOM candidate elements or dicts.
            candidate_actions: Optional list of action type IDs to consider.
            drift_threshold: Max allowed increase in goal distance before pruning (dead-end filter).
            loop_threshold: Min allowed distance to previous trajectory states before pruning (loop filter).
            plan_lambda: Goal distance weight in MPC score.
            spatial_beta: Coordinate distance penalty weight in MPC score.

        Returns:
            Tuple of:
            - selected_action_id: Integer action primitive ID.
            - selected_coords: Chosen normalized continuous coordinates (x, y).
            - planning_metadata: Dictionary of planning metrics, candidate scores, and pruning reasons.
        """
        if getattr(self.model, "jepa", None) is None:
            raise RuntimeError("Model does not have a configured JEPA world model.")

        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        z_t = z_t.to(device=self.device, dtype=self.dtype)

        if z_goal.ndim == 1:
            z_goal = z_goal.unsqueeze(0)
        z_goal = z_goal.to(device=self.device, dtype=self.dtype)

        curr_goal_dist = float(1.0 - F.cosine_similarity(z_t, z_goal, dim=-1).clamp(-1.0, 1.0).item())

        if action_logits.ndim > 1:
            logits_1d = action_logits.squeeze(0)
        else:
            logits_1d = action_logits

        top_act_id = int(torch.argmax(logits_1d, dim=-1).item())

        # Determine candidate action types
        if candidate_actions is not None:
            cand_act_ids = list(candidate_actions)
        else:
            top_k_indices = torch.topk(logits_1d, k=min(3, logits_1d.shape[-1])).indices.tolist()
            cand_act_ids = [top_act_id]
            for idx in top_k_indices:
                if idx not in cand_act_ids and idx in (
                    ActionType.CLICK.value,
                    ActionType.TYPE.value,
                    ActionType.SCROLL.value,
                    ActionType.NAVIGATE.value,
                    ActionType.FINISH.value,
                ):
                    cand_act_ids.append(idx)

        # Extract candidate element centers
        element_centers: List[Tuple[float, float]] = []
        if candidate_elements:
            for el in candidate_elements:
                c = _extract_element_center(el)
                if c is not None and 0.0 <= c[0] <= 1.0 and 0.0 <= c[1] <= 1.0:
                    element_centers.append(c)

        # Sort candidate centers by distance to raw_coords and take top 8
        rx, ry = raw_coords
        if element_centers:
            element_centers.sort(key=lambda pt: (pt[0] - rx) ** 2 + (pt[1] - ry) ** 2)
            element_centers = element_centers[:8]

        # Always include the raw predicted coordinates as a candidate target location
        all_centers = [raw_coords]
        for pt in element_centers:
            if (pt[0] - rx) ** 2 + (pt[1] - ry) ** 2 > 1e-5:
                all_centers.append(pt)

        # Construct candidate action-coordinate pairs
        cand_pairs: List[Tuple[int, Tuple[float, float]]] = []
        for act_id in cand_act_ids:
            if act_id in (ActionType.FINISH.value, ActionType.NAVIGATE.value):
                cand_pairs.append((act_id, (0.0, 0.0)))
            else:
                for pt in all_centers:
                    cand_pairs.append((act_id, pt))

        if not cand_pairs:
            return top_act_id, raw_coords, {"status": "no_candidates", "current_goal_dist": curr_goal_dist}

        # Operationalize Best-of-N JEPA Test-Time Search Harness
        from jed_m2.jepa import CandidateAction, JEPASearchHarness
        harness = self.search_harness
        if harness is None:
            harness = JEPASearchHarness(
                jepa_model=self.model.jepa,
                drift_threshold=drift_threshold,
                loop_threshold=loop_threshold,
                plan_lambda=plan_lambda,
                spatial_beta=spatial_beta,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            harness.drift_threshold = drift_threshold
            harness.loop_threshold = loop_threshold
            harness.plan_lambda = plan_lambda
            harness.spatial_beta = spatial_beta

        candidate_objs = [
            CandidateAction(
                action_type_id=p[0],
                coordinates=p[1],
                raw_score=float(logits_1d[p[0]].item()),
            )
            for p in cand_pairs
        ]

        search_res = harness.search(
            z_t=z_t,
            z_goal=z_goal,
            candidates=candidate_objs,
            latent_history=self.latent_history,
            pointer_prior=raw_coords,
        )

        details = [
            {
                "action": ActionType(c.action_type_id).name if 0 <= c.action_type_id < len(ActionType) else str(c.action_type_id),
                "action_id": c.action_type_id,
                "coords": c.coordinates,
                "goal_dist": c.goal_dist,
                "loop_dist": c.loop_dist,
                "drift": c.drift,
                "score": c.fused_score,
                "pruned": c.pruned,
                "prune_reason": c.prune_reason,
            }
            for c in search_res.candidates
        ]

        metadata = {
            "current_goal_dist": curr_goal_dist,
            "total_candidates": search_res.num_candidates_evaluated,
            "valid_candidates": search_res.num_candidates_evaluated - search_res.num_candidates_pruned,
            "fallback_used": search_res.fallback_used,
            "selected_action": ActionType(search_res.selected_action_id).name,
            "selected_coords": search_res.selected_coordinates,
            "selected_score": search_res.selected_score,
            "latency_ms": search_res.latency_ms,
            "candidates": details,
        }

        return search_res.selected_action_id, search_res.selected_coordinates, metadata

    def act(
        self,
        obs_text: str,
        obs_features: Optional[torch.Tensor] = None,
        screenshot: Optional[Union[torch.Tensor, Any]] = None,
        goal_embedding: Optional[torch.Tensor] = None,
        candidate_elements: Optional[List[Any]] = None,
        snap_coordinates: Optional[bool] = None,
        max_snap_distance: Optional[float] = None,
        use_jepa_planning: Optional[bool] = None,
        jepa_planning_lambda: Optional[float] = None,
    ) -> AgentStepOutput:
        """
        Executes a dual-system decision step given an environment observation:
        1. Encodes observation text and feeds tokens into the recurrent backbone.
        2. System 2 (Latent Reasoning): Autoregressively generates thought tokens
           until ActionGate exceeds the commitment threshold or maximum length is reached.
        3. System 1 (Action Decoding): Emits discrete primitive, normalized (x, y) coordinates,
           and argument payload, optionally grounded in visual screenshot patches, snapped to
           the nearest interactive DOM candidate element, and refined via JEPA planning.

        Args:
            obs_text: Raw observation text from environment state.
            obs_features: Optional dense observation embedding for JEPA world model.
            screenshot: Optional [3, H, W] RGB tensor or PIL Image for visual grounding.
            goal_embedding: Optional target state latent embedding z_goal for JEPA planning.
            candidate_elements: Optional list of candidate UIElements or dicts for coordinate snapping.
            snap_coordinates: Optional override to enable/disable candidate bounding box snapping.
            max_snap_distance: Optional maximum Euclidean distance threshold for snapping.
            use_jepa_planning: Optional override for active model-predictive JEPA planning.
            jepa_planning_lambda: Optional planning distance penalty factor.

        Returns:
            AgentStepOutput containing rationale, action primitive, coordinates, and argument.
        """
        # ======================================================================
        # 0. Visual Screenshot Encoding (Multimodal Grounding)
        # ======================================================================
        visual_tokens: Optional[torch.Tensor] = None
        if screenshot is not None and hasattr(self.model, "patch_encoder") and self.model.patch_encoder is not None:
            if isinstance(screenshot, torch.Tensor):
                img_t = screenshot
                if img_t.ndim == 3:
                    # [3, H, W] -> [1, 3, H, W]
                    img_t = img_t.unsqueeze(0)
                img_t = img_t.to(device=self.device, dtype=torch.float32)
                with torch.no_grad():
                    # [1, 3, H, W] -> [1, N_v, hidden_dim]
                    visual_tokens = self.model.patch_encoder(img_t)
            elif hasattr(screenshot, "size"):  # PIL Image
                import numpy as np
                img_np = np.array(screenshot.convert("RGB"), dtype=np.float32) / 255.0
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(
                    device=self.device, dtype=torch.float32
                )
                with torch.no_grad():
                    visual_tokens = self.model.patch_encoder(img_t)

        # ======================================================================
        # 1. Observation Prompt Encoding & Ingestion
        # ======================================================================
        # Format turn prompt
        is_delta = "<obs delta>" in obs_text
        if obs_text.startswith("<obs"):
            prompt = f"{obs_text} <thought>"
        else:
            prompt = f"<obs> {obs_text} </obs> <thought>"

        if self.turn_count == 0 and "<bos>" not in prompt:
            prompt = f"<bos> {prompt}"

        obs_tokens = self.tokenizer.encode(prompt)
        obs_token_count = len(obs_tokens)
        # Advance recurrent states through the observation prompt
        lm_logits, action_output = self._feed_token_sequence(obs_tokens)

        # ======================================================================
        # 2. System 2: Autoregressive Latent Reasoning Loop
        # ======================================================================
        thought_tokens: List[int] = []
        gate_prob = float(torch.sigmoid(action_output.gate_logits).item())
        end_thought_id = self.tokenizer.special_tokens["</thought>"]

        # Loop until action gate threshold exceeded or max reasoning tokens reached
        while (
            gate_prob < self.action_gate_threshold
            and len(thought_tokens) < self.max_thought_tokens
        ):
            # Greedy argmax next token from System 2 LM rationale head
            next_token_id = int(torch.argmax(lm_logits, dim=-1).item())

            # Stop if model emits explicit end-of-thought token
            if next_token_id == end_thought_id:
                break

            thought_tokens.append(next_token_id)

            # Step next thought token into recurrent state
            lm_logits, action_output = self._feed_single_token(next_token_id)
            gate_prob = float(torch.sigmoid(action_output.gate_logits).item())

        # ======================================================================
        # 3. System 1: Action Primitive, Coordinate, and Argument Decoding
        # ======================================================================
        # Advance through </thought> (if needed) and step into <action> trigger token
        # where the System 1 action heads were trained to emit decisions
        if not thought_tokens or thought_tokens[-1] != end_thought_id:
            lm_logits, action_output = self._feed_single_token(end_thought_id)

        action_token_id = self.tokenizer.special_tokens["<action>"]
        # Ground coordinate head in visual patch tokens and condition on prompt representations for pointer copy
        lm_logits, action_output = self._feed_single_token(
            action_token_id, visual_tokens=visual_tokens, use_prompt_context=True
        )
        gate_prob = float(torch.sigmoid(action_output.gate_logits).item())

        # Spatial continuous screen coordinates in [0, 1]
        raw_coords = action_output.coordinates.squeeze(0)  # [2]
        coord_x = float(raw_coords[0].clamp(0.0, 1.0).item())
        coord_y = float(raw_coords[1].clamp(0.0, 1.0).item())
        raw_coords_tuple = (coord_x, coord_y)

        # Action primitive and spatial decoding (with optional JEPA grounded MPC planning)
        should_plan = self.use_jepa_planning if use_jepa_planning is None else use_jepa_planning
        plan_lambda = self.jepa_planning_lambda if jepa_planning_lambda is None else jepa_planning_lambda
        jepa_planning_scores: Optional[Dict[str, float]] = None

        if should_plan and getattr(self.model, "jepa", None) is not None:
            # Prepare current state latent representation z_t
            if obs_features is not None:
                feat = obs_features.to(device=self.device, dtype=self.dtype)
                if feat.ndim == 1:
                    feat = feat.unsqueeze(0)
                z_t = self.model.jepa.encode(feat, visual_tokens=visual_tokens)
            else:
                feat = compute_obs_features(obs_text, [], self.config.hidden_dim).to(device=self.device, dtype=self.dtype).unsqueeze(0)
                z_t = self.model.jepa.encode(feat, visual_tokens=visual_tokens)

            # Prepare goal representation z_goal
            if goal_embedding is not None:
                z_goal = goal_embedding.to(device=self.device, dtype=self.dtype)
                if z_goal.ndim == 1:
                    z_goal = z_goal.unsqueeze(0)
            else:
                z_goal = z_t

            # Execute grounded candidate evaluation with dead-end pruning and loop elimination
            sel_act_id, sel_coords, planning_meta = self.evaluate_grounded_candidates(
                z_t=z_t,
                z_goal=z_goal,
                action_logits=action_output.action_type_logits.squeeze(0),
                raw_coords=raw_coords_tuple,
                candidate_elements=candidate_elements,
                plan_lambda=plan_lambda,
            )

            action_type_id = sel_act_id
            coords = sel_coords
            jepa_planning_scores = {
                f"{c['action']}@{c['coords'][0]:.2f},{c['coords'][1]:.2f}": float(c["score"])
                for c in planning_meta.get("candidates", [])
            }
            jepa_planning_scores["selected_score"] = float(planning_meta.get("selected_score", 0.0))
            jepa_planning_scores["fallback_used"] = 1.0 if planning_meta.get("fallback_used", False) else 0.0
            jepa_planning_scores["total_candidates"] = float(planning_meta.get("total_candidates", len(planning_meta.get("candidates", []))))
            jepa_planning_scores["latency_ms"] = float(planning_meta.get("latency_ms", 0.0))

            # Record current latent state in history for future loop elimination
            self.latent_history.append(z_t.detach())
            if len(self.latent_history) > 16:
                self.latent_history.pop(0)
        else:
            action_type_id = int(torch.argmax(action_output.action_type_logits, dim=-1).item())

            # Candidate bounding box snapping
            should_snap = self.snap_coordinates_to_candidates if snap_coordinates is None else snap_coordinates
            snap_dist_limit = self.max_snap_distance if max_snap_distance is None else max_snap_distance

            coords = raw_coords_tuple
            if should_snap and candidate_elements and ActionType(action_type_id) in (ActionType.CLICK, ActionType.SELECT):
                valid_candidates = []
                for el in candidate_elements:
                    if getattr(el, "has_bbox", True):
                        c = _extract_element_center(el)
                        if c is not None:
                            valid_candidates.append(c)

                if valid_candidates:
                    best_center = min(
                        valid_candidates,
                        key=lambda c: (c[0] - coord_x) ** 2 + (c[1] - coord_y) ** 2,
                    )
                    dist_sq = (best_center[0] - coord_x) ** 2 + (best_center[1] - coord_y) ** 2
                    if snap_dist_limit is None or (dist_sq <= snap_dist_limit ** 2):
                        coords = (float(best_center[0]), float(best_center[1]))

        action_type = ActionType(action_type_id)

        # Argument payload decoding (autoregressive unrolling for typing actions)
        if action_type in (ActionType.TYPE, ActionType.SELECT):
            arg_start_id = self.tokenizer.special_tokens.get("<arg>", 9)
            arg_end_id = self.tokenizer.special_tokens.get("</arg>", 10)
            eos_id = getattr(self.tokenizer, "EOS_TOKEN_ID", 2)
            action_end_id = self.tokenizer.special_tokens.get("</action>", 8)
            space_ids = self.tokenizer.encode(" ") if hasattr(self.tokenizer, "encode") else []
            space_id = space_ids[0] if space_ids else None

            arg_tokens: List[int] = []
            cur_lm, cur_act = self._feed_single_token(arg_start_id, use_prompt_context=True)
            for step_idx in range(16):
                logits = cur_act.argument_logits
                # Suppress leading whitespace and consecutive repeated whitespace in argument generation
                if space_id is not None and (step_idx == 0 or (arg_tokens and arg_tokens[-1] == space_id)):
                    logits = logits.clone()
                    logits[..., space_id] = -1e9

                next_tok_id = int(torch.argmax(logits, dim=-1).item())
                if next_tok_id in (arg_end_id, eos_id, action_end_id):
                    break
                arg_tokens.append(next_tok_id)
                cur_lm, cur_act = self._feed_single_token(next_tok_id, use_prompt_context=True)
            self._feed_single_token(arg_end_id, use_prompt_context=False)

            if hasattr(self.tokenizer, "decode_argument"):
                arg_str = self.tokenizer.decode_argument(arg_tokens)
            else:
                raw_arg_str = self.tokenizer.decode(arg_tokens).strip()
                arg_str = " ".join([w for w in raw_arg_str.split() if not w.startswith("<") and not w.endswith(">")])
            if not arg_str:
                arg_token_id = int(torch.argmax(action_output.argument_logits, dim=-1).item())
                arg_str = self.tokenizer.decode([arg_token_id], skip_special_tokens=True).strip() if hasattr(self.tokenizer, "decode") else self.tokenizer.decode([arg_token_id]).strip()
        else:
            arg_token_id = int(torch.argmax(action_output.argument_logits, dim=-1).item())
            arg_str = self.tokenizer.decode([arg_token_id], skip_special_tokens=True).strip() if hasattr(self.tokenizer, "decode") else self.tokenizer.decode([arg_token_id]).strip()

        # Clean rationale text
        thought_text = self.tokenizer.decode(thought_tokens).strip()
        cleaned_thought = " ".join([w for w in thought_text.split() if w not in ("<thought>", "</thought>")])
        if not cleaned_thought:
            thought_text = f"Analyzed screen elements and committed to {action_type.name} at coordinates ({coord_x:.2f}, {coord_y:.2f})."
        else:
            thought_text = cleaned_thought

        # Advance recurrent trajectory with closing tokens for next turn context
        close_prompt = f"{action_type.name} </action>"
        close_tokens = self.tokenizer.encode(close_prompt)
        self._feed_token_sequence(close_tokens, cache_as_prompt=False)

        self.turn_count += 1
        is_terminal = (action_type == ActionType.FINISH)

        return AgentStepOutput(
            action_type=action_type,
            action_type_id=action_type_id,
            action_name=action_type.name,
            coordinates=coords,
            argument=arg_str,
            thought=thought_text,
            gate_prob=gate_prob,
            num_thought_tokens=len(thought_tokens),
            raw_coordinates=raw_coords_tuple,
            raw_action_output=action_output,
            jepa_planning_scores=jepa_planning_scores,
            is_terminal=is_terminal,
            is_delta=is_delta,
            obs_token_count=obs_token_count,
        )

    def step(
        self,
        obs_text: str,
        obs_features: Optional[torch.Tensor] = None,
        screenshot: Optional[Union[torch.Tensor, Any]] = None,
        goal_embedding: Optional[torch.Tensor] = None,
        candidate_elements: Optional[List[Any]] = None,
        snap_coordinates: Optional[bool] = None,
        max_snap_distance: Optional[float] = None,
        use_jepa_planning: Optional[bool] = None,
        jepa_planning_lambda: Optional[float] = None,
    ) -> AgentStepOutput:
        """
        Step agent policy on a single observation. Alias for act().
        """
        return self.act(
            obs_text=obs_text,
            obs_features=obs_features,
            screenshot=screenshot,
            goal_embedding=goal_embedding,
            candidate_elements=candidate_elements,
            snap_coordinates=snap_coordinates,
            max_snap_distance=max_snap_distance,
            use_jepa_planning=use_jepa_planning,
            jepa_planning_lambda=jepa_planning_lambda,
        )

    def step_environment(
        self,
        env: Union[WebOSEnvironment, Any],
    ) -> Tuple[AgentStepOutput, StepResult]:
        """
        Executes a single interactive turn against a WebOSEnvironment or PlaywrightBrowserEnvironment:
        1. Obtains observation text, features, and optional screenshot from environment.
        2. Emits dual-system decision (rationale + visually-grounded action).
        3. Dispatches action into environment.
        4. Logs turn into history and returns (agent_output, step_result).
        """
        screenshot: Optional[torch.Tensor] = None
        if env.current_state is None:
            init_res = env.reset()
            obs_text = init_res.obs_text
            obs_features = init_res.obs_features
            screenshot = getattr(init_res, "screenshot", None)
            if screenshot is None and isinstance(init_res.info, dict):
                screenshot = init_res.info.get("screenshot", None)
        else:
            if hasattr(env, "_format_observation_text"):
                try:
                    obs_text = env._format_observation_text(is_delta=True)
                except TypeError:
                    obs_text = env._format_observation_text()
            else:
                obs_text = getattr(env.current_state, "last_action_str", "") or "Observation"
            if hasattr(env, "_generate_obs_features"):
                obs_features = env._generate_obs_features()
            else:
                obs_features = getattr(env.current_state, "obs_features", None)
            if hasattr(env, "capture_screenshot"):
                screenshot = env.capture_screenshot(as_tensor=True)

        candidate_elements = None
        if hasattr(env, "current_state") and env.current_state is not None:
            candidate_elements = getattr(env.current_state, "elements", None)
        if candidate_elements is None and hasattr(env, "active_elements"):
            candidate_elements = getattr(env, "active_elements", None)

        goal_embedding = getattr(env, "goal_embedding", None)
        agent_output = self.act(
            obs_text=obs_text,
            obs_features=obs_features,
            screenshot=screenshot,
            goal_embedding=goal_embedding,
            candidate_elements=candidate_elements,
        )

        step_res = env.step(
            action_type=agent_output.action_type,
            coordinates=agent_output.coordinates,
            argument=agent_output.argument,
        )

        is_done = step_res.done or agent_output.is_terminal
        is_delta = getattr(agent_output, "is_delta", False) or ("<obs delta>" in obs_text)
        delta_tokens_count = getattr(agent_output, "obs_token_count", 0)
        prefill_savings = max(0, 350 - delta_tokens_count) if is_delta else 0

        record = TurnRecord(
            turn_id=self.turn_count,
            obs_text=obs_text,
            thought=agent_output.thought,
            action_name=agent_output.action_name,
            coordinates=agent_output.coordinates,
            argument=agent_output.argument,
            reward=step_res.reward,
            done=is_done,
            gate_prob=agent_output.gate_prob,
            is_delta=is_delta,
            delta_tokens_count=delta_tokens_count,
            prefill_savings=prefill_savings,
        )
        self.history.append(record)

        return agent_output, step_res

    def run_episode(
        self,
        env: Union[WebOSEnvironment, Any],
        max_turns: int = 10,
    ) -> List[TurnRecord]:
        """
        Executes a complete closed-loop episode in the environment until termination or max_turns.
        Supports both simulated WebOSEnvironment and live PlaywrightBrowserEnvironment.
        """
        self.reset()
        env.reset()

        for _ in range(max_turns):
            agent_out, step_res = self.step_environment(env)
            if agent_out.is_terminal or step_res.done:
                break

        return list(self.history)
