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
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from jed_m2.config import JEDM2Config
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
    raw_action_output: Optional[ActionOutput] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts agent step output to dictionary."""
        return {
            "action_type": self.action_name,
            "action_type_id": self.action_type_id,
            "coordinates": self.coordinates,
            "argument": self.argument,
            "thought": self.thought,
            "gate_prob": self.gate_prob,
            "num_thought_tokens": self.num_thought_tokens,
        }

    def summary_str(self) -> str:
        """Returns concise single-line human-readable summary of the action decision."""
        cx, cy = self.coordinates
        arg_repr = f" arg='{self.argument}'" if self.argument else ""
        return (
            f"Action: {self.action_name} at ({cx:.2f}, {cy:.2f}){arg_repr} "
            f"[Gate={self.gate_prob:.3f}, ThoughtTokens={self.num_thought_tokens}]"
        )


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
            # Infer device from model parameters
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)

        self.dtype = dtype
        self.max_thought_tokens = max_thought_tokens
        self.action_gate_threshold = action_gate_threshold

        # Ensure model is in evaluation mode for live rollout
        self.model.eval()

        # Recurrent rollout states: List of (ssm_state, conv_state, kv_cache) per layer
        self.recurrent_states: Optional[List[Tuple[Any, Any, Any]]] = None
        self.current_pos: int = 0
        self.history: List[TurnRecord] = []
        self.turn_count: int = 0

    def reset(self) -> None:
        """
        Reinitializes agent runtime for a new episode.
        Clears recurrent state buffers, resets position counter, and flushes turn history.
        """
        self.recurrent_states = None
        self.current_pos = 0
        self.history.clear()
        self.turn_count = 0

    def _feed_single_token(
        self,
        token_id: int,
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ActionOutput]:
        """
        Feeds a single token ID through the model's recurrent step method.
        Preserves and updates recurrent states and position counter.
        Supports visual patch tokens for grounded action decoding.

        Args:
            token_id: Integer token ID.
            visual_tokens: Optional visual patch tokens [1, N_v, hidden_dim].

        Returns:
            Tuple of:
            - lm_logits: Tensor of shape [1, vocab_size]
            - action_output: ActionOutput container
        """
        token_tensor = torch.tensor([token_id], device=self.device, dtype=torch.long)
        with torch.no_grad():
            lm_logits, action_output, self.recurrent_states = self.model.step(
                token_tensor,
                states=self.recurrent_states,
                pos=self.current_pos,
                visual_tokens=visual_tokens,
            )
        self.current_pos += 1
        return lm_logits, action_output

    def _feed_token_sequence(
        self,
        token_ids: List[int],
        visual_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ActionOutput]:
        """
        Sequentially feeds a sequence of token IDs through the recurrent rollout.

        Args:
            token_ids: List of integer token IDs.
            visual_tokens: Optional visual patch tokens [1, N_v, hidden_dim].

        Returns:
            Final step's (lm_logits, action_output).
        """
        if not token_ids:
            raise ValueError("Token sequence cannot be empty.")

        last_lm_logits = None
        last_action_out = None
        for tok in token_ids:
            last_lm_logits, last_action_out = self._feed_single_token(tok, visual_tokens=visual_tokens)
        return last_lm_logits, last_action_out

    def act(
        self,
        obs_text: str,
        obs_features: Optional[torch.Tensor] = None,
        screenshot: Optional[Union[torch.Tensor, Any]] = None,
    ) -> AgentStepOutput:
        """
        Executes a dual-system decision step given an environment observation:
        1. Encodes observation text and feeds tokens into the recurrent backbone.
        2. System 2 (Latent Reasoning): Autoregressively generates thought tokens
           until ActionGate exceeds the commitment threshold or maximum length is reached.
        3. System 1 (Action Decoding): Emits discrete primitive, normalized (x, y) coordinates,
           and argument payload, optionally grounded in visual screenshot patches.

        Args:
            obs_text: Raw observation text from environment state.
            obs_features: Optional dense observation embedding for JEPA world model.
            screenshot: Optional [3, H, W] RGB tensor or PIL Image for visual grounding.

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
        prompt = f"<obs> {obs_text} </obs> <thought>"
        if self.turn_count == 0:
            prompt = f"<bos> {prompt}"

        obs_tokens = self.tokenizer.encode(prompt)
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
        # Ground coordinate head in visual patch tokens if available
        lm_logits, action_output = self._feed_single_token(action_token_id, visual_tokens=visual_tokens)
        gate_prob = float(torch.sigmoid(action_output.gate_logits).item())

        # Argmax discrete primitive
        action_type_id = int(torch.argmax(action_output.action_type_logits, dim=-1).item())
        action_type = ActionType(action_type_id)

        # Spatial continuous screen coordinates in [0, 1]
        raw_coords = action_output.coordinates.squeeze(0)  # [2]
        coord_x = float(raw_coords[0].clamp(0.0, 1.0).item())
        coord_y = float(raw_coords[1].clamp(0.0, 1.0).item())
        coords = (coord_x, coord_y)

        # Argument payload decoding
        arg_token_id = int(torch.argmax(action_output.argument_logits, dim=-1).item())
        arg_str = self.tokenizer.decode([arg_token_id]).strip()

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
        self._feed_token_sequence(close_tokens)

        self.turn_count += 1

        return AgentStepOutput(
            action_type=action_type,
            action_type_id=action_type_id,
            action_name=action_type.name,
            coordinates=coords,
            argument=arg_str,
            thought=thought_text,
            gate_prob=gate_prob,
            num_thought_tokens=len(thought_tokens),
            raw_action_output=action_output,
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
            obs_text = env._format_observation_text()
            obs_features = env._generate_obs_features()
            if hasattr(env, "capture_screenshot"):
                screenshot = env.capture_screenshot(as_tensor=True)

        agent_output = self.act(obs_text=obs_text, obs_features=obs_features, screenshot=screenshot)

        step_res = env.step(
            action_type=agent_output.action_type,
            coordinates=agent_output.coordinates,
            argument=agent_output.argument,
        )

        record = TurnRecord(
            turn_id=self.turn_count,
            obs_text=obs_text,
            thought=agent_output.thought,
            action_name=agent_output.action_name,
            coordinates=agent_output.coordinates,
            argument=agent_output.argument,
            reward=step_res.reward,
            done=step_res.done,
            gate_prob=agent_output.gate_prob,
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
            if step_res.done:
                break

        return list(self.history)
