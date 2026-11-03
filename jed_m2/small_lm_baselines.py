"""
JED-M2 Small Language Model Baselines (T5-Small & DistilGPT-2).
Provides sequence serialization, action parsing, fine-tuning wrappers, and evaluation agents
for empirical comparison against JED-M2 on identical Mind2Web splits.

STRICT CONVENTION: 100% native PyTorch without external compiled C++/CUDA packages.
STRICT CONVENTION: Pure 7-bit ASCII logging (no Unicode box-drawing characters).
"""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from jed_m2.core.types import UIElement
from jed_m2.heads import ActionType

logger = logging.getLogger("jed_m2.small_lm_baselines")

# Special control tokens for browser interaction sequences
SPECIAL_CONTROL_TOKENS: List[str] = [
    "[GOAL]",
    "[OBS]",
    "[HISTORY]",
    "[ACTION]",
    "[ARG]",
    "[/ARG]",
    "<thought>",
    "</thought>",
]


# ==============================================================================
# 1. Action String Parsing & Structured Container
# ==============================================================================

@dataclass
class ParsedAction:
    """Structured representation of an action decoded from an LM text emission."""

    action_type: ActionType
    coordinates: Tuple[float, float]
    argument: str = ""
    thought: str = ""
    is_valid: bool = True
    raw_text: str = ""


ACTION_NAME_TO_TYPE: Dict[str, ActionType] = {
    "CLICK": ActionType.CLICK,
    "TYPE": ActionType.TYPE,
    "SCROLL": ActionType.SCROLL,
    "NAVIGATE": ActionType.NAVIGATE,
    "API_CALL": ActionType.API_CALL,
    "FINISH": ActionType.FINISH,
    "SELECT": ActionType.SELECT,
}


def parse_action_string(
    text: str,
    fallback_elements: Optional[List[UIElement]] = None,
) -> ParsedAction:
    """
    Parses generated model text into a structured ParsedAction object.
    Robustly handles formatted output:
      <thought> ... </thought> [ACTION] CLICK (0.350, 0.420) [ARG] ... [/ARG]
    as well as malformed or partially generated strings with sensible fallbacks.
    """
    cleaned = text.strip()

    # 1. Extract thought / rationale
    thought = ""
    thought_match = re.search(r"<thought>(.*?)</thought>", cleaned, re.IGNORECASE | re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # 2. Extract argument
    argument = ""
    arg_match = re.search(r"\[ARG\](.*?)\[/ARG\]", cleaned, re.IGNORECASE | re.DOTALL)
    if arg_match:
        argument = arg_match.group(1).strip()
    else:
        # Fallback: check unclosed [ARG] at the end
        unclosed_arg = re.search(r"\[ARG\](.*)$", cleaned, re.IGNORECASE | re.DOTALL)
        if unclosed_arg:
            argument = unclosed_arg.group(1).strip()

    # 3. Extract action type and coordinates
    action_type = ActionType.CLICK
    coords = (0.5, 0.5)
    is_valid = True

    # Primary regex: [ACTION] TYPE (x, y)
    action_match = re.search(
        r"\[ACTION\]\s*([A-Za-z_]+)\s*\(\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*\)",
        cleaned,
        re.IGNORECASE,
    )

    if action_match:
        act_str = action_match.group(1).upper()
        if act_str in ACTION_NAME_TO_TYPE:
            action_type = ACTION_NAME_TO_TYPE[act_str]
        else:
            is_valid = False

        try:
            cx = float(action_match.group(2))
            cy = float(action_match.group(3))
            coords = (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)))
        except ValueError:
            is_valid = False
    else:
        # Secondary fallback regex: look for any keyword
        found_act = False
        for name, atype in ACTION_NAME_TO_TYPE.items():
            if re.search(rf"\b{name}\b", cleaned, re.IGNORECASE):
                action_type = atype
                found_act = True
                break

        # Look for any coordinate pair (x, y) or [x, y]
        coord_match = re.search(
            r"[\(\[]\s*([-\d\.]+)\s*,\s*([-\d\.]+)\s*[\)\]]",
            cleaned,
        )
        if coord_match:
            try:
                cx = float(coord_match.group(1))
                cy = float(coord_match.group(2))
                coords = (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)))
            except ValueError:
                coords = fallback_elements[0].center if fallback_elements else (0.5, 0.5)
        else:
            coords = fallback_elements[0].center if fallback_elements else (0.5, 0.5)
            if not found_act:
                is_valid = False

    return ParsedAction(
        action_type=action_type,
        coordinates=coords,
        argument=argument,
        thought=thought,
        is_valid=is_valid,
        raw_text=cleaned,
    )


# ==============================================================================
# 2. Text-to-Action Sequence Formatter
# ==============================================================================

class DOMActionSequenceFormatter:
    """
    Serializes browser agent interaction turns into standardized input-output text strings
    for sequence-to-sequence and causal language model training and evaluation.
    """

    def __init__(self, max_elements: int = 32, max_text_len: int = 40):
        self.max_elements = max_elements
        self.max_text_len = max_text_len

    def format_element(self, el: UIElement) -> str:
        """Serializes a single UIElement into a compact DOM descriptor."""
        clean_text = " ".join(el.text.split())[:self.max_text_len] if el.text else ""
        x0, y0, x1, y1 = el.bbox
        bbox_str = f"[{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}]"
        role_part = f' role="{el.role}"' if el.role else ""
        text_part = f' text="{clean_text}"' if clean_text else ""
        return f"[ID {el.element_id}] {el.tag}{role_part}{text_part} bbox={bbox_str}"

    def format_turn_prompt(
        self,
        goal: str,
        elements: List[UIElement],
        history: Optional[List[str]] = None,
    ) -> str:
        """
        Builds the input sequence prompt:
        [GOAL] <goal> [OBS] <elem_0> ; <elem_1> ... [HISTORY] <prev_0> | <prev_1> ...
        """
        el_strs = [self.format_element(el) for el in elements[:self.max_elements]]
        obs_str = " ; ".join(el_strs) if el_strs else "[EMPTY_DOM]"

        hist_str = ""
        if history:
            hist_str = " | ".join(history[-4:])

        parts = [f"[GOAL] {goal.strip()}", f"[OBS] {obs_str}"]
        if hist_str:
            parts.append(f"[HISTORY] {hist_str}")

        return " ".join(parts)

    def format_target_action(
        self,
        action_type: Union[ActionType, int, str],
        coordinates: Tuple[float, float],
        argument: str = "",
        thought: str = "",
    ) -> str:
        """
        Builds the target sequence string:
        <thought> <thought> </thought> [ACTION] <TYPE> (<x>, <y>) [ARG] <arg> [/ARG]
        """
        if isinstance(action_type, (ActionType, int)):
            act_name = ActionType(action_type).name
        else:
            act_name = str(action_type).upper()

        x, y = coordinates
        thought_str = thought.strip() if thought else f"Execute {act_name} on target element."
        arg_str = argument.strip()

        return (
            f"<thought> {thought_str} </thought> "
            f"[ACTION] {act_name} ({x:.3f}, {y:.3f}) "
            f"[ARG] {arg_str} [/ARG]"
        )


# ==============================================================================
# 3. T5-Small Sequence-to-Sequence Baseline Adaptor
# ==============================================================================

class T5SmallActionAgent(nn.Module):
    """
    Encoder-Decoder baseline agent based on Google T5-Small (~60.5M parameters).
    Accepts serialized DOM prompts, emits autoregressive action strings,
    and supports teacher-forced sequence cross-entropy training.
    """

    def __init__(
        self,
        model_name_or_path: str = "t5-small",
        local_files_only: bool = True,
        max_source_len: int = 512,
        max_target_len: int = 48,
    ):
        super().__init__()
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.formatter = DOMActionSequenceFormatter()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            )
        except Exception as e:
            logger.warning(
                "Could not load T5-Small with local_files_only=%s (%s). Attempting online fetch...",
                local_files_only,
                e,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

        # Add special tokens
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_CONTROL_TOKENS})
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def active_parameters(self) -> int:
        return self.total_parameters

    def compute_loss(
        self,
        prompts: List[str],
        targets: List[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Computes seq2seq cross-entropy loss with padding tokens masked with -100."""
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_source_len,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=self.max_target_len,
            return_tensors="pt",
        ).input_ids

        # Mask padding in labels
        labels[labels == self.tokenizer.pad_token_id] = -100

        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

        outputs = self.model(**inputs, labels=labels)
        return outputs.loss

    @torch.no_grad()
    def act(
        self,
        prompt: str,
        available_elements: Optional[List[UIElement]] = None,
        max_new_tokens: int = 40,
        device: Optional[torch.device] = None,
    ) -> ParsedAction:
        """Generates an action string greedily and parses it into a ParsedAction."""
        dev = device or next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_source_len,
            return_tensors="pt",
        ).to(dev)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )

        decoded = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return parse_action_string(decoded, available_elements)

    def save_pretrained(self, save_directory: Union[str, Path]):
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(p))
        self.tokenizer.save_pretrained(str(p))

    def load_pretrained(self, load_directory: Union[str, Path]):
        p = Path(load_directory)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(p))
        self.tokenizer = AutoTokenizer.from_pretrained(str(p))


# ==============================================================================
# 4. DistilGPT-2 Causal LM Baseline Adaptor
# ==============================================================================

class DistilGPT2ActionAgent(nn.Module):
    """
    Causal language model baseline agent based on Hugging Face DistilGPT-2 (~81.9M parameters).
    Accepts concatenated [Prompt] [Target] sequences with prompt tokens masked from next-token loss.
    """

    def __init__(
        self,
        model_name_or_path: str = "distilbert/distilgpt2",
        local_files_only: bool = True,
        max_seq_len: int = 576,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.formatter = DOMActionSequenceFormatter()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
            )
        except Exception as e:
            logger.warning(
                "Could not load DistilGPT-2 with local_files_only=%s (%s). Attempting online fetch...",
                local_files_only,
                e,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_CONTROL_TOKENS})
        if num_added > 0:
            self.model.resize_token_embeddings(len(self.tokenizer))

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def active_parameters(self) -> int:
        return self.total_parameters

    def compute_loss(
        self,
        prompts: List[str],
        targets: List[str],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Computes next-token prediction loss on target tokens, with prompt tokens
        and pad tokens masked out (-100).
        """
        dev = device or next(self.model.parameters()).device
        batch_input_ids = []
        batch_labels = []

        for prompt, target in zip(prompts, targets):
            t_ids = self.tokenizer.encode(f" {target.strip()}", add_special_tokens=False) + [self.tokenizer.eos_token_id]
            max_p_len = max(16, self.max_seq_len - len(t_ids))
            p_ids = self.tokenizer.encode(prompt.strip(), add_special_tokens=False)[:max_p_len]

            combined_ids = p_ids + t_ids
            combined_labels = [-100] * len(p_ids) + list(t_ids)

            batch_input_ids.append(torch.tensor(combined_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(combined_labels, dtype=torch.long))

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(dev)

        padded_labels = torch.nn.utils.rnn.pad_sequence(
            batch_labels,
            batch_first=True,
            padding_value=-100,
        ).to(dev)

        attention_mask = (padded_inputs != self.tokenizer.pad_token_id).long().to(dev)
        outputs = self.model(input_ids=padded_inputs, attention_mask=attention_mask, labels=padded_labels)
        return outputs.loss

    @torch.no_grad()
    def act(
        self,
        prompt: str,
        available_elements: Optional[List[UIElement]] = None,
        max_new_tokens: int = 40,
        device: Optional[torch.device] = None,
    ) -> ParsedAction:
        """Generates continuation tokens and parses action output."""
        dev = device or next(self.model.parameters()).device
        inputs = self.tokenizer(
            prompt.strip(),
            truncation=True,
            max_length=self.max_seq_len - max_new_tokens,
            return_tensors="pt",
        ).to(dev)

        input_len = inputs.input_ids.shape[1]
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_ids = output_ids[0, input_len:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        return parse_action_string(decoded, available_elements)

    def save_pretrained(self, save_directory: Union[str, Path]):
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(p))
        self.tokenizer.save_pretrained(str(p))

    def load_pretrained(self, load_directory: Union[str, Path]):
        p = Path(load_directory)
        self.model = AutoModelForCausalLM.from_pretrained(str(p))
        self.tokenizer = AutoTokenizer.from_pretrained(str(p))


# ==============================================================================
# 5. Duck-Typed Agent Interface Wrapper for Benchmark Evaluation
# ==============================================================================

class SmallLMAgentWrapper:
    """
    Duck-typed wrapper for T5SmallActionAgent and DistilGPT2ActionAgent
    to conform to JEDM2Agent interface in TaskEvaluator / evaluate_offline_dataset.
    """

    def __init__(
        self,
        baseline_model: Union[T5SmallActionAgent, DistilGPT2ActionAgent],
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ):
        from jed_m2.config import JEDM2Config

        self.baseline = baseline_model
        self.model = getattr(baseline_model, "model", baseline_model)
        self.device = device
        self.dtype = dtype or torch.float32
        self.config = getattr(baseline_model, "config", JEDM2Config())
        self.tokenizer = baseline_model.tokenizer
        self.formatter = DOMActionSequenceFormatter()
        self.history: List[str] = []

    def reset(self):
        """Clears dialog and action history at episode boundaries."""
        self.history = []

    def act(
        self,
        obs_text: str,
        obs_features: Optional[torch.Tensor] = None,
        screenshot: Optional[torch.Tensor] = None,
        candidate_elements: Optional[List[Any]] = None,
        available_elements: Optional[List[UIElement]] = None,
        **kwargs,
    ) -> Any:
        """Emits an action step conforming to JEDM2Agent.act signature."""
        from jed_m2.agent import AgentStepOutput

        elements = candidate_elements or available_elements
        prompt = obs_text
        if not prompt.startswith("[GOAL]"):
            prompt = f"[GOAL] Perform action on page [OBS] {obs_text}"
        if self.history:
            prompt += f" [HISTORY] {' | '.join(self.history[-3:])}"

        parsed: ParsedAction = self.baseline.act(
            prompt,
            available_elements=elements,
            device=self.device,
        )

        cx, cy = parsed.coordinates
        self.history.append(f"{parsed.action_type.name} ({cx:.2f}, {cy:.2f})")

        num_thought_tokens = 0
        if parsed.thought:
            try:
                num_thought_tokens = len(self.tokenizer.encode(parsed.thought))
            except Exception:
                num_thought_tokens = 4

        act_type = parsed.action_type
        act_id = int(act_type.value) if hasattr(act_type, "value") else int(act_type)
        act_name = act_type.name if hasattr(act_type, "name") else str(act_type)
        is_terminal = (act_type == ActionType.FINISH)

        return AgentStepOutput(
            action_type=act_type,
            action_type_id=act_id,
            action_name=act_name,
            coordinates=parsed.coordinates,
            argument=parsed.argument,
            thought=parsed.thought,
            gate_prob=1.0,
            num_thought_tokens=num_thought_tokens,
            raw_coordinates=parsed.coordinates,
            is_terminal=is_terminal,
        )

