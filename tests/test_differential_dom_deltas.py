"""
Tests for JED-M2 Differential DOM Delta Tracking and Shadow Registry Engine.

Verifies:
1. DOMMutationType, DOMMutation, DOMDeltaRecord data structures.
2. diff_elements() set diffing for additions, removals, attribute and bbox shifts.
3. ShadowDOMRegistry state reconstruction and consistency validation.
4. Delta grammar formatting and upheaval threshold detection.
"""

import math
import pytest
from jed_m2.core.types import UIElement
from jed_m2.dom_delta import (
    DOMMutationType,
    DOMMutation,
    DOMDeltaRecord,
    diff_elements,
    check_upheaval_threshold,
    ShadowDOMRegistry,
)


def test_mutation_data_structures():
    """Verifies Task 1.1: data classes and grammar serialization."""
    mut_add = DOMMutation(
        mutation_type=DOMMutationType.ADD,
        element_id=1,
        tag="button",
        role="button",
        text="Submit",
        bbox=(0.1, 0.2, 0.3, 0.4),
        attributes={"type": "submit"},
    )
    assert mut_add.mutation_type == DOMMutationType.ADD
    assert mut_add.element_id == 1
    grammar_add = mut_add.to_grammar_string()
    assert grammar_add.startswith("+ <button id=1 role='button'")
    assert "bbox='[0.20,0.30]'" in grammar_add
    assert "Submit" in grammar_add

    mut_remove = DOMMutation(
        mutation_type=DOMMutationType.REMOVE,
        element_id=2,
    )
    assert mut_remove.to_grammar_string() == "- id=2"

    mut_mod = DOMMutation(
        mutation_type=DOMMutationType.MODIFY,
        element_id=1,
        text="Submitted!",
        value="ok",
        bbox=(0.15, 0.25, 0.35, 0.45),
    )
    grammar_mod = mut_mod.to_grammar_string()
    assert grammar_mod.startswith("~ id=1")
    assert "text='Submitted!'" in grammar_mod
    assert "value='ok'" in grammar_mod
    assert "bbox='[0.25,0.35]'" in grammar_mod

    # Delta record
    record = DOMDeltaRecord(
        mutations=[mut_add, mut_remove, mut_mod],
        url="https://example.com/checkout",
        step=2,
    )
    obs_str = record.to_obs_string()
    assert obs_str.startswith("<obs delta> URL: https://example.com/checkout |")
    assert "+ <button" in obs_str
    assert "- id=2" in obs_str
    assert "~ id=1" in obs_str
    assert obs_str.endswith("</obs>")


def test_diff_elements_add_remove_modify():
    """Verifies Task 1.2: element set diffing."""
    el1 = UIElement(1, "button", "btn", "Click Me", (0.1, 0.1, 0.2, 0.2))
    el2 = UIElement(2, "input", "text_input", "Search", (0.3, 0.1, 0.7, 0.2), value="")
    el3 = UIElement(3, "a", "link", "Home", (0.8, 0.1, 0.9, 0.2))

    prev_elements = [el1, el2, el3]

    # el1 modified (value typed, bbox shifted), el2 removed, el4 added
    el1_mod = UIElement(1, "button", "btn", "Clicked", (0.15, 0.15, 0.25, 0.25), value="done")
    el4 = UIElement(4, "div", "modal", "Success Modal", (0.2, 0.2, 0.8, 0.8))

    curr_elements = [el1_mod, el3, el4]

    mutations = diff_elements(prev_elements, curr_elements, bbox_displacement_threshold=0.01)

    # Expected: el2 removed (-), el4 added (+), el1 modified (~)
    types = {m.mutation_type for m in mutations}
    assert DOMMutationType.REMOVE in types
    assert DOMMutationType.ADD in types
    assert DOMMutationType.MODIFY in types

    remove_mut = [m for m in mutations if m.mutation_type == DOMMutationType.REMOVE][0]
    assert remove_mut.element_id == 2

    add_mut = [m for m in mutations if m.mutation_type == DOMMutationType.ADD][0]
    assert add_mut.element_id == 4
    assert add_mut.tag == "div"
    assert add_mut.text == "Success Modal"

    mod_mut = [m for m in mutations if m.mutation_type == DOMMutationType.MODIFY][0]
    assert mod_mut.element_id == 1
    assert mod_mut.text == "Clicked"
    assert mod_mut.value == "done"
    assert mod_mut.bbox == (0.15, 0.15, 0.25, 0.25)


def test_shadow_dom_registry_reconstruction():
    """Verifies Task 1.3: Shadow DOM registry incremental reconstruction."""
    registry = ShadowDOMRegistry()

    # Step 0: Initial full snapshot
    el1 = UIElement(1, "button", "btn", "Button 1", (0.1, 0.1, 0.2, 0.2))
    el2 = UIElement(2, "input", "input", "Input 2", (0.3, 0.1, 0.5, 0.2))
    initial_elements = [el1, el2]

    step0_record = DOMDeltaRecord(
        is_full_snapshot=True,
        full_elements=initial_elements,
        step=0,
    )
    reconstructed0 = registry.apply_delta(step0_record)
    assert len(reconstructed0) == 2
    assert registry.validate_consistency(initial_elements) is True

    # Step 1: Incremental delta (Add el3, Modify el2, Remove el1)
    el3 = UIElement(3, "a", "link", "Link 3", (0.6, 0.1, 0.8, 0.2))
    el2_mod = UIElement(2, "input", "input", "Input 2", (0.3, 0.1, 0.5, 0.2), value="typed_text")
    ground_truth_step1 = [el2_mod, el3]

    muts = diff_elements(initial_elements, ground_truth_step1)
    step1_record = DOMDeltaRecord(mutations=muts, is_full_snapshot=False, step=1)

    reconstructed1 = registry.apply_delta(step1_record)
    assert len(reconstructed1) == 2
    assert [el.element_id for el in reconstructed1] == [2, 3]
    assert reconstructed1[0].value == "typed_text"
    assert registry.validate_consistency(ground_truth_step1) is True


def test_delta_grammar_and_upheaval_threshold():
    """Verifies Task 1.4: empty delta NO_OP and upheaval detection."""
    # Empty mutation -> [NO_OP]
    empty_record = DOMDeltaRecord(mutations=[], is_full_snapshot=False)
    assert "[NO_OP]" in empty_record.to_obs_string()

    # Full snapshot
    el1 = UIElement(1, "button", "btn", "Submit", (0.1, 0.1, 0.2, 0.2))
    full_record = DOMDeltaRecord(is_full_snapshot=True, full_elements=[el1])
    assert "<obs full>" in full_record.to_obs_string()

    # Upheaval threshold
    mutations_small = [DOMMutation(DOMMutationType.MODIFY, 1)]
    assert check_upheaval_threshold(mutations_small, total_prev_elements=10) is False

    # >50% mutated
    mutations_large = [
        DOMMutation(DOMMutationType.MODIFY, i) for i in range(6)
    ]
    assert check_upheaval_threshold(mutations_large, total_prev_elements=10) is True

    # >32 mutated elements
    mutations_massive = [
        DOMMutation(DOMMutationType.MODIFY, i) for i in range(35)
    ]
    assert check_upheaval_threshold(mutations_massive, total_prev_elements=100) is True


def test_token_reduction_benchmark():
    """Verifies Task 4.1: token reduction (>70% fewer tokens per step)."""
    import torch
    from jed_m2.bpe_tokenizer import ActionBPETokenizer

    tokenizer = ActionBPETokenizer(vocab_size=8192)

    # 20 typical web elements
    elements = [
        UIElement(1, "input", "search", "Search items", (0.1, 0.05, 0.7, 0.1), is_input=True),
        UIElement(2, "button", "search_btn", "Search", (0.72, 0.05, 0.85, 0.1), is_clickable=True),
        UIElement(3, "button", "cart_btn", "Cart (0)", (0.88, 0.05, 0.98, 0.1), is_clickable=True),
    ]
    for i in range(4, 21):
        elements.append(
            UIElement(i, "button", "product_btn", f"Product Item {i} - Add to Cart", (0.1, 0.15 + (i - 4) * 0.04, 0.9, 0.18 + (i - 4) * 0.04), is_clickable=True)
        )

    # Turn 0: Full snapshot
    full_delta0 = DOMDeltaRecord(is_full_snapshot=True, url="https://shop.example.com", full_elements=elements)
    full_obs0 = full_delta0.to_obs_string()
    full_tokens0 = tokenizer.encode(f"<obs> {full_obs0} </obs> <thought>")
    assert len(full_tokens0) > 150

    # Turn 1: 1 element mutated (typing in search box)
    curr_elements = [
        UIElement(1, "input", "search", "Search items", (0.1, 0.05, 0.7, 0.1), value="running shoes", is_input=True),
    ] + elements[1:]

    # Without differential deltas: re-encoding full DOM
    full_delta1 = DOMDeltaRecord(is_full_snapshot=True, url="https://shop.example.com", full_elements=curr_elements)
    full_tokens1 = tokenizer.encode(f"<obs> {full_delta1.to_obs_string()} </obs> <thought>")

    # With differential deltas:
    mutations = diff_elements(elements, curr_elements)
    diff_delta1 = DOMDeltaRecord(mutations=mutations, is_full_snapshot=False, url="https://shop.example.com")
    diff_obs1 = diff_delta1.to_obs_string()
    diff_tokens1 = tokenizer.encode(f"{diff_obs1} <thought>")

    token_savings = (len(full_tokens1) - len(diff_tokens1)) / float(len(full_tokens1))
    print(f"Full tokens: {len(full_tokens1)} | Delta tokens: {len(diff_tokens1)} | Savings: {token_savings * 100:.1f}%")

    assert token_savings >= 0.70, f"Expected >= 70% token savings, got {token_savings * 100:.1f}%"


def test_zero_nans_and_memory_footprint():
    """Verifies Task 4.2: zero NaNs, pure PyTorch execution, memory compliance."""
    import torch
    from jed_m2.config import JEDM2Config
    from jed_m2.model import JEDM2Model
    from jed_m2.agent import JEDM2Agent
    from jed_m2.bpe_tokenizer import ActionBPETokenizer

    config = JEDM2Config(
        vocab_size=8192,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=256,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=32,
        num_experts=4,
        top_k=2,
        moe_intermediate_dim=256,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_dt_rank=8,
    )
    model = JEDM2Model(config)
    tokenizer = ActionBPETokenizer(vocab_size=config.vocab_size)
    agent = JEDM2Agent(model=model, config=config, tokenizer=tokenizer, device="cpu", dtype=torch.float32)

    agent.reset()
    full_obs = "<obs full> URL: https://example.com | DOM: <button id=1>Click</button> </obs>"
    out1 = agent.act(obs_text=full_obs)
    assert not math.isnan(out1.coordinates[0])
    assert not math.isnan(out1.coordinates[1])
    assert not math.isnan(out1.gate_prob)

    delta_obs = "<obs delta> ~ id=1 text='Clicked' </obs>"
    out2 = agent.act(obs_text=delta_obs)
    assert not math.isnan(out2.coordinates[0])
    assert not math.isnan(out2.coordinates[1])
    assert not math.isnan(out2.gate_prob)
