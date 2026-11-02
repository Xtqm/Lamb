"""
Unit tests for JED-M2 ActionBPETokenizer.
Verifies:
1. Exact round-trip string reconstruction: decode(encode(s)) == s.
2. Control token preservation and exact ID mapping in [0..10].
3. Native argument string extraction via decode_argument().
4. Robustness on URLs, punctuation, numbers, and Unicode strings.
"""

import pytest
from jed_m2.bpe_tokenizer import ActionBPETokenizer


@pytest.fixture
def tokenizer() -> ActionBPETokenizer:
    return ActionBPETokenizer(vocab_size=8192)


def test_special_token_id_mapping(tokenizer: ActionBPETokenizer):
    """Verifies that all 11 control delimiters are mapped to IDs 0 through 10."""
    expected_mapping = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<obs>": 3,
        "</obs>": 4,
        "<thought>": 5,
        "</thought>": 6,
        "<action>": 7,
        "</action>": 8,
        "<arg>": 9,
        "</arg>": 10,
    }
    for token, expected_id in expected_mapping.items():
        assert tokenizer.special_tokens[token] == expected_id
        assert tokenizer.inv_special_tokens[expected_id] == token

    # Check encode of control token
    enc = tokenizer.encode("<arg>")
    assert enc == [9]
    enc_end = tokenizer.encode("</arg>")
    assert enc_end == [10]


@pytest.mark.parametrize(
    "text",
    [
        "headphones",
        "Sony WH-1000XM5 Wireless Noise-Cancelling Headphones ($398.00)",
        "https://travel.booking.internal/hotels?city=san_francisco&rooms=1",
        "Check-in 3:00 PM, Check-out 11:00 AM. 100% Guaranteed!",
        "Special symbols: @#$%^&*()_+-=[]{}|;':,.<>/?`~",
        "Unicode characters: cafe resume naive 100€ 50£",
        "<action> TYPE <arg> headphones </arg> </action>",
        "<obs> button: Search [0.72, 0.08, 0.85, 0.14] </obs>",
        "<thought> I should book the Grand Bayview Hotel </thought>",
    ],
)
def test_exact_round_trip_reconstruction(tokenizer: ActionBPETokenizer, text: str):
    """Verifies 100% bijective round-trip reconstruction for diverse web strings."""
    token_ids = tokenizer.encode(text)
    assert len(token_ids) > 0
    # Every token ID must fall within [0, 8191]
    assert all(0 <= tid < 8192 for tid in token_ids)

    reconstructed = tokenizer.decode(token_ids, skip_special_tokens=False)
    assert reconstructed == text


def test_decode_argument_delimited(tokenizer: ActionBPETokenizer):
    """Verifies extraction and decoding of argument text enclosed in <arg>...</arg>."""
    full_action_text = "<action> TYPE <arg> wireless headphones </arg> </action>"
    tokens = tokenizer.encode(full_action_text)
    extracted_arg = tokenizer.decode_argument(tokens)
    assert extracted_arg == "wireless headphones"


def test_decode_argument_standalone(tokenizer: ActionBPETokenizer):
    """Verifies decoding when tokens are just the argument payload without delimiters."""
    arg_text = "San Francisco, CA"
    tokens = tokenizer.encode(arg_text)
    extracted_arg = tokenizer.decode_argument(tokens)
    assert extracted_arg == "San Francisco, CA"


def test_pure_python_fallback(tokenizer: ActionBPETokenizer):
    """Verifies that the tokenizer handles empty strings and boundary conditions safely."""
    assert tokenizer.encode("") == []
    assert tokenizer.decode([]) == ""
    assert tokenizer.decode_argument([]) == ""
