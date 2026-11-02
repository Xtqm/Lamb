"""
Train Byte-Level BPE Vocabulary for JED-M2 Large Action Model.
Vocabulary budget: Exactly 8,192 tokens.
Reserved special tokens in [0..10]:
  0: <pad>, 1: <bos>, 2: <eos>, 3: <obs>, 4: </obs>,
  5: <thought>, 6: </thought>, 7: <action>, 8: </action>,
  9: <arg>, 10: </arg>

Trained on:
1. Real Mind2Web task intents, DOM accessibility trees, and argument payloads.
2. WebArena and WebOS simulated navigation scenarios and HTML structures.
3. Common web control and browser interaction subwords.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Generator, List

# Ensure repo root is on sys.path for unpickling RealWebTrajectorySample
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<obs>",
    "</obs>",
    "<thought>",
    "</thought>",
    "<action>",
    "</action>",
    "<arg>",
    "</arg>",
]


def collect_training_texts(
    cache_path: str = "data/real_trajectories_cache/real_web_real_v4_8ef0c1b7844dc5d8_train_h512.pt",
) -> List[str]:
    """Collects text samples from cached trajectories and common web corpus."""
    texts: List[str] = []

    # 1. Base web navigation and domain terminology
    base_phrases = [
        "Search for noise-cancelling headphones and add top result to cart",
        "Book hotel in San Francisco for upcoming weekend",
        "Update security group rules in cloud administration console",
        "Select departure date, choose first available nonstop flight, and proceed to passenger info",
        "Find blue denim jeans size 32, filter by price under $50, add to shopping bag and checkout",
        "Navigate to settings, click security, edit inbound rules, save changes",
        "Enter username and password, click sign in, submit verification code",
        "Click search button, scroll down page, select item from dropdown, submit form",
        "Sony WH-1000XM5 Wireless Noise-Cancelling Headphones ($398.00)",
        "Grand Bayview Hotel SF - $220/night, check-in 3:00 PM, check-out 11:00 AM",
        "Default Web Tier Inbound Rules sg-01a2b port 443 protocol TCP",
        "<obs> button: Search [0.72, 0.08, 0.85, 0.14] </obs>",
        "<obs> input: search_bar 'Sony WH-1000XM5' [0.20, 0.08, 0.70, 0.14] </obs>",
        "<obs> a: first_product_link [0.10, 0.15, 0.90, 0.45] </obs>",
        "<obs> button: checkout_btn 'Add to Cart' [0.10, 0.35, 0.40, 0.44] </obs>",
        "<thought> I need to search for headphones and select the top product result </thought>",
        "<action> TYPE <arg> headphones </arg> </action>",
        "<action> CLICK </action>",
        "<action> SELECT <arg> Option 1 </arg> </action>",
        "<action> FINISH </action>",
    ]
    texts.extend(base_phrases)

    # 2. Extract from cached real web trajectories if available
    if os.path.exists(cache_path):
        print(f"Loading real web training trajectories from {cache_path}...")
        try:
            samples = torch.load(cache_path, map_location="cpu", weights_only=False)
            if isinstance(samples, list):
                print(f"Extracted {len(samples)} real trajectory samples.")
                for sample in samples:
                    goal = getattr(sample, "goal", "")
                    if goal:
                        texts.append(goal)
                    turns = getattr(sample, "turns", [])
                    for turn in turns:
                        obs = getattr(turn, "obs_text", "")
                        if obs:
                            # Add chunks of observation text
                            texts.append(obs[:1000])
                        arg = getattr(turn, "argument", "")
                        if arg:
                            texts.append(arg)
                            texts.append(f"<arg> {arg} </arg>")
                        thought = getattr(turn, "thought", "")
                        if thought:
                            texts.append(thought)
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_path}: {e}")

    # 3. Add synthetic WebArena / WebOS variations
    elements = [
        "input", "button", "select", "textarea", "a", "div", "span", "table", "tr", "td",
        "form", "header", "footer", "nav", "main", "section", "article", "label",
        "submit", "reset", "search", "checkout", "cart", "add to cart", "buy now",
        "continue", "back", "next", "previous", "filter", "sort by", "price",
        "high to low", "low to high", "rating", "reviews", "in stock", "out of stock",
        "sign in", "sign up", "log in", "log out", "register", "password", "email",
        "first name", "last name", "address", "city", "state", "zip code", "phone",
        "credit card", "expiration date", "cvv", "place order", "confirm order",
    ]
    for el in elements:
        texts.append(f"<{el}>")
        texts.append(f"role={el}")
        texts.append(f"click {el}")
        texts.append(f"type into {el}")

    print(f"Total training text passages collected: {len(texts)}")
    return texts


def train_action_bpe_tokenizer(
    output_path: str = "jed_m2/assets/bpe_vocab_8192.json",
    vocab_size: int = 8192,
) -> Tokenizer:
    """Trains an 8,192 Byte-Level BPE model with reserved control tokens [0..10]."""
    texts = collect_training_texts()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=1,
    )

    print(f"Training Byte-Level BPE model (target vocab size: {vocab_size})...")
    tokenizer.train_from_iterator(texts, trainer)

    actual_size = tokenizer.get_vocab_size()
    print(f"Vocabulary training complete. Actual vocabulary size: {actual_size}")

    # Verify special tokens alignment
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(token)
        assert actual_id == expected_id, (
            f"Special token '{token}' mapped to ID {actual_id}, expected {expected_id}!"
        )
        print(f"  Verified special token {token:<12} -> ID {actual_id}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    print(f"Saved trained BPE tokenizer to: {output_path}")

    # Verify round-trip reconstruction on test string
    test_strings = [
        "Search for noise-cancelling headphones",
        "<action> TYPE <arg> headphones </arg> </action>",
        "https://console.cloud.internal/vpc/security-groups?id=123",
        "Special symbols: 100% $25 #50 (item_123)",
    ]
    for s in test_strings:
        enc = tokenizer.encode(s)
        dec = tokenizer.decode(enc.ids, skip_special_tokens=False)
        assert dec == s, f"Round-trip mismatch!\nExpected: {repr(s)}\nActual:   {repr(dec)}"
        safe_str = s.encode("ascii", "backslashreplace").decode("ascii")
        print(f"  Verified round-trip: {safe_str[:40]!r} (tokens: {len(enc.ids)})")

    return tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train JED-M2 ActionBPETokenizer")
    parser.add_argument(
        "--output",
        type=str,
        default="jed_m2/assets/bpe_vocab_8192.json",
        help="Destination path for tokenizer.json asset",
    )
    parser.add_argument("--vocab_size", type=int, default=8192, help="Target vocabulary budget")
    args = parser.parse_args()

    train_action_bpe_tokenizer(output_path=args.output, vocab_size=args.vocab_size)
