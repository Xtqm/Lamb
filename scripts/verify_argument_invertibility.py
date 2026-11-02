"""
Verification of Invertible BPE Argument Reconstruction on Real Mind2Web Test Turns.
Compares LegacyHashTokenizer against ActionBPETokenizer on exact string reconstruction.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jed_m2.bpe_tokenizer import ActionBPETokenizer
from jed_m2.config import JEDM2Config
from jed_m2.dataset import LegacyHashTokenizer
from jed_m2.real_dataset import RealWebTrajectoryDataset


def main():
    print("Loading RealWebTrajectoryDataset (split='test')...")
    config = JEDM2Config()
    test_ds = RealWebTrajectoryDataset(config=config, split="test")

    # Extract all non-empty ground truth argument strings from test split
    test_args = []
    for sample in test_ds:
        for turn in sample.turns:
            arg = (turn.argument or "").strip()
            if arg:
                test_args.append(arg)

    print(f"Extracted {len(test_args)} non-empty argument strings from held-out test split.")
    assert len(test_args) > 0, "No arguments found in test dataset!"

    legacy_tok = LegacyHashTokenizer(vocab_size=8192)
    bpe_tok = ActionBPETokenizer(vocab_size=8192)

    # 1. Evaluate LegacyHashTokenizer
    legacy_exact_matches = 0
    for arg in test_args:
        tokens = legacy_tok.encode(arg)
        decoded = legacy_tok.decode(tokens)
        if decoded == arg:
            legacy_exact_matches += 1

    legacy_match_rate = (legacy_exact_matches / len(test_args)) * 100.0

    # 2. Evaluate ActionBPETokenizer
    bpe_exact_matches = 0
    bpe_arg_token_matches = 0
    for arg in test_args:
        # Test standard encode / decode
        tokens = bpe_tok.encode(arg)
        decoded = bpe_tok.decode(tokens)
        if decoded == arg:
            bpe_exact_matches += 1

        # Test delimited <arg> arg </arg> decode_argument
        delimited_str = f"<arg> {arg} </arg>"
        delimited_tokens = bpe_tok.encode(delimited_str)
        decoded_arg = bpe_tok.decode_argument(delimited_tokens)
        if decoded_arg == arg:
            bpe_arg_token_matches += 1

    bpe_match_rate = (bpe_exact_matches / len(test_args)) * 100.0
    bpe_arg_rate = (bpe_arg_token_matches / len(test_args)) * 100.0

    print("=" * 70)
    print("HELD-OUT REAL MIND2WEB TEST SET: ARGUMENT INVERTIBILITY RESULTS")
    print("=" * 70)
    print(f"Total Evaluated Test Arguments: {len(test_args)}")
    print(f"Legacy Hash Tokenizer Exact Match: {legacy_exact_matches}/{len(test_args)} ({legacy_match_rate:.2f}%)")
    print(f"ActionBPETokenizer Exact Match:    {bpe_exact_matches}/{len(test_args)} ({bpe_match_rate:.2f}%)")
    print(f"ActionBPETokenizer Arg Delimited:  {bpe_arg_token_matches}/{len(test_args)} ({bpe_arg_rate:.2f}%)")
    print("=" * 70)

    # Sample comparisons
    print("\nRepresentative Sample Reconstructions:")
    for sample_arg in test_args[:5]:
        legacy_dec = legacy_tok.decode(legacy_tok.encode(sample_arg))
        bpe_dec = bpe_tok.decode(bpe_tok.encode(sample_arg))
        print(f"  Input:    '{sample_arg}'")
        print(f"  Legacy:   '{legacy_dec}'")
        print(f"  BPE:      '{bpe_dec}'")
        print("  " + "-" * 50)

    assert bpe_match_rate == 100.0, f"Expected 100% exact match for BPE, got {bpe_match_rate:.2f}%"
    print("\nSUCCESS: 1.45% non-invertibility barrier is completely resolved (100.00% exact match)!")


if __name__ == "__main__":
    main()
