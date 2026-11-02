# Tasks: Invertible Byte-Level BPE Tokenizer for JED-M2

## 1. BPE Vocabulary Asset Generation

- [x] 1.1 Create `scripts/train_bpe_vocab.py` to fit a Byte-Level BPE model on Mind2Web / WebArena task goals, DOM trees, and argument strings with an 8,192-token budget and reserved special tokens `[0..10]`
- [x] 1.2 Generate and save `jed_m2/assets/bpe_vocab_8192.json`, and verify the vocabulary file has exactly 8,192 entries and maps control tokens `<pad>` through `</arg>` to IDs 0 through 10

## 2. Core `ActionBPETokenizer` Implementation

- [x] 2.1 Implement `ActionBPETokenizer` in `jed_m2/bpe_tokenizer.py` with fast `tokenizers.Tokenizer` backend and pure-Python BPE merge engine fallback
- [x] 2.2 Create `tests/test_bpe_tokenizer.py` and verify 100% exact round-trip reconstruction ($\text{decode}(\text{encode}(s)) \equiv s$) across alphanumeric text, punctuation, URLs, and Unicode characters
- [x] 2.3 Add tests verifying control token preservation (`<obs>`, `</obs>`, `<thought>`, `</thought>`, `<action>`, `</action>`, `<arg>`, `</arg>`) and argument payload string decoding

## 3. Pipeline & Runtime Integration

- [x] 3.1 Expose `ActionBPETokenizer` in `jed_m2/dataset.py`, updating `SyntheticTrajectoryGenerator` and `AgentTrajectoryDataset` to use it by default while maintaining `SimpleLAMTokenizer` as a legacy alias
- [x] 3.2 Update `RealTrajectoryCollator` in `jed_m2/real_dataset.py` to use `ActionBPETokenizer`
- [x] 3.3 Update `ExecutionScaffold` in `jed_m2/scaffold.py` to dispatch native decoded argument strings directly to browser inputs when valid text is present
- [x] 3.4 Update `rl_pipeline.py` and `agent.py` to decode action arguments through `ActionBPETokenizer`

## 4. End-to-End Verification & Test Suite

- [x] 4.1 Run the complete test suite (`pytest tests/`) and verify all unit tests pass with zero regressions
- [x] 4.2 Verify round-trip argument decoding on real Mind2Web test turns, confirming the 1.45% non-invertibility barrier is resolved
