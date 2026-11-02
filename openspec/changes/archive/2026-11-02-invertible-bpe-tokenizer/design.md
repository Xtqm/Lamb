# Design: Invertible Byte-Level BPE Tokenizer for JED-M2

## Context

JED-M2's current tokenizer (`SimpleLAMTokenizer` in `jed_m2/dataset.py`) uses an unkeyed MD5 hash modulo 8,181 to map words to token IDs. Because MD5 is a one-way compression function, token IDs cannot be mapped back to words; decoding simply emits `"tok_<id>"`. As identified in the scientific audit, this prevents the model from generating natural text arguments during interactive browsing (1.45% held-out argument match) and forced `ExecutionScaffold` to artificially inject search queries from goal prompts.

The system requires an invertible, deterministic tokenizer with a fixed vocabulary budget of 8,192 tokens that guarantees 100% round-trip string reconstruction without out-of-vocabulary tokens.

## Goals / Non-Goals

**Goals:**
- Provide `ActionBPETokenizer` with exact round-trip string reconstruction ($\text{decode}(\text{encode}(s)) \equiv s$) for arbitrary UTF-8 text.
- Preserve the exact 8,192 vocabulary budget, matching JED-M2's embedding layer `[8192, 512]` across Nano and Medium models.
- Reserve control tokens at fixed indices `[0..10]` (`<pad>`, `<bos>`, `<eos>`, `<obs>`, `</obs>`, `<thought>`, `</thought>`, `<action>`, `</action>`, `<arg>`, `</arg>`), ensuring backward-compatible loss masking and sequence delimitation.
- Provide a bundled, pre-trained web vocabulary asset (`jed_m2/assets/bpe_vocab_8192.json`) trained on representative Mind2Web and WebArena DOM trees, user intents, and form arguments.
- Include a fast Hugging Face `tokenizers` backend with a pure-Python fallback for standalone environments.
- Decouple `_resolve_argument_macro()` in `ExecutionScaffold` so the model's decoded text directly executes browser typing actions.

**Non-Goals:**
- Expanding vocabulary beyond 8,192 tokens (which would alter model parameter count).
- Modifying vision patch tokenization (handled separately by `VisionPatchEncoder`).
- Changing discrete action heads or coordinate regression heads.

## Decisions

### Decision 1: Byte-Level BPE with Byte-Fallback
- **Choice:** Byte-level Byte-Pair Encoding (BPE) starting from 256 individual byte tokens + 11 special tokens, followed by 7,925 subword merges.
- **Rationale:** Byte-level BPE guarantees that *any* arbitrary string (Unicode, special characters, URLs, code snippets) can be tokenized into valid token IDs without ever emitting `<unk>` or failing on unseen words.
- **Alternatives Considered:**
  - *WordPiece / SentencePiece with `<unk>`:* Rejected because encountering an unknown character would emit `<unk>`, breaking form typing.
  - *Char-level tokenizer:* Rejected because sequence lengths would explode by $4\times$, exceeding the 1024 sequence budget.

### Decision 2: Fixed Reserved Control Token Alignment
- **Choice:** Reserve IDs `[0..10]` strictly for special delimiters:
  - `0: <pad>`, `1: <bos>`, `2: <eos>`, `3: <obs>`, `4: </obs>`, `5: <thought>`, `6: </thought>`, `7: <action>`, `8: </action>`, `9: <arg>`, `10: </arg>`.
  - Base byte tokens mapped to `[11..266]`, merges mapped to `[267..8191]`.
- **Rationale:** Keeps collators, loss masks, and sequence parsing completely intact.

### Decision 3: Pre-Trained Domain Vocabulary Asset
- **Choice:** Train the BPE merge table on the corpus of task goals, pruned accessibility DOM trees, and real-world form arguments from Mind2Web and WebArena, and serialize to `jed_m2/assets/bpe_vocab_8192.json`.
- **Rationale:** Ensures optimal subword compression for HTML tags, attributes, and common web terminology (e.g. `button`, `input`, `headphones`, `search`, `checkout`) within the compact 8k budget.

### Decision 4: Fast Backend with Pure-Python Engine Fallback
- **Choice:** `ActionBPETokenizer` uses `tokenizers.Tokenizer` for microsecond-level C-speed tokenization when installed, while embedding a pure-Python BPE merge loop as fallback.
- **Rationale:** Satisfies the strict repository rule: 100% native PyTorch/Python fallback without mandatory C++ compilation, while leveraging `tokenizers` for maximum speed.

## Risks / Trade-offs

- **[Risk] Pretrained Checkpoint Embedding Misalignment:** Weights in existing checkpoints (`pretrain_unified`, `grpo_step_100`) were trained on the old hash vocabulary.
  - *Mitigation:* Retain `SimpleLAMTokenizer` as a legacy class so older checkpoints remain evaluable. When fine-tuning or continuing RL with `ActionBPETokenizer`, re-initialize or adapt token embeddings, or train an adaptation phase.
- **[Risk] Sequence Length Variation on Long Form Arguments:**
  - *Mitigation:* Subword merges compress typical English words into 1-2 tokens. Padded collators truncate gracefully at `max_seq_len` (1024).

## Migration Plan

1. Create `jed_m2/assets/bpe_vocab_8192.json` and `jed_m2/bpe_tokenizer.py`.
2. Expose `ActionBPETokenizer` in `jed_m2/dataset.py` and alias it as the primary tokenizer.
3. Update `RealTrajectoryCollator` and `AgentTrajectoryDataset` to use `ActionBPETokenizer`.
4. Update `ExecutionScaffold._resolve_argument_macro()` to trust native decoded text when valid.
5. Add comprehensive unit tests verifying round-trip reconstruction, special token preservation, and argument decoding.
