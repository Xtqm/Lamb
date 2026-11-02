# Proposal: Invertible Byte-Level BPE Tokenizer for JED-M2

## Why

JED-M2 currently relies on `SimpleLAMTokenizer`, which maps words to token IDs using a lossy, one-way MD5 hash function (`hash(w) % 8181`), causing decoded tokens to emerge as non-invertible `"tok_<id>"` strings. This prevents the model from generating real text arguments during interactive execution (achieving only 1.45% exact argument match on held-out web traces) and forces the runtime scaffold to artificially intercept and substitute search queries from goal prompts. Replacing this hash mechanism with an invertible Byte-Level BPE tokenizer with guaranteed round-trip reconstruction ($\text{decode}(\text{encode}(s)) \equiv s$) is essential for native, open-ended web form typing and autonomous agent execution.

## What Changes

- **Invertible Byte-Level BPE Tokenizer (`ActionBPETokenizer`)**: Implement a deterministic, byte-level BPE tokenizer with a vocabulary budget of 8,192 tokens that guarantees 100% bijective round-trip text reconstruction without out-of-vocabulary (OOV) tokens.
- **Reserved Special Tokens**: Maintain exact alignment of special control tokens in IDs `[0..10]` (`<pad>`, `<bos>`, `<eos>`, `<obs>`, `</obs>`, `<thought>`, `</thought>`, `<action>`, `</action>`, `<arg>`, `</arg>`), plus optional action primitive tokens.
- **Pretrained Web/DOM Vocabulary**: Train/calibrate the 8,192 BPE vocabulary on representative Mind2Web and WebArena DOM trees, element attributes, user intents, and form argument strings.
- **Embedding Table Compatibility**: Retain the existing model embedding matrix shape (`[8192, 512]`), ensuring seamless architectural compatibility across Nano and Medium configurations.
- **Scaffold Macro-Action Decoupling**: Update `ExecutionScaffold` and runtime agents to decode genuine argument strings directly from the model's generated token stream rather than falling back to regex goal extraction.
- **Dataset and Pipeline Migration**: Update `SimpleLAMTokenizer` references across `dataset.py`, `real_dataset.py`, `rl_pipeline.py`, `train.py`, and `train_rl.py` to use `ActionBPETokenizer`.

## Capabilities

### New Capabilities
- `action-tokenization`: Defines bidirectional, invertible byte-level subword tokenization for web/OS action models, guaranteeing exact round-trip string reconstruction ($\text{decode}(\text{encode}(s)) \equiv s$) across arbitrary UTF-8 text and reserved control token parsing.

### Modified Capabilities
- `real-web-trajectory-dataset`: Update sequence tokenization requirement to mandate invertible subword encoding/decoding for task intents, DOM representations, and argument payloads instead of lossy hashing.

## Impact

- **Affected Code**: `jed_m2/dataset.py`, `jed_m2/real_dataset.py`, `jed_m2/agent.py`, `jed_m2/rl_pipeline.py`, `jed_m2/scaffold.py`, `train.py`, `train_rl.py`, `eval_benchmarks.py`.
- **Dependencies**: Uses the standard, installed `tokenizers` library with pure-Python fallback for offline reproducibility.
- **Model Checkpoints**: Model weights for `tok_embeddings` and `lm_head` will be initialized/adapted to the new BPE vocabulary indices.
