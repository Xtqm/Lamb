# Spec Delta: action-tokenization

## Purpose

Provides deterministic, byte-level Byte-Pair Encoding (BPE) tokenization for web and OS action agents, guaranteeing lossless round-trip string reconstruction without out-of-vocabulary tokens across arbitrary UTF-8 text and reserved control token boundaries.

## ADDED Requirements

### Requirement: Lossless Invertible Tokenization
The system SHALL provide an invertible Byte-Level BPE tokenizer (`ActionBPETokenizer`) with a fixed vocabulary budget (default 8,192 tokens) that guarantees exact round-trip string reconstruction without out-of-vocabulary (OOV) tokens.

#### Scenario: Round-trip text reconstruction
- **WHEN** any arbitrary valid UTF-8 string including alphanumeric characters, whitespace, punctuation, and web URLs is encoded into token IDs and immediately decoded
- **THEN** the decoded output string matches the input string exactly ($\text{decode}(\text{encode}(s)) \equiv s$)

#### Scenario: Subword tokenization within vocabulary budget
- **WHEN** text sequences are tokenized
- **THEN** all emitted token IDs strictly reside in the range $[0, V - 1]$ where $V$ is the configured vocabulary size (default 8,192)

### Requirement: Reserved Special Control Tokens
The system SHALL reserve deterministic control token IDs for interaction boundaries at fixed indices in $[0..10]$: `<pad>` (0), `<bos>` (1), `<eos>` (2), `<obs>` (3), `</obs>` (4), `<thought>` (5), `</thought>` (6), `<action>` (7), `</action>` (8), `<arg>` (9), and `</arg>` (10).

#### Scenario: Special token encoding and preservation
- **WHEN** text containing explicit control tokens such as `<arg> headphones </arg>` is encoded
- **THEN** the control delimiters are parsed into their exact designated token IDs and are preserved distinctly from subword text tokens

#### Scenario: Special token decoding
- **WHEN** a token sequence containing IDs in $[0..10]$ is decoded
- **THEN** the corresponding control delimiter string is rendered without replacement or truncation

### Requirement: Native Text Argument Decoding
The system SHALL decode action argument token sequences into natural language strings for typing and form entry, eliminating non-invertible token placeholders.

#### Scenario: Native argument extraction
- **WHEN** the model generates an argument token payload enclosed in `<arg>` and `</arg>` delimiters
- **THEN** the tokenizer decodes the payload into the exact natural language string for direct dispatch to browser input elements without synthetic token identifiers
