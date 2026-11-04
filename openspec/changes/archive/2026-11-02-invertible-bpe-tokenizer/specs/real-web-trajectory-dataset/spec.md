# Spec Delta: real-web-trajectory-dataset

## MODIFIED Requirements

### Requirement: Sequence Tokenization and Intent Representation
The system SHALL tokenize task intent strings, pruned accessibility DOM representations, and reasoning thoughts into standard sequence tensors matching model vocabulary configurations using an invertible byte-level subword tokenizer that supports exact string reconstruction for argument payloads.

#### Scenario: Multimodal sequence tokenization
- **WHEN** task intent, DOM element descriptions, and thoughts are passed to the tokenizer
- **THEN** standard token ID sequences are emitted with dedicated delimiters for observations, thoughts, actions, and arguments

#### Scenario: Invertible argument sequence extraction
- **WHEN** action argument payloads from real web trajectories are tokenized and subsequently decoded from model predictions
- **THEN** the decoded argument string reconstructs the original target text rather than opaque hash tokens
