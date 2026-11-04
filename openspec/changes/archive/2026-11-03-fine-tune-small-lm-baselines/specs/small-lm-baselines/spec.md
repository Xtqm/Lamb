# Spec Delta: small-lm-baselines

## MODIFIED Requirements

### Requirement: T5-Small Sequence-to-Sequence Baseline Adaptor
The system SHALL provide an encoder-decoder baseline adaptor based on Google T5-Small (~60M parameters) fine-tuned with teacher forcing and cross-entropy sequence loss on identical Mind2Web manifests, with support for validation loss tracking, optimizer checkpointing, and saving fine-tuned weights for benchmark evaluation.

#### Scenario: T5-Small forward loss computation
- **WHEN** paired input prompt tokens and target action tokens are provided
- **THEN** the model computes sequence cross-entropy loss with padding tokens masked from loss computation

#### Scenario: T5-Small action generation
- **WHEN** an input prompt is passed during evaluation
- **THEN** the model decodes autoregressively using greedy or temperature-controlled beam search and outputs a valid action string

#### Scenario: T5-Small fine-tuning and checkpoint saving
- **WHEN** the baseline training runner executes fine-tuning on the Mind2Web training split
- **THEN** optimization steps update model parameters via AdamW, report step loss, and save the fine-tuned model and tokenizer to a checkpoint directory

### Requirement: DistilGPT-2 Causal LM Baseline Adaptor
The system SHALL provide a causal decoder baseline adaptor based on Hugging Face DistilGPT-2 (~82M parameters) fine-tuned under causal next-token cross-entropy loss, with support for validation loss tracking, optimizer checkpointing, and saving fine-tuned weights for benchmark evaluation.

#### Scenario: DistilGPT-2 training pass
- **WHEN** concatenated prompt and action token sequences are fed with prompt tokens masked from loss
- **THEN** the model computes next-token prediction loss strictly on the action and argument token targets

#### Scenario: DistilGPT-2 inference generation
- **WHEN** prompt tokens are passed to the model
- **THEN** the model continues generation until an end-of-action delimiter is reached or max generation tokens are emitted

#### Scenario: DistilGPT-2 fine-tuning and checkpoint saving
- **WHEN** the baseline training runner executes fine-tuning on the Mind2Web training split
- **THEN** optimization steps update causal LM parameters via AdamW, report step loss, and save the fine-tuned model and tokenizer to a checkpoint directory
