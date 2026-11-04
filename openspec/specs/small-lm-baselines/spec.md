# small-lm-baselines Specification

## Purpose
Provides pretrained small language model baselines (T5-Small and DistilGPT-2) for text-to-action browser grounding, prompt serialization, sequence training, and action decoding on identical dataset splits.

## Requirements

### Requirement: Text-to-Action Sequence Serialization and Parsing
The system SHALL serialize multi-turn browser agent observations, natural language goals, and candidate DOM elements into text sequences, and SHALL parse generated sequence tokens into discrete action primitives, continuous normalized coordinates, and text arguments.

#### Scenario: Serializing an interaction turn
- **WHEN** an interaction turn containing goal, DOM candidates, and action history is processed
- **THEN** the system produces an input prompt string containing delimited goal, history, and candidate elements with bounding box coordinates

#### Scenario: Parsing generated model response
- **WHEN** a baseline language model generates an output string
- **THEN** the system extracts the rationale, discrete ActionType, normalized coordinate pair (x, y) in [0, 1], and optional text argument, returning a fallback failure only if the syntax is unparseable

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

### Requirement: Hardware and Parameter Accounting Conformance
The system SHALL track total parameters, active parameters per token, memory allocation curves, and per-step inference latency in milliseconds for each baseline under native PyTorch bfloat16 without exceeding 6.0 GB VRAM.

#### Scenario: Memory and latency profiling
- **WHEN** baseline inference is evaluated on CUDA
- **THEN** peak memory footprint remains below 6144 MB and inference latency per step is measured after warmup
