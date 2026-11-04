# Spec Delta

## ADDED Requirements

### Requirement: Dual Multimodal Observation Ingestion
The system SHALL ingest dual multimodal observations combining visual screenshot patch tokens and discrete DOM/text token sequences into a unified sequence representation.

#### Scenario: Sequence forward pass with visual patch tokens
- **WHEN** visual inputs (`images` of shape `[B, 3, H, W]` or `visual_tokens` `[B, N_v, D]`) and text token sequences `[B, S]` are provided to `JEDM2Model`
- **THEN** the model fuses visual tokens with text tokens via early-fusion prefixing and processes the joint sequence through all recurrent blocks

### Requirement: Visually Grounded Action Decoding in LAM Backbone
The system SHALL forward visual patch representations alongside hidden states to the action decoding heads to visually ground spatial coordinate regressions.

#### Scenario: Grounded action prediction in forward pass
- **WHEN** multimodal observations are evaluated by `JEDM2Model`
- **THEN** the model passes visual patch tokens to the action decoding head, yielding visually grounded coordinate outputs

### Requirement: Multimodal Recurrent Step Rollout Mode
The system SHALL support recurrent single-step rollout with visual observations, updating recurrent states and decoding actions with visual grounding.

#### Scenario: Recurrent step with visual observation
- **WHEN** `step(...)` is called with a token input and visual patch context
- **THEN** the model updates internal states and returns current step rationale and visually grounded action predictions
