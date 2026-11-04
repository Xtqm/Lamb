# Spec Delta: interactive-agent-runtime

## MODIFIED Requirements

### Requirement: Decoupled Dual-System Decision Loop
The agent runtime SHALL execute a decoupled dual-system decision loop that generates latent thought tokens (System 2) until the ActionGate logit exceeds the commitment threshold, then emits the discrete action primitive (optionally refined via JEPA latent rollout planning), continuous $(x, y) \in [0, 1]$ screen coordinates, and argument payloads (System 1).

#### Scenario: System 2 reasoning commits to System 1 action
- **WHEN** the agent processes a new environment turn observation
- **THEN** it generates reasoning thought tokens until the action gate fires or maximum thought length is reached, returning both the rationale and structured action

#### Scenario: Thought generation guardrail
- **WHEN** the action gate does not exceed threshold within a configurable maximum reasoning budget
- **THEN** the runtime forcefully triggers action commitment to prevent execution deadlock

#### Scenario: Latent planning integration within decision loop
- **WHEN** active JEPA planning is enabled and thought generation concludes
- **THEN** the decision loop runs latent action rollouts to verify or re-rank candidate action primitives before dispatching the chosen action to the environment

#### Scenario: Autonomous termination action commitment
- **WHEN** System 1 emits an `ActionType.FINISH` primitive upon completing the task objective
- **THEN** the agent runtime records the terminal turn, signals session completion, and yields execution without waiting for external scaffold or referee intervention
