# Design

## Context

The JED-M2 Large Action Model uses dual-system action decoding: System 2 generates latent rationale tokens, and System 1 emits discrete primitive logits, continuous $(x, y) \in [0, 1]^2$ pointer coordinates, and text argument logits. In web navigation, HTML interactive elements occupy discrete spatial bounding boxes. While continuous regression places predictions near target elements (MAE $\approx 0.26$), sub-pixel variance causes unconstrained clicks to land in element padding or whitespace, yielding low raw hit rates ($0.41\%$). Furthermore, default configs omitted `SELECT` (causing 54 Mind2Web test actions to be discarded as unscorable), argument token loss was damped by $\ln(8192)$ normalization, and agent rollout only decoded single-token arguments.

## Goals / Non-Goals

**Goals:**
- Provide candidate DOM element bounding-box snapping in `jed_m2/agent.py` during `act()` and `step_environment()`, with optional distance gating $\delta_{\max}$.
- Preserve both `raw_coordinates` and `coordinates` (snapped) in `AgentStepOutput` for full observability.
- Enable dual-mode offline evaluation in `jed_m2/real_eval.py` to report both raw continuous and candidate-snapped metrics simultaneously.
- Set `num_action_types: int = 7` as default in `JEDM2Config` and add zero-padding backward compatibility for legacy 6-class checkpoints.
- Rebalance `arg_weight` in `JEDM2LossConfig` to ensure text typing gradients match action and coordinate gradient magnitudes.
- Implement autoregressive multi-token argument generation for `TYPE` actions in `JEDM2Agent` and `rl_pipeline.py`.

**Non-Goals:**
- Retraining model checkpoints or modifying model parameter dimensions.
- Changing DOM parser structure or altering Mind2Web dataset cache format.

## Decisions

### 1. Candidate Bounding Box Snapping Algorithm
- **Approach**: For predicted coordinates $\hat{c} = (\hat{x}, \hat{y})$ and candidate elements $\mathcal{E} = \{e_i \mid e_i.\text{has\_bbox}\}$:
  $$e^* = \arg\min_{e_i \in \mathcal{E}} \|\hat{c} - \text{center}(e_i)\|_2$$
  $$\hat{c}_{\text{snapped}} = \begin{cases} \text{center}(e^*), & \text{if } \|\hat{c} - \text{center}(e^*)\|_2 \le \delta_{\max} \\ \hat{c}, & \text{otherwise} \end{cases}$$
- **Rationale**: Center-distance minimization is $O(K)$ in candidate elements ($\sim 30-80$ elements per page), adding sub-millisecond overhead. The distance gate $\delta_{\max} = 0.20$ prevents off-target clicks from snapping across the screen.
- **Alternatives Considered**:
  - *Unconditional snapping*: Snaps every click even if aimed at empty background. Rejected because canvas or background dismiss clicks should not snap to distant buttons.
  - *Bounding box edge projection*: Projecting to the closest point on the box perimeter. Rejected because hitting element centers maximizes activation reliability in browser click event dispatchers.

### 2. Dual-Mode Evaluation Reporting in `real_eval.py`
- **Approach**: In `score_offline_windows()`, maintain two sets of metric accumulators (or a configurable evaluation flag `snap_to_candidates: bool = True`) that log both `raw` and `snapped` metrics in parallel.
- **Rationale**: Paper reviewers and engineers need both metrics: raw MAE verifies backbone spatial convergence, while snapped metrics verify end-to-end task grounding.

### 3. Default 7 Action Primitives and Checkpoint Padding
- **Approach**: Set `num_action_types: int = 7` in `JEDM2Config`. In `JEDM2Model.load_state_dict()` or checkpoint loading helpers, detect when `state_dict['action_head.action_type_head.fc2.weight']` has 6 rows and zero-pad row 6 to shape `[7, 256]` (and bias to `[7]`).
- **Rationale**: Eliminates unscorable `SELECT` actions on real datasets while preventing `RuntimeError: size mismatch` when evaluating legacy 6-class checkpoints.

### 4. Loss Weight Rebalancing for Text Arguments
- **Approach**: Increase `arg_weight` in `JEDM2LossConfig` from `2.0` to `8.0` (or bypass $\ln(8192)$ scaling for argument cross-entropy).
- **Rationale**: With $\ln(8192) \approx 9.01$ normalization, effective argument gradient weight was $\approx 0.22$, which was 45x weaker than coordinate loss ($10.0$). Increasing `arg_weight` to `8.0` brings the effective weight to $\approx 0.89$, aligning with discrete action type loss ($2.0$) and preventing garbled search queries.

### 5. Autoregressive Multi-Token Argument Generation
- **Approach**: In `JEDM2Agent.act()`, when `action_type == ActionType.TYPE`, unroll argument generation by feeding `<arg>` into `model.step()`, greedily decoding argument token IDs until emitting `</arg>` or reaching `max_arg_tokens = 16`.
- **Rationale**: Replaces single-token truncation (`decode([arg_token_id])`) with full query string emission (e.g., "running shoes").

## Risks / Trade-offs

- **[Risk: Viewport with No Interactive Elements]** → Mitigation: If candidate list is empty or lacks bounding boxes, snapping gracefully defaults to $\hat{c}$.
- **[Risk: Autoregressive Argument Decoding Latency]** → Mitigation: Only triggers for `TYPE` actions and is strictly capped at 16 tokens ($< 15\text{ ms}$ on GPU).
- **[Risk: Legacy Checkpoint Incompatibility]** → Mitigation: Explicit tensor padding guardrail in checkpoint loader ensures zero regressions across legacy checkpoints.
