# Design: Ground GRPO Rollouts on Diverse Real Web Tasks

## Context

In initial online GRPO training, exploration rollouts were gathered using `PlaywrightEnvironment`, which cycled through 3 local static HTML mock fixtures (`ecommerce.html`, `form_portal.html`, `travel.html`). Because the policy interacted exclusively with these 3 fixed layouts, the continuous spatial coordinate heads and discrete action primitive heads overfit to the fixed button coordinates of those three templates. On the held-out real web evaluation benchmark, this caused coordinate MAE to degrade from 0.3203 to 0.3335.

Meanwhile, the repository has ingested 1,009 genuine Mind2Web task traces (with full cleaned/raw HTML DOM trees, candidate interactive elements, and multi-turn action targets) and 812 WebArena task definitions. By transitioning the GRPO rollout sampler from static local fixtures to dynamic offline real-world DOM snapshots with programmatic ground-truth element-matching oracles, the model can explore across thousands of realistic web layouts without overfitting to synthetic button coordinates.

## Goals / Non-Goals

**Goals:**
- Implement `RealTaskDOMEnvironment` and refactor `TrajectoryRolloutSampler` to sample rollout episodes from the 1,009 Mind2Web and 812 WebArena tasks.
- For each sampled task, load the authentic HTML DOM tree, parse interactive candidate element bounding boxes, and formulate the initial agent observation with `Goal: {intent}`.
- Implement programmatic multi-attribute matching oracles: compare the agent's predicted action type, target element (tag, ARIA role, textual label/placeholder), and spatial coordinates (IoU >= 0.5, 2D Gaussian proximity) against ground-truth step annotations.
- Maintain group rollout generation ($G=4$) on identical initial task states to compute group-relative normalized advantages for policy gradient updates.
- Keep sequence length at 1024 tokens and peak VRAM strictly below 6.0 GB on the NVIDIA GeForce RTX 5060 Laptop GPU.

**Non-Goals:**
- Spinning up multi-container Docker servers (WebArena live servers): these require heavy CPU/RAM/VRAM overhead and introduce non-deterministic state drift. High-speed DOM snapshot replay provides identical layout fidelity deterministically.
- Unconstrained live internet scraping during online training: all rollout states must remain self-contained, reproducible, and sandboxed offline.

## Decisions

### 1. In-Memory DOM Snapshot Replay vs Live Container Sandboxes
- **Decision**: Ingest DOM trees directly from Mind2Web and WebArena JSON traces into an in-memory DOM simulation and candidate element parser (`FastDOMParser`). Optionally inject into headless Chromium when full JavaScript rendering is requested.
- **Rationale**: An in-memory DOM replay executes in < 5 milliseconds per step compared to 500-1500 ms per live browser navigation. Over 4,000 rollout steps with group size $G=4$, this reduces rollout generation time from hours to minutes while retaining authentic real-world element trees.
- **Alternatives considered**: Running live WebArena docker containers (rejected due to 8GB system RAM limits and networking complexity).

### 2. Multi-Attribute Dense Oracle Reward Formulation
- **Decision**: Formulate step reward as a composite metric:
  $$R_{\text{step}} = R_{\text{action\_type}} + R_{\text{element\_match}} + R_{\text{spatial\_iou}} + R_{\text{coord\_dist}} + R_{\text{argument}}$$
  - Action type match: +1.0 if predicted action matches target action type (e.g., CLICK, TYPE).
  - Element match: +1.5 if chosen element matches ground-truth element ID, ARIA role, or text.
  - Spatial IoU: $+2.0 \times \text{IoU}(\text{pred\_box}, \text{target\_box})$.
  - Coordinate distance: $+1.0 \times \exp(-0.5 \cdot \|\mathbf{c}_{\text{pred}} - \mathbf{c}_{\text{gt}}\|^2 / \sigma^2)$.
  - Argument text: +1.5 for normalized string match on TYPE inputs.
  - Off-target penalty: -0.5 for clicks outside interactive elements.
- **Rationale**: Dense intermediate rewards provide gradient signal to both discrete token heads (action selection, argument generation) and continuous spatial coordinate heads ($x, y$), preventing policy divergence.
- **Alternatives considered**: Sparse terminal-only binary reward (rejected: exploration space on real DOMs with 64+ elements is too large for binary reward without massive sample sizes).

### 3. Group Rollout Batching and Normalized Relative Advantage
- **Decision**: For each online GRPO step, select a mini-batch of $B$ unique real-world tasks. For each task, sample $G=4$ stochastic exploration trajectories with temperature $T=0.7$. Normalize advantages within each group:
  $$A_i = \frac{R_i - \mu_G}{\sigma_G + \epsilon}$$
- **Rationale**: Preserves the GRPO paradigm without requiring an auxiliary critic/value network, saving ~1.5 GB VRAM.
- **Alternatives considered**: PPO with value network (rejected: requires extra memory and critic calibration).

## Risks / Trade-offs

- **[Risk]** Large HTML DOM sizes exceeding sequence length (1024 tokens)
  - **Mitigation**: Use `FastDOMParser` with `max_elements=48` and DOM subtree pruning (stripping scripts, styles, metadata) so the observation comfortably fits within the 1024-token budget.
- **[Risk]** Coordinate jitter causing low spatial IoU on very small buttons
  - **Mitigation**: Use Gaussian proximity scoring alongside rectangular IoU so that clicks near small buttons receive smooth partial credit.
- **[Risk]** VRAM spikes during group rollout generation
  - **Mitigation**: Generate rollouts under `torch.no_grad()` in micro-batches of 2, collate the trajectory batches onto CPU memory, and stream to GPU only during the policy update pass with gradient checkpointing enabled.
