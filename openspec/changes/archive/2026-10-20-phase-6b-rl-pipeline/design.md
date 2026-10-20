# Design

## Context

In Phase 5, theoretical loss formulations and trainer scaffolding for Direct Policy Optimization (`DPOTrainer`) and Group Relative Policy Optimization (`GRPOTrainer`) were implemented in `jed_m2/rl.py`. In Phase 6A, the interactive agent runtime (`JEDM2Agent`) and simulated environment sandbox (`WebOSEnvironment`) were introduced, enabling $O(1)$ constant-memory autoregressive rollouts.

Phase 6B connects these modules into an end-to-end interactive reinforcement learning pipeline. Because the active policy and frozen reference policy must coexist on an 8GB GDDR7 RTX 5060 Laptop GPU, the memory footprint of dual models (~51M parameters each = ~204MB total in BF16), optimizer states (~180MB), activations, and rollout buffers must remain strictly below the 6.0 GB (6144 MB) operational VRAM ceiling.

## Goals / Non-Goals

**Goals:**
- Provide `TrajectoryRolloutSampler` in `jed_m2/rl_pipeline.py` that executes stochastic multi-turn rollouts in `WebOSEnvironment` using temperature-scaled sampling.
- Support group rollout sampling ($G$ rollouts per prompt) and group-normalized relative advantage estimation for GRPO.
- Support preference pair generation (winning vs losing trajectories) for offline DPO preference optimization.
- Provide tensor serialization converting multi-turn interaction rollouts into tokenized training tensors (`input_ids`, `targets`, `mask`, `old_token_logps`, `rewards`).
- Implement `train_rl.py` providing a configurable CLI for online GRPO and offline DPO policy training with live ASCII telemetry logging.
- Implement `verify_phase6b.py` validating rollout sampling, preference pairing, GRPO optimization, 25-iteration reward improvement, and VRAM ceiling compliance (< 6.0 GB).

**Non-Goals:**
- Large-scale multi-epoch production pretraining / SFT convergence (reserved for Phase 6C).
- Vision / multimodal pixel observation processing (reserved for Phase 6D).
- External distributed multi-node rollout orchestration (single-node local execution is sufficient for local hardware).

## Decisions

### 1. Stochastic Exploration Sampling in Agent Rollouts
- **Decision**: Extend or wrap `JEDM2Agent` to support stochastic sampling with temperature $T$ during rollouts. When $T > 0$, the System 2 reasoning loop samples next tokens from $\text{Categorical}(\text{logits} / T)$ rather than greedy $\text{argmax}$. The System 1 action primitive is sampled from $\text{Categorical}(\text{logits} / T)$, and spatial coordinates have optional Gaussian exploration perturbation bounded to $[0, 1]$.
- **Rationale**: Policy gradient algorithms like GRPO require behavioral diversity across the $G$ rollouts within a group to discover rewarding pathways and compute non-trivial variance for group advantage normalization.
- **Alternatives Considered**: Epsilon-greedy random action selection (discards semantic model distributions; produces non-informative gibberish actions).

### 2. Rollout Trajectory Representation and Tensor Serialization
- **Decision**: Define a lightweight `RolloutTrajectory` dataclass that captures:
  - Sequence tokens: Observation prompts, generated thought tokens, special separator tokens, action representations.
  - Per-token log-probabilities under the rollout policy ($\pi_{\text{old}}$).
  - Step rewards and terminal goal outcome reward ($R \in \{0.0, 1.0\}$).
  - Mask indicating which tokens correspond to trainable policy actions/thoughts.
  A serialization method converts a group of $G$ trajectories into batched tensors `(input_ids, targets, mask, old_token_logps, rewards)` padded to the maximum sequence length of the group, ready for `GRPOTrainer.compute_loss`.
- **Rationale**: Keeps rollout collection detached from gradient graphs during environment interaction, conserving GPU memory, then executes forward/backward passes on compact padded batches.

### 3. Dual-Model Memory Architecture (Active + Frozen Reference)
- **Decision**: Both the active `JEDM2Model` and frozen `ref_model` are instantiated in `torch.bfloat16`. The reference model parameters have `requires_grad = False` and are executed strictly inside `torch.no_grad()` blocks. Activation checkpointing is enabled on the active policy during policy gradient computation.
- **Rationale**: Two 51M parameter models in BF16 consume only ~204 MB of parameter VRAM. Activation checkpointing keeps backpropagation activations under 1.5 GB even with sequence lengths up to 2048, leaving > 4 GB headroom well below the 6.0 GB ceiling.

### 4. Reward Shaping & Advantage Normalization
- **Decision**: For GRPO, trajectory outcome rewards are computed from ground-truth task completion: $R_{\text{outcome}} = 1.0$ if the goal is satisfied, plus step-efficiency rewards and penalties from `WebOSEnvironment`. Within each group of $G$ rollouts, advantages are normalized: $A_i = (R_i - \mu) / (\sigma + \epsilon)$. If $\sigma = 0$ (all rollouts achieved identical reward), $A_i = 0$ to prevent spurious gradient spikes.
- **Rationale**: Matches DeepSeekMath / GRPO formulation; provides stable variance-reduced policy gradient signals.

## Risks / Trade-offs

- **[Risk] High Variance / Degenerate Groups in Online GRPO**: If all $G$ rollouts in a group fail or succeed, group variance $\sigma = 0$, producing zero advantages and stalled learning.
  - *Mitigation*: Ensure temperature $T \approx 0.7 - 1.0$ promotes sufficient action diversity. Add small step-level reward signals (e.g. valid click/type transitions) to differentiate partial progress.
- **[Risk] Peak VRAM Exceeding 6.0 GB during Simultaneous Rollouts and Gradients**: Creating large rollout batches could exhaust GPU memory.
  - *Mitigation*: Rollouts are executed in inference mode (`torch.no_grad()`) sequentially or with small micro-batches. Intermediate recurrent states are discarded after rollout completion.
- **[Risk] Reference Policy Drift & KL Divergence Explosion**: If policy updates take large unconstrained steps, the active policy may diverge from the reference distribution.
  - *Mitigation*: PPO-style surrogate objective clipping ($\epsilon=0.2$) combined with Schulman token-level KL penalty ($\beta_{\text{kl}}=0.04$) tightly regulates policy drift.
