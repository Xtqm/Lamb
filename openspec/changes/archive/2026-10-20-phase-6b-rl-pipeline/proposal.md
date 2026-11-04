# Proposal

## Why

In Phase 5, theoretical loss formulations and trainer scaffolding for Direct Policy Optimization (DPO) and Group Relative Policy Optimization (GRPO) were established. In Phase 6A, the interactive agent runtime (`JEDM2Agent`) and simulated sandbox (`WebOSEnvironment`) were created, enabling closed-loop rollouts.

However, the policy optimization algorithms in `jed_m2/rl.py` are currently disconnected from live environment execution. The agent cannot autonomously sample diverse exploration rollouts, evaluate group-relative advantages, or optimize its policy parameters based on digital task success rewards. Implementing Phase 6B establishes an interactive RL pipeline (`jed_m2/rl_pipeline.py`) and command-line training harness (`train_rl.py`) that bridges live environment rollouts with online GRPO and offline DPO policy updates, while strictly adhering to the 6.0 GB VRAM budget on the RTX 5060 Laptop GPU.

## What Changes

1. **Implement `jed_m2/rl_pipeline.py`**:
   - `RolloutTrajectory`: Data container capturing prompt tokens, generated thought/action tokens, action types, coordinates, arguments, log-probabilities, step rewards, terminal reward, and cumulative success flag.
   - `TrajectoryRolloutSampler`: Execution harness that wraps `JEDM2Agent` and `WebOSEnvironment` to run interactive rollouts with stochastic temperature-scaled token and action sampling.
   - Group Rollout Collection: Generates groups of $G$ rollouts per instruction scenario for Group Relative Policy Optimization (GRPO) with outcome reward scoring.
   - Preference Dataset Pairing: Pairs winning ($R_{\text{term}}=1.0$) and losing ($R_{\text{term}}=0.0$) trajectories for Direct Policy Optimization (DPO).
   - Trajectory Tensor Serialization: Converts multi-turn interaction rollouts into tokenized batches (`input_ids`, `targets`, `loss_masks`, `old_token_logps`) compatible with `GRPOTrainer` and `DPOTrainer`.

2. **Implement `train_rl.py` CLI Script**:
   - Production CLI interface supporting both `--mode grpo` (online interactive policy optimization) and `--mode dpo` (offline preference fine-tuning).
   - Online GRPO training loop: samples groups of $G$ rollouts across benchmark environments, normalizes advantages within each group, and computes clipped surrogate policy gradient updates with reference policy KL divergence regularization.
   - Offline DPO training loop: samples or loads paired preference trajectories and optimizes the implicit reward margin between winning and losing trajectories.
   - Live telemetry and metrics logging: reporting step reward mean, advantage variance, reference KL divergence, policy entropy, gradient norm, and token throughput.
   - Checkpoint persistence for RL-optimized policy weights.

3. **Public API Exposure**:
   - Export new pipeline constructs (`TrajectoryRolloutSampler`, `RolloutTrajectory`, etc.) in `jed_m2/__init__.py`.

4. **Phase 6B Verification Suite (`verify_phase6b.py`)**:
   - Verify multi-group stochastic rollout generation and ground-truth task reward assignment.
   - Verify GRPO policy gradient updates and Schulman KL constraint adherence against the frozen reference model.
   - Verify measurable reward improvement across 25 RL training iterations on simulated environment tasks.
   - Verify VRAM ceiling compliance ensuring active policy, frozen reference policy, rollout buffers, and optimizer states remain strictly within 6.0 GB (6144 MB) on CUDA in `bfloat16`.

## Capabilities

### New Capabilities
- `rl-rollout-pipeline`: Connects DPO and GRPO policy optimization algorithms to live interactive rollouts in simulated environments, managing stochastic group sampling, preference pair generation, advantage normalization, and RL policy gradient training within the 6.0 GB VRAM ceiling.

### Modified Capabilities
<!-- None: existing trajectory-policy-optimization specification remains satisfied and unchanged. -->

## Impact

- **Codebase Modules**: Adds `jed_m2/rl_pipeline.py`, `train_rl.py`, and `verify_phase6b.py`; updates `jed_m2/__init__.py`.
- **Dependencies & Hardware**: Pure PyTorch; relies on existing `JEDM2Agent`, `WebOSEnvironment`, `DPOTrainer`, and `GRPOTrainer`. Maintains dual models (active policy + frozen reference model) in `bfloat16` within the 6.0 GB VRAM ceiling.
- **Breaking Changes**: None. All changes are strictly additive and preserve existing SFT and inference APIs.
