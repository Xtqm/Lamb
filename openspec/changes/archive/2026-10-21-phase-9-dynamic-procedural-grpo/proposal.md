# Proposal

## Why

Training Large Action Models with Group Relative Policy Optimization (GRPO) on fixed, static simulated environments leads to severe macro memorization, policy entropy collapse, and spatial coordinate drift. When environments present static layouts and unchanging element coordinates, the policy memorizes hardcoded $(x, y)$ regression coordinates and canned action token sequences instead of learning generalized visual grounding, DOM traversal, and robust planning. Furthermore, treating coordinate regression purely as an auxiliary supervised loss during reinforcement learning decouples spatial accuracy from the policy gradient objective.

In addition, critical architectural gaps exist in the current foundation:
1. **Disconnected World Model**: In `jed_m2/model.py`, JEPA relies on a decoupled external observation input (`obs_t`) rather than deriving its latent state directly from the contextual backbone representations $h_t$, preventing the JEPA predictive loss ($\mathcal{L}_{\text{JEPA}}$) from training the recurrent backbone (Mamba SSM, MLA, MoE) and token embeddings.
2. **Suboptimal MLA KV-Caching**: In `jed_m2/mla.py`, autoregressive generation decompresses full multi-head keys and values $(K, V)$ into the KV-cache (1024 floats/token) rather than caching only the low-rank latent representation $c_{KV} \in \mathbb{R}^{B, 1, d_c}$ and decoupled RoPE key $K_{\text{rope}} \in \mathbb{R}^{B, 1, d_R}$ (160 floats/token) and absorbing $W_{UK}$ into $W_Q$.
3. **Loss Gradient Imbalance**: In `jed_m2/losses.py`, token cross-entropy on thoughts and arguments dominates coordinate Huber loss and primitive classification by orders of magnitude, causing unstable multi-task convergence.

Phase 9 resolves these bottlenecks by first applying Priority 1 architectural fixes to the backbone representation pipeline, MLA caching, and multi-task loss calibration, then introducing an unbounded procedural environment synthesis engine coupled with a joint discrete-continuous GRPO policy gradient formulation.

## What Changes

- **Priority 1 Architectural Fixes**:
  - **Connect JEPA to Backbone Representations (`jed_m2/model.py`)**: Remove separate external `obs_t` input for JEPA. Route the contextual backbone hidden state $h_t$ (from `h_text` or pooled visual patch tokens) into `self.jepa`, ensuring predictive consistency loss $\mathcal{L}_{\text{JEPA}}$ backpropagates directly into the token embeddings and recurrent blocks (Mamba SSM, MLA, MoE).
  - **True MLA Latent Caching & Weight Absorption (`jed_m2/mla.py`)**: Update `MultiHeadLatentAttention.step()` to store only the compressed latent tensor $c_{KV} \in \mathbb{R}^{B, 1, d_c}$ ($d_c=128$) and shared decoupled RoPE key $K_{\text{rope}} \in \mathbb{R}^{B, 1, d_R}$ ($d_R=32$) in the KV-cache (160 floats/token, slashing cache memory by 84.4%). Absorb $W_{UK}$ into $W_Q$ during autoregressive generation to eliminate multi-head key decompression while maintaining exact numerical parity.
  - **Calibrate Composite Multi-Task Loss Weights (`jed_m2/losses.py`)**: Properly normalize token-level reasoning and argument cross-entropy losses and rebalance coefficients (`coord_weight`, `reason_weight`, `action_type_weight`, `arg_weight`) so that no single loss head dominates parameter gradients.
- **Procedural Scenario Generation (`jed_m2/scenario_generator.py`)**:
  - Implement `DynamicScenarioGenerator` and 4 procedural interactive sandbox environments:
    - `DynamicEcommerce`: Procedural catalogs (10-50 items, dynamic pricing/discounts), multi-filter queries, category trees, coupon validation, checkout validation.
    - `DynamicFormPortal`: Multi-step enterprise forms with heterogeneous inputs (text, numbers, dates, dropdowns, radio toggles), client-side regex validators, and dynamic error banners.
    - `DynamicCloudAdmin`: Hierarchical infrastructure resource trees (VPCs, subnets, instances, security groups), status toggles, deletion confirmation modals, and table pagination.
    - `DynamicWebBrowser`: Multi-page directed navigation graphs with hyperlinks, back/forward history stacks, adversarial popup modals, cookie consent overlays, and dynamic simulated network delays.
  - Implement layout randomization, bounding box jitter noise, and distractor UI elements to eliminate coordinate memorization.
- **Joint Token + Continuous Coordinate GRPO (`jed_m2/rl.py`, `jed_m2/rl_pipeline.py`)**:
  - Formulate Gaussian spatial policy for `CoordinateHead`, computing continuous 2D spatial log-probabilities $\log \pi_{\text{coord}}(c \mid \mu_\theta, \sigma)$.
  - Extend the GRPO surrogate objective to a joint token-coordinate importance ratio:
    $$r_{i,t} = \exp\left((\log \pi_{\text{token}} - \log \pi_{\text{old\_token}}) + \lambda_{\text{coord}} (\log \pi_{\text{coord}} - \log \pi_{\text{old\_coord}})\right)$$
  - Implement dense multi-factor reward shaping in `jed_m2/rl.py`: task completion bonus (+2.0), sub-goal milestone rewards (+0.5), coordinate IoU hit rewards ($+0.3 \times \text{IoU}$), step efficiency penalty (-0.05/turn), and anti-looping penalties (-0.5 for repeated action-coordinate pairs).
  - Update `TrajectoryRolloutSampler` to record coordinate log-probabilities and integrate dynamic procedural environments.
- **Curriculum Learning & Exploration Scheduling (`train_rl.py`)**:
  - Add `--curriculum` flag supporting dynamic progression across Level 1 (single-step atomic actions), Level 2 (2-3 step workflows), and Level 3 (5-8 step long-horizon workflows) based on a moving window success rate threshold (> 80%).
  - Add adaptive entropy regularization and temperature scheduling to maintain policy entropy and prevent premature mode collapse.
- **Verification Harness (`verify_phase9.py`)**:
  - Standalone verification script testing JEPA backbone gradient flow, MLA latent caching numerical equivalence, loss weight calibration, procedural scenario diversity (50+ seeds without crashes), joint token-coordinate gradient flow, curriculum progression, reward shaping sanity, and strict < 6.0 GB VRAM ceiling compliance.

## Capabilities

### New Capabilities
- `procedural-scenario-generator`: Procedural generation of dynamic, randomized interactive Web/OS task environments (`DynamicEcommerce`, `DynamicFormPortal`, `DynamicCloudAdmin`, `DynamicWebBrowser`) with layout jitter, bounding box noise, and distractor elements.

### Modified Capabilities
- `unified-lam-backbone`: Support true MLA latent cache compression ($c_{KV}, K_{\text{rope}}$) and $W_{UK}$ matrix absorption during `step()`, and pass contextual backbone representations $h_t$ directly to JEPA.
- `jepa-world-model`: Connect online state encoder directly to contextual backbone hidden states and pooled visual tokens, backpropagating predictive loss into recurrent blocks and embeddings.
- `composite-loss-harness`: Calibrate and normalize composite multi-task loss components to balance gradient contributions across reasoning, primitive classification, and coordinate regression heads.
- `trajectory-policy-optimization`: Upgrade GRPO surrogate loss to joint token + continuous coordinate policy optimization using Gaussian spatial log-probabilities, joint importance ratios, and dense multi-factor reward shaping.
- `rl-rollout-pipeline`: Upgrade rollout sampling to capture continuous spatial log-probabilities, sample from procedural environment generators, execute curriculum progression, and schedule policy exploration.

## Impact

- **Code Base**:
  - New module: `jed_m2/scenario_generator.py`.
  - Upgraded modules: `jed_m2/model.py`, `jed_m2/mla.py`, `jed_m2/losses.py`, `jed_m2/rl.py`, `jed_m2/rl_pipeline.py`, `train_rl.py`, `jed_m2/__init__.py`.
  - New verification suite: `verify_phase9.py`.
- **APIs & Data Structures**:
  - `JEDM2Model.forward`: Routes contextual hidden states $h_t$ to `self.jepa` without requiring disconnected `obs_t`.
  - `MultiHeadLatentAttention.step`: KV-cache stores `(c_kv, k_rope)` of dimension $(d_c, d_R)$ instead of multi-head $(K, V)$.
  - `JEDM2LossConfig`: Calibrated weights balancing reasoning cross-entropy and coordinate Huber regression.
  - `GRPOConfig`: Added `lambda_coord`, `coord_sigma`, `entropy_coeff`.
  - `RolloutTurn`: Added `coord_logp` and `subgoal_completed` fields.
  - `calculate_grpo_loss`: Accepts coordinate logps and old coordinate logps.
  - `compute_shaped_reward`: New function computing dense multi-factor trajectory rewards.
- **Hardware & VRAM**:
  - MLA KV cache size during generation reduced by 84.4% (from 1024 floats/token to 160 floats/token).
  - Maintains strict memory footprint under the 6.0 GB GDDR7 ceiling on NVIDIA GeForce RTX 5060 Laptop GPU in native `torch.bfloat16`.
- **Dependencies**:
  - 100% pure PyTorch and Python standard library (no external C++/CUDA packages).
