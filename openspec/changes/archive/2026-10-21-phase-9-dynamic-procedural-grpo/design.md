# Design

## Context

In Phases 6A, 6B, and 8, JED-M2 established simulated environments (`WebOSEnvironment`) with three static scenarios (`ecommerce`, `travel`, `os_admin`). While sufficient for basic rollout pipeline verification, training policy gradient algorithms like GRPO on fixed scenarios causes the model to memorize specific token sequences and hardcoded $(x, y)$ coordinate targets, resulting in rapid entropy collapse and spatial overfitting.

Furthermore, three foundational architectural shortcomings need addressing:
1. **Disconnected World Model**: In `jed_m2/model.py`, JEPA relies on a decoupled external observation input (`obs_t`) rather than deriving its latent state directly from the contextual backbone representations $h_t$, preventing the JEPA predictive loss ($\mathcal{L}_{\text{JEPA}}$) from training the recurrent backbone (Mamba SSM, MLA, MoE) and token embeddings.
2. **Suboptimal MLA KV-Caching**: In `jed_m2/mla.py`, autoregressive generation decompresses full multi-head keys and values $(K, V)$ into the KV-cache (1024 floats/token) rather than caching only the low-rank latent representation $c_{KV} \in \mathbb{R}^{B, 1, d_c}$ and decoupled RoPE key $K_{\text{rope}} \in \mathbb{R}^{B, 1, d_R}$ (160 floats/token) and absorbing $W_{UK}$ into $W_Q$.
3. **Loss Gradient Imbalance**: In `jed_m2/losses.py`, token cross-entropy on thoughts and arguments dominates coordinate Huber loss and primitive classification by orders of magnitude, causing unstable multi-task convergence.

This design introduces:
1. Priority 1 architectural fixes: Connecting JEPA representations to backbone states $h_t$, implementing True MLA latent caching with matrix absorption, and calibrating multi-task loss normalization.
2. `DynamicScenarioGenerator` in `jed_m2/scenario_generator.py` for unbounded procedural environment synthesis.
3. Continuous Gaussian spatial policy gradient formulation in `jed_m2/rl.py` and `jed_m2/rl_pipeline.py`.
4. Dense multi-factor reward shaping.
5. Curriculum progression and adaptive exploration scheduling in `train_rl.py`.

## Goals / Non-Goals

**Goals:**
- Pass contextual backbone hidden states $h_t$ from `h_text` directly to `self.jepa`, ensuring $\mathcal{L}_{\text{JEPA}}$ gradients flow into Mamba SSM, MLA, MoE, and embeddings.
- Store only compressed latent vectors $c_{KV} \in \mathbb{R}^{B, 1, d_c}$ and shared RoPE key $K_{\text{rope}} \in \mathbb{R}^{B, 1, d_R}$ in MLA `step()`, absorbing $W_{UK}$ into $W_Q$ during generation to slash cache memory by 84.4% while maintaining exact numerical attention equivalence.
- Calibrate composite multi-task loss weights and normalization so reasoning, primitive, coordinate Huber, and argument heads contribute balanced gradient magnitudes.
- Provide 4 distinct procedural environment types (`DynamicEcommerce`, `DynamicFormPortal`, `DynamicCloudAdmin`, `DynamicWebBrowser`) with rich interactive state machines and anti-memorization layout jitter.
- Formulate a 2D Gaussian spatial policy enabling GRPO to optimize both discrete token choices and continuous coordinate regression via joint policy gradient ratios.
- Implement dense multi-factor reward shaping (+2.0 completion, +0.5 milestones, $+0.3 \times \text{IoU}$, -0.05 step penalty, -0.5 loop penalty).
- Implement a 3-tier curriculum learning scheduler (Levels 1, 2, 3) advancing on $\ge 80\%$ success rate, with adaptive entropy regularization and temperature annealing.
- Strictly adhere to the 6.0 GB VRAM limit and 100% pure PyTorch build rule.

**Non-Goals:**
- Modifying the core Mamba SSM recurrence equations or MoE SwiGLU expert routing architecture.
- Integrating external gym environments or third-party RL frameworks (e.g., Stable-Baselines3, Ray/RLlib).
- Replacing Playwright live browser integration (Phase 7); procedural environments operate as high-throughput in-memory sandboxes.

## Decisions

### 1. Connecting JEPA to Contextual Backbone Representations (`jed_m2/model.py`)

**Decision**: In `JEDM2Model.forward()`, eliminate the external `obs_t` argument requirement. Derive current state representation $h_t$ directly from the contextual output of the stacked recurrent blocks (`h_text`).
- For sequence batches, $h_t$ is extracted at the action commitment token index or pooled across observation tokens: $h_t = \text{mean}_{t \in \text{obs}}(h_{\text{text}, t})$ or $h_t = h_{\text{text}}[:, -1, :]$.
- $h_t$ is passed directly to `self.jepa.encode(h_t, visual_tokens=visual_tokens_t)`.
- When $s_{t+1}$ targets are evaluated, target representations are computed via detached target encoder `self.jepa.encode_target(obs_next)`.
- The JEPA predictive loss $\mathcal{L}_{\text{JEPA}} = \text{SmoothL1}(\hat{z}_{t+1}, \text{sg}(z_{t+1})) + \lambda_{\text{cos}}(1 - \cos(\hat{z}_{t+1}, \text{sg}(z_{t+1})))$ backpropagates directly into $h_t \to \text{norm\_f} \to \text{JEDM2Blocks} \to \text{tok\_embeddings}$.

*Alternatives Considered*:
- *Passing raw token embeddings to JEPA*: Misses long-horizon recurrence and multi-layer context accumulated by Mamba and MLA.
- *Keeping external `obs_t`*: Disconnects the world model from the representation learner, failing to shape backbone features.

### 2. True MLA Latent Caching & Weight Absorption (`jed_m2/mla.py`)

**Decision**: In DeepSeek Multi-Head Latent Attention, keys without RoPE are generated by $K_{\text{nope}} = c_{KV} W_{UK}^T$.
The unnormalized attention score for head $h$ is:
$$S_h = Q_{\text{nope}, h} K_{\text{nope}, h}^T + Q_{\text{rope}, h} K_{\text{rope}}^T$$
Substituting $K_{\text{nope}, h} = c_{KV} W_{UK, h}^T$:
$$Q_{\text{nope}, h} K_{\text{nope}, h}^T = Q_{\text{nope}, h} (c_{KV} W_{UK, h}^T)^T = (Q_{\text{nope}, h} W_{UK, h}) c_{KV}^T = \tilde{Q}_h c_{KV}^T$$
where $\tilde{Q}_h = Q_{\text{nope}, h} W_{UK, h} \in \mathbb{R}^{B, 1, d_c}$.

Therefore, during autoregressive generation in `step()`:
1. $W_{UK}$ is absorbed into query projection: $\tilde{Q}_h = Q_{\text{nope}, h} W_{UK, h}$.
2. The KV-cache stores **only**:
   - $c_{KV} \in \mathbb{R}^{B, S_{\text{past}}, d_c}$ ($d_c = \text{kv\_lora\_rank} = 128$)
   - $K_{\text{rope}} \in \mathbb{R}^{B, S_{\text{past}}, d_R}$ ($d_R = \text{qk\_rope\_head\_dim} = 32$)
   Total: $128 + 32 = 160$ elements per token, compared to $2 \times N_h \times d_h = 2 \times 8 \times 64 = 1024$ elements in uncompressed multi-head caching (84.4% memory reduction).
3. Attention weights are computed by $A_h = \text{softmax}((\tilde{Q}_h c_{KV}^T + Q_{\text{rope}, h} K_{\text{rope}}^T) / \sqrt{d_k})$.
4. Head values are computed by $O_h = (A_h c_{KV}) W_{UV, h}^T \in \mathbb{R}^{B, 1, v\_head\_dim}$.

*Alternatives Considered*:
- *Decompressing $c_{KV}$ into $K_{\text{nope}}$ and $V$ at each step*: Wastes memory bandwidth and quadruples KV-cache size.

### 3. Composite Multi-Task Loss Balancing (`jed_m2/losses.py`)

**Decision**: Calibrate loss weighting and token reduction in `JEDM2CompositeLoss`.
- Reasoning cross-entropy and argument cross-entropy average across active token count, typically yielding values in $[2.0, 5.0]$.
- Spatial coordinate Smooth L1 / Huber regression on $[0, 1]$ coordinates typically produces values in $[0.01, 0.05]$.
- To balance gradient norms across parameter tensors:
  - `coord_weight` is set to $10.0$ (calibrated against Smooth L1 with $\beta=0.05$).
  - `reason_weight` is set to $0.5$.
  - `action_type_weight` is set to $2.0$.
  - `arg_weight` is set to $0.5$.
  - `jepa_weight` is set to $1.0$.
  - `aux_weight` is set to $1.0$.
This ensures that gradients from the spatial coordinate regressor and discrete action heads match the magnitude of reasoning token gradients, preventing the LM head from monopolizing backbone updates.

### 4. Procedural Environment Architecture (`jed_m2/scenario_generator.py`)

**Decision**: Implement `DynamicScenarioGenerator` yielding modular environment instances conforming to the standard `WebOSEnvironment` state machine and `StepResult` interface.
- `DynamicEcommerce`: Procedural catalogs with 10-50 randomized products, variable price ranges, dynamic category hierarchies, discount coupons (e.g. `PROMO10`, `SAVE20`), cart state management, and multi-field checkout.
- `DynamicFormPortal`: Multi-step form wizards (Personal -> Professional -> Review) featuring text, integer, date, dropdown (`select`), and radio toggle inputs, accompanied by regex validators and dynamic client-side error notifications.
- `DynamicCloudAdmin`: Hierarchical resource topologies (VPC -> Subnets -> Instances -> Security Groups), action modals (delete confirmation requiring confirmation string), status toggles, and paginated resource tables.
- `DynamicWebBrowser`: Directed navigation graphs (3-6 pages) with forward/backward history stacks, adversarial popup overlays (cookie consent banners, promotional popups) that intercept clicks until dismissed, and simulated network delays.
- Anti-memorization: Every reset applies random coordinate jitter $\Delta \in [-\delta, \delta]$ to element bounding boxes, randomly orders sibling DOM elements, and injects 3-8 distractor elements (e.g. unclickable spans, banner images, dummy links).

### 5. Joint Token + Continuous Coordinate GRPO Formulation (`jed_m2/rl.py`, `jed_m2/rl_pipeline.py`)

**Decision**: Treat spatial coordinates $c = (x, y) \in [0, 1]^2$ as a continuous action sampled from a bivariate Gaussian policy:
$$c \sim \mathcal{N}(\mu_\theta, \sigma^2 I), \quad \mu_\theta = \text{CoordinateHead}(h)$$
where $\sigma = \sigma_{\text{base}} \cdot T$ ($T$ is sampling temperature).
The continuous log-probability density is:
$$\log \pi_{\text{coord}}(c \mid \mu_\theta, \sigma) = -\sum_{k \in \{x, y\}} \left( \frac{(c_k - \mu_{\theta, k})^2}{2\sigma^2} + \ln(\sigma \sqrt{2\pi}) \right)$$

The joint importance sampling ratio for turn $t$ is defined as:
$$r_{i,t} = \exp\left((\log \pi_{\text{token}} - \log \pi_{\text{old\_token}}) + \lambda_{\text{coord}} (\log \pi_{\text{coord}} - \log \pi_{\text{old\_coord}})\right)$$
where $\lambda_{\text{coord}} = 0.5$ balances the gradient scale between token cross-entropy and coordinate continuous regression.

The clipped surrogate objective evaluates:
$$\mathcal{L}_{\text{surr}} = -\frac{1}{G}\sum_{i=1}^G \frac{1}{|T_i|}\sum_{t} \min\left(r_{i,t} A_i, \text{clip}(r_{i,t}, 1-\epsilon, 1+\epsilon) A_i\right)$$
coupled with reference policy token KL divergence and coordinate drift regularization:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{surr}} + \beta_{\text{kl}} D_{\text{KL}}(\pi_{\text{token}} \parallel \pi_{\text{ref\_token}}) + \beta_{\text{coord\_kl}} \frac{\|\mu_\theta - \mu_{\text{ref}}\|^2}{2\sigma^2}$$

### 6. Dense Multi-Factor Reward Shaping (`jed_m2/rl.py`)

**Decision**: Implement `compute_shaped_reward(...)` aggregating 5 distinct signals:
1. Task Completion Bonus: $+2.0$ for satisfying terminal goal conditions.
2. Sub-Goal Milestone Rewards: $+0.5$ per verified intermediate progression state.
3. Coordinate IoU Hit Reward: $+0.3 \times \text{IoU}(c, \text{bbox}_{\text{target}})$ measuring spatial alignment with target bounding boxes.
4. Step Efficiency Penalty: $-0.05$ per turn taken to encourage minimal path solutions.
5. Anti-Looping Penalty: $-0.5$ when identical action primitives and near-identical coordinates are repeated consecutively without state progress.

### 7. Curriculum Learning & Adaptive Exploration Scheduler (`train_rl.py`)

**Decision**: Implement automated 3-tier curriculum progression:
- **Level 1 (Atomic Tasks)**: 1-step targets (e.g., click a specific button, fill a single labeled input).
- **Level 2 (Compound Workflows)**: 2-3 step workflows (e.g., search product + add to cart, fill 2-step form).
- **Level 3 (Long-Horizon Tasks)**: 5-8 step workflows (e.g., complete checkout, multi-page graph navigation with adversarial popups).
Progression occurs automatically when the rolling window success rate over the last 10 episodes exceeds $80\%$.

Adaptive entropy regularization monitors discrete action entropy $H(\pi) = -\sum p \log p$ and applies an auxiliary entropy bonus $-\alpha_{\text{ent}} H(\pi)$ whenever entropy falls below a critical threshold $H_{\min}$, while annealing temperature $T$ from $0.8$ down to $0.2$.

## Risks / Trade-offs

- **[Risk] Continuous Gaussian log-probability scale mismatch**: Gaussian log-density values can be large negative numbers, overwhelming token log-probs in $r_{i,t}$.
  - *Mitigation*: Calibrate $\lambda_{\text{coord}} = 0.2 - 0.5$ and clamp joint log-ratio deltas to $[-5.0, 5.0]$ before exponentiation.
- **[Risk] High VRAM usage with multiple dynamic environments**: Loading dynamic element trees and tensors across group rollouts could exceed memory budget.
  - *Mitigation*: Run rollouts in lightweight inference mode (`torch.no_grad()`, `step(...)` recurrent buffers), collate tensors with pinned micro-batch sizes, and enforce CUDA cache clearing between iterations.
- **[Risk] Non-stationary reward distribution under dense shaping**: Negative step penalties and positive bonuses could induce reward hacking.
  - *Mitigation*: Ensure terminal task completion (+2.0) strictly dominates cumulative sub-goal bonuses, and cap maximum step rewards per turn.
