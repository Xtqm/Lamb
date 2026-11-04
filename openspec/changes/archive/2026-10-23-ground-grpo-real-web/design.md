# Design: Ground JED-M2 Reinforcement Learning in Real-World Web Data

## Context

JED-M2 utilizes a hybrid dual-system architecture combining:
1. **System 2 (Autoregressive Latent Reasoning)**: Mamba SSM + MLA transformer generating rationale tokens between `<thought>` and `</thought>`.
2. **System 1 (Action Primitives & Spatial Grounding)**: Parallel projection heads emitting discrete action types (`CLICK`, `TYPE`, `SCROLL`, `NAVIGATE`, `FINISH`), continuous Gaussian pointer coordinates $(x, y) \in [0, 1]^2$, and argument tokens.

While pretraining is grounded on real-world HTML documents and Mind2Web annotations (`jed_m2/real_dataset.py`), the reinforcement learning harness (`train_rl.py`, `jed_m2/rl_pipeline.py`) currently executes rollouts exclusively within `WebOSEnvironment` (`jed_m2/environment.py`), which models procedural dictionary-based mocks. This introduces a severe representation gap between pretraining and RL.

To bridge this gap, this design specifies:
- **Track A**: Offline Direct Policy Optimization (DPO) on real Mind2Web traces with spatial and primitive perturbations.
- **Track B**: A headless Chromium interactive browser environment (`PlaywrightWebEnvironment`) matching the `WebOSEnvironment` contract with deterministic programmatic oracle verifiers.
- **Track C**: Dual-modality Group Relative Policy Optimization (GRPO) updating both discrete token heads and continuous Gaussian spatial coordinates on interactive browser rollouts.

## Goals / Non-Goals

**Goals:**
- Provide `jed_m2/real_preference_dataset.py` with `RealWebPreferenceDataset` and `RealPreferenceCollator` to construct paired preference trajectories $(\tau_w, \tau_l)$ from authentic Mind2Web traces.
- Connect `train_rl.py --mode dpo --dataset real_mind2web` to train policy weights against frozen reference models using `calculate_dpo_loss()`.
- Implement `jed_m2/playwright_env.py` with `PlaywrightWebEnvironment` matching the `WebOSEnvironment` interface (`reset()`, `step(action_type, coordinates, argument)`), local HTML fixture support, and deterministic programmatic oracle verification.
- Update `TrajectoryRolloutSampler` in `jed_m2/rl_pipeline.py` and `train_rl.py --mode grpo` to collect rollout groups against browser sandboxes and execute dual-modality GRPO updates (`lambda_coord`).
- Stay strictly within 6.0 GB peak VRAM on NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7) using bfloat16 and gradient checkpointing.
- Enforce 100% native PyTorch and standard library compliance (zero compiled C++/CUDA/Triton binaries).
- Implement pure 7-bit ASCII logging for Windows console compatibility.
- Provide comprehensive test coverage in `tests/test_real_rl.py`.

**Non-Goals:**
- Modifying the underlying model backbone dimensions or tokenizer vocabulary.
- Scraping or querying live public internet websites during training (avoiding external flakiness, rate limits, or network dependencies).
- Implementing asynchronous multi-process distributed rollout actors (focusing on rock-solid synchronous local execution under 6.0 GB VRAM).

## Decisions

### 1. Offline Preference Dataset Architecture (`jed_m2/real_preference_dataset.py`)
- **Structure**:
  `RealWebPreferenceDataset` ingests authentic task demonstrations from `data/raw_mind2web/` (or cached pre-processed traces). For each task demonstration:
  - **Winning Trajectory ($\tau_w$)**: Ground-truth action sequence and normalized element coordinates $(x^*, y^*)$ from the human demonstration.
  - **Losing Trajectory ($\tau_l$)**: Generated through targeted perturbation:
    * *Spatial Perturbation Mode*: Add coordinate displacement $(\Delta x, \Delta y)$ where $|\Delta x| \ge 0.15$ and $|\Delta y| \ge 0.15$, clamping to $[0, 1]^2$. Because web interactive targets have typical normalized radii $< 0.10$, this displacement guarantees a miss while maintaining realistic web context.
    * *Action Substitution Mode*: Replace the target action primitive with a sub-optimal action (e.g. `SCROLL` instead of `CLICK`, or clicking an unrelated coordinate).
- **Collation**:
  `RealPreferenceCollator` packages winning and losing sequences into aligned tensors: `chosen_input_ids`, `chosen_targets`, `chosen_mask`, `rejected_input_ids`, `rejected_targets`, and `rejected_mask`.
- **Alternative Considered**: Running active model rollouts on real HTML snapshots to generate negative trajectories.
  *Rationale for Rejection*: Active rollout generation on real HTML snapshots without a live browser backend is fragile, whereas synthetic perturbation of authentic traces is deterministic, zero-latency, and provides exact negative coordinate supervision without VRAM overhead.

### 2. Interactive Headless Browser Sandbox (`jed_m2/playwright_env.py`)
- **Interface Conformance**:
  `PlaywrightWebEnvironment` implements the exact interface expected by `TrajectoryRolloutSampler`:
  ```python
  def reset(self, scenario_id: Optional[str] = None) -> StepResult
  def step(self, action_type: ActionType, coordinates: Tuple[float, float], argument: str) -> StepResult
  ```
  It tracks `self.active_scenario`, `self.current_state`, `self.goal_satisfied`, and `self.is_terminated`.
- **Coordinate Mapping & Native Dispatch**:
  - Normalized coordinates $(cx, cy) \in [0, 1]^2$ are mapped to pixel coordinates:
    $$\text{pixel}_x = cx \times \text{viewport\_width},\quad \text{pixel}_y = cy \times \text{viewport\_height}$$
  - Actions execute native Chromium primitives:
    * `CLICK`: `page.mouse.click(pixel_x, pixel_y)`
    * `TYPE`: `page.keyboard.type(argument, delay=20)`
    * `SCROLL`: `page.mouse.wheel(0, delta_y)`
    * `NAVIGATE`: `page.goto(url)`
    * `FINISH`: Terminates episode and marks goal evaluation.
- **Local Fixtures & Benchmark Tasks**:
  Includes built-in local HTML fixture templates (e.g. e-commerce search & cart, multi-input form submission, informational search) stored locally or served via local loopback.
- **Programmatic Oracle Verifiers**:
  Each task defines a `WebTaskOracle` containing declarative checks:
  * `url_pattern`: Regex or substring match against current `page.url`.
  * `required_elements`: CSS selectors that must exist in DOM.
  * `forbidden_elements`: CSS selectors that must not exist in DOM.
  * `dom_assertions`: Key-value attribute or text assertions.
  * Reward shaping: awards milestone rewards (+0.2 to +0.5) when sub-conditions are met, and terminal reward (+1.0) upon task completion.

### 3. Dual-Modality GRPO in Browser Sandbox (`jed_m2/rl_pipeline.py` & `train_rl.py`)
- **Group Rollout Sampling**:
  `TrajectoryRolloutSampler.sample_group_rollouts()` is extended to accept `env_type="browser"` (or instantiate `PlaywrightWebEnvironment`).
  For each task in the group, $G$ rollouts are collected with temperature-scaled action selection and spatial Gaussian exploration:
  $$(x, y) \sim \mathcal{N}(\mu_{\text{coord}}, \sigma^2 I)$$
- **Joint Policy Updates**:
  In `GRPOTrainer.compute_loss()`:
  - Discrete token log-probability ratios:
    $$r_{t, i}(\theta) = \exp(\log \pi_\theta(w_{t, i}) - \log \pi_{\text{old}}(w_{t, i}))$$
  - Continuous coordinate log-probability ratios:
    $$r_{c, i}(\theta) = \exp(\log p_\theta(c_i) - \log p_{\text{old}}(c_i))$$
  - Surrogate objectives with clipping:
    $$\mathcal{L}_{\text{surr}} = \frac{1}{G} \sum_{i=1}^G \left[ \min(r_i \hat{A}_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon) \hat{A}_i) \right]$$
  - Combined objective:
    $$\mathcal{L}_{\text{GRPO}} = \mathcal{L}_{\text{token}} + \lambda_{\text{coord}} \mathcal{L}_{\text{coord}} + \beta_{\text{KL}} \mathcal{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) - c_{\text{ent}} \mathcal{H}(\pi_\theta)$$
  - Relative advantage $\hat{A}_i = \frac{R_i - \bar{R}}{\sigma_R + 1\text{e-}6}$ scales both token rationale and spatial coordinate policy gradients.

### 4. Memory Optimization on RTX 5060 Laptop GPU (< 6.0 GB VRAM)
- Precision: `torch.bfloat16` for policy and reference models.
- Gradient Checkpointing: Active across all Mamba and attention blocks.
- Group Size: $G=4$ rollouts per group, processed with micro-batching during backward passes.
- Reference Model Execution: Evaluated under `torch.no_grad()` to avoid holding reference activations in memory during active policy backward passes.
- Periodic Cache Clearing: `torch.cuda.empty_cache()` at the boundary of each iteration.

## Risks / Trade-offs

- **[Risk] Headless Chromium startup overhead slows down rollout loop**
  *Mitigation*: Maintain a persistent browser context across group rollouts, resetting page state via `about:blank` or navigation rather than re-launching the Chromium process on every turn.
- **[Risk] Playwright installation missing in headless CI or minimal developer environments**
  *Mitigation*: Provide graceful fallback in `PlaywrightWebEnvironment` to synthetic DOM emulation with explicit warning logs when Playwright or Chromium binaries are not detected.
- **[Risk] Continuous coordinate log-probabilities dominating token rationale gradients**
  *Mitigation*: Calibrate `lambda_coord` (default 0.5) and clamp continuous coordinate ratios to avoid numerical destabilization during advantage scaling.
- **[Risk] Windows CP1252 console Unicode crash**
  *Mitigation*: Enforce pure 7-bit ASCII logging (`[OK]`, `[ERROR]`, `-->`, `|`, `*`) across all newly introduced files and training scripts.
