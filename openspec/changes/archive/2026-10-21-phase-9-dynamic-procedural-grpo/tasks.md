# Tasks

## 1. Priority 1 Architectural Fixes (Backbone, MLA, Loss Harness)

- [x] 1.1 Connect JEPA to contextual backbone representations in `jed_m2/model.py`: replace external `obs_t` input with contextual backbone hidden states $h_t$ (from `h_text` and pooled visual tokens) and verify gradient backpropagation from $\mathcal{L}_{\text{JEPA}}$ directly into Mamba SSM, MLA, MoE, and token embeddings.
- [x] 1.2 Implement true MLA latent caching and $W_{UK}$ weight absorption in `jed_m2/mla.py`: update `step()` to store only $(c_{KV}, K_{\text{rope}})$ in the KV-cache, absorb $W_{UK}$ into $W_Q$ during generation, and verify numerical equivalence with sequence forward attention.
- [x] 1.3 Calibrate composite multi-task loss weights and normalization in `jed_m2/losses.py`: properly balance reasoning cross-entropy against coordinate Smooth L1 / Huber regression and primitive classification; verify balanced gradient norm contributions across heads.

## 2. Procedural Scenario Synthesis Engine (`jed_m2/scenario_generator.py`)

- [x] 2.1 Implement core procedural data structures, `DynamicScenarioGenerator`, and anti-memorization utilities (layout jitter, bounding box noise, distractor elements); verify via unit test generating element trees.
- [x] 2.2 Implement `DynamicEcommerce` environment with procedural product catalogs (10-50 items), dynamic pricing/discounts, multi-filter search, category trees, coupon validation, and multi-step checkout; verify step transitions and cart state.
- [x] 2.3 Implement `DynamicFormPortal` environment featuring multi-step wizards, heterogeneous input types, client-side regex validators, and dynamic error banners; verify input validation and milestone transitions.
- [x] 2.4 Implement `DynamicCloudAdmin` environment featuring hierarchical resource trees, action modals (delete confirmation), status toggles, and paginated tables; verify modal confirmation and resource state modification.
- [x] 2.5 Implement `DynamicWebBrowser` environment featuring multi-page directed navigation graphs, history stacks (back/forward), adversarial popup/cookie overlays, and simulated latency; verify graph traversal and modal dismissal.
- [x] 2.6 Expose `scenario_generator` classes in `jed_m2/__init__.py`; verify import and instance creation across all 4 environments.

## 3. Joint Token + Continuous Coordinate Policy Optimization (`jed_m2/rl.py`, `jed_m2/rl_pipeline.py`)

- [x] 3.1 Formulate 2D Gaussian spatial policy for `CoordinateHead` predicting continuous screen coordinate log-probabilities $\log \pi_{\text{coord}}(c \mid \mu_\theta, \sigma)$; verify log-prob output shape and gradient flow.
- [x] 3.2 Extend GRPO surrogate loss in `calculate_grpo_loss` to joint policy ratio $r_{i,t} = \exp((\log \pi_{\text{token}} - \log \pi_{\text{old\_token}}) + \lambda_{\text{coord}}(\log \pi_{\text{coord}} - \log \pi_{\text{old\_coord}}))$; verify ratio clipping and loss computation.
- [x] 3.3 Implement dense multi-factor reward shaping in `compute_shaped_reward` (+2.0 completion, +0.5 milestones, $+0.3 \times \text{IoU}$, -0.05 step penalty, -0.5 anti-looping penalty); verify reward calculation on synthetic rollouts.
- [x] 3.4 Update `TrajectoryRolloutSampler` in `jed_m2/rl_pipeline.py` to record continuous coordinate log-probabilities, sample from procedural environments, and collate joint training batches; verify rollout collection.

## 4. Curriculum Learning & Exploration Scheduling (`train_rl.py`)

- [x] 4.1 Implement multi-tier curriculum manager in `train_rl.py` (Level 1: 1-step atomic actions, Level 2: 2-3 step workflows, Level 3: 5-8 step workflows) with automatic advancement on $\ge 80\%$ moving success rate; verify tier transition logic.
- [x] 4.2 Implement adaptive entropy regularization and temperature annealing scheduler in `train_rl.py`; verify entropy penalty calculation and temperature decay.
- [x] 4.3 Add CLI flags `--curriculum`, `--entropy_coeff`, `--lambda_coord` to `train_rl.py` and connect to procedural scenario generator; verify dry-run iteration.

## 5. Verification Harness & Hardware Validation (`verify_phase9.py`)

- [x] 5.1 Implement `verify_phase9.py` checking deterministic and procedural generation across 50+ unique random seeds without exceptions or boundary violations.
- [x] 5.2 Verify numerical correctness of joint token + continuous coordinate policy ratios, clipped surrogate loss, and parameter gradients (zero NaNs/Infs).
- [x] 5.3 Verify dense reward shaping component calculations and curriculum progression threshold triggers.
- [x] 5.4 Execute full multi-task rollout and GRPO training loop on GPU, asserting strict compliance with the < 6.0 GB VRAM ceiling.
- [x] 5.5 Update `PROJECT_CONTEXT.md` documenting Phase 9 architecture, metrics, and completion status.
