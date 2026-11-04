# Proposal: Ground JED-M2 Reinforcement Learning in Real-World Web Data

## Why

Currently, JED-M2 pretraining trains on authentic real-world web trajectories from Mind2Web (`train_pretrain.py`, `jed_m2/real_dataset.py`), yet the reinforcement learning pipeline (`train_rl.py`, `jed_m2/rl.py`, `jed_m2/rl_pipeline.py`) samples rollouts exclusively against `WebOSEnvironment` (`jed_m2/environment.py`), which uses procedural synthetic DOM mocks and Python state machines. This creates an architectural domain distribution gap: the policy is pretrained on complex real HTML DOMs, but fine-tuned on simplistic mock dictionaries. 

To achieve robust real-world generalization without policy degeneration, we must ground policy optimization in authentic web data through both offline preference optimization (DPO on real Mind2Web traces with spatial perturbations) and online interactive browser sandboxing (GRPO on headless Chromium with deterministic programmatic oracle verifiers).

## What Changes

- **Track A — Offline Preference Optimization (DPO on Real Mind2Web Traces)**:
  - Create `jed_m2/real_preference_dataset.py` implementing `RealWebPreferenceDataset` and `RealPreferenceCollator`:
    * Constructs paired preference trajectories $(\tau_w, \tau_l)$ directly from Mind2Web offline traces.
    * Winning Trajectory ($\tau_w$): Authentic human expert demonstration from `data/raw_mind2web/` or disk cache.
    * Losing Trajectory ($\tau_l$): Ground-truth actions perturbed by spatial coordinate noise ($|\Delta x| > 0.15, |\Delta y| > 0.15$) or substituted with sub-optimal action primitives (e.g. spurious scrolls or incorrect element clicks).
    * Collates paired token sequences, attention masks, and action loss masks for policy and reference log-probability evaluation.
  - Wire dataset selection into `train_rl.py` via `--dataset real_mind2web`, executing `calculate_dpo_loss()` in `jed_m2/rl.py`.

- **Track B — Headless Interactive Browser Sandbox (Playwright + Local Web Benchmarks)**:
  - Create `jed_m2/playwright_env.py` implementing `PlaywrightWebEnvironment` matching the `WebOSEnvironment` interface (`reset()`, `step(action_type, coordinates, argument)`):
    * Launches headless Chromium via Playwright (`viewport 1280x800`).
    * Supports loading local HTML fixtures and locally served reproducible web applications (e.g., Saleor e-commerce, Postmill forum, or WebArena tasks).
    * Maps normalized coordinates $(x, y) \in [0, 1]^2$ to pixel coordinates and executes native browser mouse and keyboard events (`page.mouse.click()`, `page.keyboard.type()`, `page.mouse.wheel()`).
    * Extracts live accessibility DOM trees and normalized element bounding boxes for observation formatting.
    * Implements deterministic programmatic oracle verifiers (URL state regex, DOM element presence/absence, and element property checks) for milestone and terminal reward shaping.

- **Track C — Dual-Modality GRPO Verification on Interactive Web**:
  - Extend `TrajectoryRolloutSampler` in `jed_m2/rl_pipeline.py` and `train_rl.py --mode grpo` to collect groups of $G$ rollouts against `PlaywrightWebEnvironment` alongside existing procedural generators.
  - Verify group-normalized advantage updates across both discrete token policy heads (thought and action tokens) and continuous spatial Gaussian coordinate heads (`lambda_coord`), ensuring joint policy learning.

- **Verification & Constraints**:
  - Add comprehensive test suite `tests/test_real_rl.py` verifying preference dataset collation, browser stepping, oracle verification, and backward pass gradient flow.
  - Guarantee strict adherence to the 6.0 GB VRAM ceiling on NVIDIA RTX 5060 Laptop GPU (bfloat16, gradient checkpointing).
  - Enforce pure 7-bit ASCII logging (no Unicode box-drawing characters).
  - Maintain strict git hygiene for dataset caches and weights.

## Capabilities

### New Capabilities
- `real-web-preference-dataset`: Offline preference trajectory pairing directly from Mind2Web real-world traces, generating winning human expert demonstrations and spatial/action-perturbed losing trajectories formatted for Direct Policy Optimization (DPO).

### Modified Capabilities
- `playwright-browser-environment`: Expand requirements to mandate `PlaywrightWebEnvironment` drop-in interface parity with `WebOSEnvironment`, local HTML fixture and benchmark app serving support, and deterministic programmatic oracle verification for automated milestone and terminal reward calculation.
- `rl-rollout-pipeline`: Expand requirements to support online group rollout collection within interactive browser sandboxes, dual-modality GRPO optimization updating both discrete token heads and continuous spatial Gaussian coordinate heads (`lambda_coord`), and offline preference optimization via `--dataset real_mind2web`.

## Impact

- **Affected Code**: `train_rl.py`, `jed_m2/rl_pipeline.py`, `jed_m2/rl.py`, `jed_m2/browser.py`, `jed_m2/__init__.py`.
- **New Files**: `jed_m2/real_preference_dataset.py`, `jed_m2/playwright_env.py`, `tests/test_real_rl.py`.
- **Dependencies**: 100% native PyTorch and standard library; Playwright for browser execution (no external compiled C++/CUDA binary packages like FlashAttention or Triton).
- **Hardware Budget**: Peak VRAM strictly under 6.0 GB on NVIDIA GeForce RTX 5060 Laptop GPU (8GB GDDR7).
