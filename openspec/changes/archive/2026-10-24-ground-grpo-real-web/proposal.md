# Proposal: Ground GRPO on Real Web Environments

## Why

The Step 15,800 Nano model achieved strong spatial grounding (0.3203 coordinate MAE and an 18.64% tolerance accuracy at epsilon <= 0.05) through pretraining on 1,009 diverse real-world Mind2Web websites. However, the existing online RL rollout pipeline is severely decoupled from this pretraining distribution: observations are artificially truncated to 24 tokens (rendering the model 99% blind to the DOM), task goals (`Goal: {goal}`) are completely omitted from rollout prompts, environments default to toy procedural state machines, and programmatic task oracles only exist for three static HTML mock fixtures.

Attempting to run online GRPO in this state causes policy collapse, catastrophic forgetting of real-world HTML structures, and zero-advantage gradient starvation. To enable genuine capability scaling on real, complex websites, the RL architecture must be grounded in authentic web observations, calibrated with offline preference alignment on the expanded 1,009-task corpus, and paired with verifiable live browser tasks.

## What Changes

- **Fix Rollout Observation Ingestion**: Replace the hardcoded 24-token slice with structured DOM tokenization up to 512 tokens, utilizing `FastDOMParser` to extract top candidate interactive elements and normalized bounding boxes matching pretraining format.
- **Enforce Goal Conditioning in Rollouts**: Prepend explicit `Goal: {goal}` prompt prefixing at episode initialization in `TrajectoryRolloutSampler` to match the exact token grammar established in `RealTrajectoryCollator`.
- **Stage 1 Offline Preference Calibration (DPO)**: Implement and execute offline Direct Policy Optimization on the 1,009-task Mind2Web dataset (`RealWebPreferenceDataset`) using paired expert trajectories and negative spatial/action perturbations to align decision boundaries prior to online exploration.
- **Stage 2 Verifiable Headless Browser Sandbox**: Upgrade `PlaywrightEnvironment` to execute multi-step interactive rollouts on realistic multi-domain web applications with automated state verification and dense spatial IoU reward shaping.
- **Stage 3 Calibrated Online GRPO Execution**: Run online Group Relative Policy Optimization ($G=4$) with reference model KL divergence regularization ($\beta = 0.04$) and temperature-annealed exploration on verified web tasks within the strict 6.0 GB VRAM laptop ceiling.

## Capabilities

### Modified Capabilities

- `rl-rollout-pipeline`: Update trajectory sampling requirements to mandate explicit goal prompt prefixing, high-fidelity DOM observation ingestion up to sequence budget (removing the 24-token truncation), and staged execution starting with offline real-world DPO calibration before online browser rollouts.
- `playwright-browser-environment`: Extend programmatic oracle verification and reward shaping to compute dense spatial element IoU rewards, multi-step milestone verification, and structured DOM candidate extraction for live Playwright Chromium sessions.

## Impact

- **Affected Code**: `jed_m2/rl_pipeline.py`, `train_rl.py`, `jed_m2/browser.py`, `jed_m2/real_preference_dataset.py`.
- **Hardware Footprint**: Maintained under strict 6.0 GB GDDR7 VRAM on NVIDIA RTX 5060 Laptop GPU through activation checkpointing and bfloat16 mixed precision.
- **Dependencies**: Native PyTorch, standard library, and existing Playwright 1.64.0 Chromium runtime (zero compiled C++/CUDA extensions).
