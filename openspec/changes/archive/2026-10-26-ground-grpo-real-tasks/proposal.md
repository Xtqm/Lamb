# Proposal: Ground GRPO Rollouts on Diverse Real Web Tasks

## Why

In previous experiments, online GRPO reinforcement learning run across only 3 static local HTML fixtures (`ecommerce`, `form_portal`, `travel`) caused severe spatial coordinate overfitting, degrading coordinate MAE on held-out real websites from 0.3203 to 0.3335. Now that we have 1,009 genuine Mind2Web task traces and 812 WebArena task definitions on disk, grounding the GRPO rollout sampler in real DOM snapshots and programmatic element-matching oracles will enable exploration across thousands of diverse web layouts without spatial overfitting.

## What Changes

- Refactor `TrajectoryRolloutSampler` and `PlaywrightEnvironment` to ingest diverse offline real-world web task definitions (Mind2Web HTML snapshots and WebArena task goals) rather than cycling through 3 static local HTML fixtures.
- Implement `RealTaskDOMEnvironment`: a high-speed sandboxed DOM replay environment that injects genuine cleaned/raw DOM snapshots from real web traces and evaluates agent exploration actions against real elements.
- Introduce programmatic ground-truth element-matching oracles: compute step rewards using element tag matching, ARIA role matching, textual content matching, target coordinate bounding-box IoU, and string match criteria from ground-truth task annotations.
- Add real-task group rollout collation for GRPO: group-relative advantages are computed per real web task prompt, normalizing rewards across $G$ sampled exploration trajectories on identical initial DOM states.
- Retain strict hardware and memory budgets: maintain sequence length at 1024 tokens and peak VRAM strictly below 6.0 GB on the RTX 5060 Laptop GPU during rollout collection and policy updates.

## Capabilities

### New Capabilities
None.

### Modified Capabilities
- `rl-rollout-pipeline`: Modifies `Headless Interactive Browser Sandbox Rollout Execution` and `Group Rollout Collection for GRPO` to sample from real-world DOM snapshots (1,009 Mind2Web tasks and 812 WebArena tasks) with programmatic element-matching and bounding-box IoU oracles, replacing reliance on 3 static local fixtures.

## Impact

- Affected files: `jed_m2/browser.py`, `jed_m2/rl.py`, `train_rl.py`, and `eval_benchmarks.py`.
- Dependencies: Pure PyTorch, standard library HTML parser, and optional Playwright (no external CUDA kernels or breaking API changes).
- Data dependencies: Ingests `data/raw_mind2web/*.json` and `data/raw_webarena/test.raw.json`.
- Hardware impact: Peak VRAM remains under 6.0 GB during dual-modality GRPO policy optimization.
