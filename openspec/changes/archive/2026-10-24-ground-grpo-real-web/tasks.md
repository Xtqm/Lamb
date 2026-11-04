# Tasks: Ground GRPO on Real Web Environments - Phase 1: Offline DPO Calibration

## 1. Offline Preference Alignment (DPO) on Full Mind2Web Dataset

- [x] 1.1 Verify `RealWebPreferenceDataset` and collator on the expanded 1,009-task Mind2Web corpus with spatial coordinate perturbations ($|\Delta x| > 0.15, |\Delta y| > 0.15$) and action primitive substitutions.
- [x] 1.2 Run offline DPO calibration via `train_rl.py --mode dpo --dataset real_mind2web` seeded from `checkpoints/pretrain_nano/jed_m2_nano_step_15800.pt` for 25 iterations, verifying peak VRAM stays below 6.0 GB.
- [x] 1.3 Verify DPO convergence metrics (positive reward margin, accuracy > 70%, decreasing loss) and save calibrated policy checkpoint `checkpoints_rl/jed_m2_nano_dpo_calibrated.pt`.
