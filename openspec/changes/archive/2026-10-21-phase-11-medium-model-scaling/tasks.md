# Tasks

## 1. Architectural Configuration & Scaling Tiers

- [x] 1.1 Add `model_size: Literal["nano", "medium"] = "nano"` field and calibrated `JEDM2MediumConfig` factory in `jed_m2/config.py`, verifying all medium hyperparameters ($d_{\text{model}}=768$, 6 layers, 16 experts, 12 heads MLA, $d_{\text{inner}}=1536$, $d_z=384$).
- [x] 1.2 Export `JEDM2MediumConfig` in `jed_m2/__init__.py` and verify clean import in Python.

## 2. Model Factory & Dynamic Parameter Accounting

- [x] 2.1 Implement `create_jed_m2_medium(config=None)` and `create_jed_m2_nano(config=None)` factory functions in `jed_m2/model.py`.
- [x] 2.2 Update `count_parameters()` in `jed_m2/model.py` to dynamically compute total, active, trainable, and inactive parameters for 16-expert MoE configurations.
- [x] 2.3 Verify seamless activation checkpointing across all 6 recurrent blocks in `JEDM2Model.forward()` without activation leakage or memory fragmentation.
- [x] 2.4 Export `create_jed_m2_medium` and `create_jed_m2_nano` in `jed_m2/__init__.py` and verify module export integrity.

## 3. CLI & Training Harness Integration

- [x] 3.1 Update `train.py` argument parser and model initialization to support `--model_size {nano,medium}` flag with automatic activation checkpointing.
- [x] 3.2 Update `train_rl.py` argument parser and dual-model instantiation (active policy and frozen reference policy) to support `--model_size {nano,medium}`.
- [x] 3.3 Update `eval_benchmarks.py` argument parser and checkpoint loader to support `--model_size {nano,medium}`.

## 4. Verification Suite & Performance Profiling

- [x] 4.1 Author `verify_phase11.py` implementing Test 1 (Parameter budget parity: total params within $273\text{M} \pm 5\%$ and active params within $75\text{M} \pm 5\%$).
- [x] 4.2 Add Test 2 in `verify_phase11.py` (End-to-end forward and backward gradient integrity across LM loss, action gate loss, action type loss, coordinate loss, argument loss, and JEPA loss with zero NaNs/Infs).
- [x] 4.3 Add Test 3 in `verify_phase11.py` (Recurrent `step()` rollout latency profiling and $O(1)$ constant-memory state caching over multi-step horizons).
- [x] 4.4 Add Test 4 in `verify_phase11.py` (Peak training VRAM allocation verification under sequence lengths 1024 and 2048 in native `bfloat16`, asserting peak memory strictly $< 6.0\text{ GB}$).
- [x] 4.5 Execute `python verify_phase11.py` on the NVIDIA GeForce RTX 5060 Laptop GPU and verify that all 4 validation gates pass with exit code 0.
