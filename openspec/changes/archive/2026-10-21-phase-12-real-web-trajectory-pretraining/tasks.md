# Tasks

## 1. DOM Parser & Interactive Element Pruning

- [x] 1.1 Implement lightweight streaming HTML tree parser and interactive node filter in `jed_m2/real_dataset.py`, pruning passive nodes (`div`, `span`, `style`, `script`) from 10,000+ node HTML strings to candidate interactive elements (`button`, `input`, `a`, `select`, `textarea`, ARIA interactive roles). Verify candidate extraction on a mock 10,000-node DOM document.
- [x] 1.2 Implement spatial bounding box extractor and coordinate normalizer in `jed_m2/real_dataset.py`, extracting normalized $[x_{\text{min}}, y_{\text{min}}, x_{\text{max}}, y_{\text{max}}] \in [0, 1]^2$ and calculating center pointer coordinates $(x, y)$. Verify coordinate calculation, bounding box clipping, and viewport normalization.

## 2. Real Trajectory Dataset & Caching

- [x] 2.1 Implement offline Mind2Web / WebArena trajectory reader and data structures (`RealWebInteractionTurn`, `RealWebTrajectorySample`) in `jed_m2/real_dataset.py`. Verify loading and structuring offline trajectory records with user intentions and turn steps.
- [x] 2.2 Implement `RealWebTrajectoryDataset` in `jed_m2/real_dataset.py` with support for `train`, `val`, and `test` splits and persistent on-disk binary caching (`.pt`). Verify that a second dataset initialization hits the disk cache and returns identical samples without re-parsing raw HTML.
- [x] 2.3 Implement tokenization of user goals, pruned accessibility DOM trees, and thoughts into standard sequence tensors matching model vocab size in `jed_m2/real_dataset.py`. Verify token sequence bounds and delimiter token IDs.

## 3. Dynamic PyTorch Trajectory Collator

- [x] 3.1 Implement `RealTrajectoryCollator` in `jed_m2/real_dataset.py` to batch variable-length trajectories, generating padded `input_ids`, 4D causal attention masks, and action emission token index mappings. Verify padding and attention mask shapes across varying sequence lengths.
- [x] 3.2 Align dual-process loss masks in `RealTrajectoryCollator` (reasoning NLL mask, discrete action type targets/mask, continuous coordinate Huber loss mask, and argument token CE mask) and assemble JEPA transition pairs $(s_t, a_t, s_{t+1})$. Verify that all masks correctly align with `JEDM2CompositeLoss`.

## 4. Scaled Pretraining Engine

- [x] 4.1 Implement `train_pretrain.py` CLI supporting `--model_size {nano,medium}`, mixed precision `torch.bfloat16`, fused/8-bit AdamW optimizer, and cosine learning rate decay with linear warmup. Verify CLI argument parsing and model initialization on target hardware.
- [x] 4.2 Implement gradient accumulation loop (effective batch size 16 to 32 with micro-batch size 2) in `train_pretrain.py` with gradient clipping and multi-task composite loss evaluation. Verify accumulation mechanics and parameter updates.
- [x] 4.3 Implement real-time telemetry logging in `train_pretrain.py` reporting decomposed losses ($L_{\text{reason}}$, $L_{\text{type}}$, $L_{\text{coord}}$, $L_{\text{arg}}$, $L_{\text{jepa}}$, $L_{\text{aux}}$), token processing throughput (tok/s), gradient norms, and live GPU VRAM usage. Verify console and structured logging output.
- [x] 4.4 Implement periodic checkpointing and automated validation evaluation in `train_pretrain.py` computing discrete action accuracy, continuous coordinate MAE, and JEPA latent cosine similarity against target convergence criteria. Verify validation pass execution and checkpoint saving.

## 5. Verification Suite & End-to-End Validation

- [x] 5.1 Implement Test 1 in `verify_phase12.py`: DOM parser and bounding box extraction unit tests on sample realistic HTML pages. Verify parser execution time and candidate filtering accuracy.
- [x] 5.2 Implement Test 2 in `verify_phase12.py`: Dataset loading, persistent disk caching, and batch collation tensor shapes. Verify that tensor dimensions, device placement, and mask boolean types match specifications.
- [x] 5.3 Implement Test 3 in `verify_phase12.py`: Multi-step convergence test on 100 sample real trajectories verifying stable multi-task loss decrease, discrete action accuracy $> 70\%$, coordinate MAE $< 0.05$, and JEPA latent cosine similarity $> 0.85$.
- [x] 5.4 Implement Test 4 in `verify_phase12.py`: Peak GPU VRAM compliance test verifying memory remains strictly $< 6.0\text{ GB}$ during pretraining forward and backward passes on the RTX 5060 Laptop GPU.
- [x] 5.5 Update `jed_m2/__init__.py` to export `RealWebTrajectoryDataset`, `RealTrajectoryCollator`, and DOM utilities, and execute `verify_phase12.py` end-to-end to confirm all tests pass.
