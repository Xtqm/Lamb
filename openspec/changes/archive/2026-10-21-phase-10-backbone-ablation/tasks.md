# Tasks

## 1. Architectural Baseline Modules & Config Toggles

- [x] 1.1 Add architectural configuration toggles to `jed_m2/config.py` (`attn_type`, `ffn_type`, `use_mamba`, `use_jepa`, `dense_intermediate_dim`), and verify default configuration instantiation preserves backwards compatibility.
- [x] 1.2 Implement `StandardMHA` (with rotary position embeddings, sequence forward, and uncompressed per-head recurrent KV-caching) and `DenseSwiGLUFFN` (dense gated SwiGLU feed-forward network) in `jed_m2/baselines.py`, and verify unit forward/step execution on dummy tensors.
- [x] 1.3 Implement calibrated model factory functions (`create_baseline_a`, `create_baseline_b`, `create_baseline_c`, `create_baseline_d`) in `jed_m2/baselines.py`, and verify all four baselines match the ~50.0M total parameter budget within $\pm 5\%$.

## 2. Configurable Backbone Integration

- [x] 2.1 Update `JEDM2Block` in `jed_m2/model.py` to conditionally instantiate and execute Mamba SSM, attention (`MLA` vs `StandardMHA`), and FFN (`SparseTop2MoE` vs `DenseSwiGLUFFN`) in both sequence forward and recurrent `step` modes, and verify block forward outputs and residual connections.
- [x] 2.2 Update `JEDM2Model` in `jed_m2/model.py` to support optional JEPA world model execution (`use_jepa`), variable layer depths (3 vs 4 layers), and updated parameter counting in `count_parameters()`, and verify end-to-end forward and backward passes.
- [x] 2.3 Export baseline modules and factory functions in `jed_m2/__init__.py`, and verify public package exports via python import test.

## 3. Ablation Benchmark Harness CLI

- [x] 3.1 Implement Experiment 1 (Memory Scaling Curve) in `run_ablation.py` profiling allocated VRAM, reserved VRAM, and KV-cache footprints across 16, 32, 64, and 128 recurrent turns for all four baselines, and verify $O(1)$ vs $O(N)$ memory tracking output.
- [x] 3.2 Implement Experiment 2 (Inference Step Latency) in `run_ablation.py` benchmarking per-step execution latency (mean, median, p95 ms/step in `bfloat16`) with CUDA warmup and synchronization, and verify latency measurement output.
- [x] 3.3 Implement Experiment 3 (Controlled Convergence Run) in `run_ablation.py` training all four baselines for 300 steps on identical seeded trajectory batches, and verify loss curve and metric logging.
- [x] 3.4 Implement Experiment 4 (Paper Artifacts Export) in `run_ablation.py` generating `ABLATION_REPORT.md` with markdown tables, 7-bit ASCII scaling charts, and publication-ready LaTeX `tabular` snippets, and verify report formatting and valid LaTeX syntax.

## 4. Verification Suite & End-to-End Execution

- [x] 4.1 Implement `verify_phase10.py` verifying all 4 baseline architectures, parameter budget parity ($\pm 5\%$ of 50M), forward/backward gradient integrity (zero NaNs/Infs), memory profiling, and strict adherence to the 6.0 GB VRAM limit.
- [x] 4.2 Execute `verify_phase10.py` and run full ablation benchmarks via `run_ablation.py`, verifying all tests pass and `ABLATION_REPORT.md` is populated with empirical results.
