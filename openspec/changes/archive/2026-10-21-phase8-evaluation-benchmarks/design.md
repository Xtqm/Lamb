# Technical Design: Phase 8 Standardized Evaluation Benchmark Harness

## Context

JED-M2 has established its core neural modules (MLA, Mamba SSM, MoE, JEPA), action heads (discrete primitive, continuous coordinate, argument decoder, action gate), training infrastructure (SFT and GRPO RL), multimodal visual grounding, and live Playwright browser integration. 

To formally measure agent capability and quantify policy improvements (such as Base SFT vs. GRPO Step 40 vs. Converged RL policies), a standardized evaluation harness is required. This harness must operate under the strict hardware budget (RTX 5060 8GB VRAM ceiling < 6.0 GB) and adhere to project engineering conventions (pure PyTorch, type annotations, tensor shape documentation, and ASCII-safe terminal output).

See `proposal.md` for motivation and `specs/evaluation-benchmark-harness/spec.md` for formal requirements.

## Goals / Non-Goals

**Goals:**
- Provide `jed_m2/benchmark.py` containing `TaskEvaluator` and standardized evaluation routines.
- Implement formal metric calculations:
  - Task Success Rate (SR %)
  - Step Success Rate
  - Action Primitive Precision, Recall, and F1 (micro and macro across 6 primitives)
  - Spatial Coordinate MAE and Tolerance Accuracy ($\epsilon \le 0.05$ and bounding box containment)
  - Trajectory Efficiency Ratio ($L_{\text{optimal}} / L_{\text{agent}}$)
  - Decision Latency (mean/median ms/step) and Token Throughput (tokens/sec)
- Support offline trajectory replay evaluation against Mind2Web and WebArena test datasets.
- Support online closed-loop multi-turn evaluation across simulated `WebOSEnvironment` and live Playwright sessions.
- Provide `eval_benchmarks.py` CLI supporting side-by-side comparative benchmarking of multiple checkpoints.
- Generate ASCII terminal comparison tables and export comprehensive Markdown reports (`BENCHMARK_REPORT.md`).
- Implement `verify_phase8.py` to validate evaluation pipelines, numerical sanity, comparative logic, and VRAM limits.

**Non-Goals:**
- Altering model architecture, loss functions, or core training pipelines.
- Modifying weights or initiating training within the evaluation harness.
- Requiring external heavyweight Dockerized benchmark environments (e.g. full WebArena cloud clusters); evaluation harnesses use native synthetic/mock trajectories and local simulated/Playwright environments.

## Decisions

### 1. Metric Container & Architecture Design
- **Decision**: Define structured dataclasses (`StepMetric`, `EpisodeMetric`, `BenchmarkSummary`) in `jed_m2/benchmark.py` to encapsulate step-level, episode-level, and run-level results.
- **Rationale**: Dataclasses ensure type safety, structured serialization to dict/JSON, and clean decoupled aggregation in `TaskEvaluator`.
- **Alternatives Considered**: Raw dictionary passing. Rejected due to vulnerability to key typos and inconsistent field tracking.

### 2. Multi-Class Action Primitive F1 Calculation
- **Decision**: Implement confusion matrix tracking across the 6 discrete primitives (`CLICK`, `TYPE`, `SCROLL`, `NAVIGATE`, `API_CALL`, `FINISH`). Compute per-class precision and recall, macro F1, and micro (overall) F1.
- **Rationale**: Evaluating aggregate accuracy alone obscures performance imbalances on rare but critical primitives (e.g. `FINISH` or `SCROLL`).
- **Zero-Division Guard**: When true positives and false positives are zero for a class, define precision/recall as $1.0$ if ground truth and prediction counts are both zero, else $0.0$.

### 3. Spatial Coordinate Precision & Tolerance
- **Decision**: Evaluate continuous pointer coordinates $(x, y) \in [0, 1]^2$ using two complementary metrics:
  1. Mean Absolute Error: $\text{MAE} = \frac{1}{2}(|\hat{x} - x^*| + |\hat{y} - y^*|)$.
  2. Tolerance Match: Marked correct if Euclidean distance $\sqrt{(\hat{x} - x^*)^2 + (\hat{y} - y^*)^2} \le 0.05$ OR predicted coordinate falls within target element bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$.
- **Rationale**: Continuous screen click accuracy requires both continuous error measurement and practical target acquisition tolerance.

### 4. Offline Dataset Replay vs. Online Closed-Loop Rollout
- **Decision**: 
  - `evaluate_offline_dataset(...)`: Feeds pre-recorded sequential observations from validation/test trajectories into the agent, evaluating predictions against recorded ground-truth actions step-by-step.
  - `evaluate_online_environment(...)`: Executes multi-turn closed-loop rollouts where agent actions change the environment state (`WebOSEnvironment` or Playwright) dynamically until `FINISH` or `max_steps`.
- **Rationale**: Offline replay tests imitation fidelity and perception accuracy without simulation drift; online rollout tests autonomous policy execution and recovery.

### 5. Side-by-Side Checkpoint Comparison & CLI Reporting
- **Decision**: In `eval_benchmarks.py`, accept multiple checkpoint paths via `--checkpoints path1.pt path2.pt ...` (or `--model_path`). Evaluate each checkpoint sequentially on the exact same task seeds, accumulating metrics in a comparative table.
- **Output Formats**:
  - ASCII terminal summary using basic standard characters (`+`, `-`, `|`) to ensure compatibility with Windows CP1252 shells.
  - Automated export to `BENCHMARK_REPORT.md` with full Markdown tables and comparative analysis.

### 6. Hardware Budget & Memory Management
- **Decision**: Execute all evaluations inside `torch.no_grad()`, with `model.eval()`, clearing recurrent state buffers between episodes and calling `torch.cuda.empty_cache()` between checkpoint loads.
- **Rationale**: Prevents memory leaks and maintains peak VRAM allocation well below the 6.0 GB ceiling (< 1.5 GB expected).

## Risks / Trade-offs

- **[Risk: Division by zero when evaluating empty or zero-step episodes]**  
  *Mitigation*: Guard all metric denominators with explicit count checks and safe fallbacks (e.g. efficiency ratio defaults to 0.0 if agent took 0 steps).
- **[Risk: VRAM accumulation when sequentially evaluating multiple large checkpoints]**  
  *Mitigation*: Explicitly delete model instances, clear PyTorch CUDA cache via `torch.cuda.empty_cache()`, and reset memory stats between checkpoint evaluations.
- **[Risk: Windows Unicode encoding crashes on non-ASCII characters]**  
  *Mitigation*: Restrict all CLI output to 7-bit ASCII formatting (avoid unicode borders or emojis).
- **[Risk: Live Playwright browser flakiness during automated evaluation]**  
  *Mitigation*: Provide simulated environment fallback (`--benchmark simulated`) and include timeout wrappers with graceful error reporting during Playwright runs.
