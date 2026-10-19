# Tasks: Phase 3 - Action Heads & Unified LAM Backbone

## 1. Dual-System Action Decoding Heads

- [x] 1.1 Implement `ActionType` enum and sub-heads in `jed_m2/heads.py`: `ActionGate` (`hidden_dim -> 1`), `ActionTypeHead` (`hidden_dim -> hidden_dim // 2 -> GELU -> Dropout -> num_action_types`), `CoordinateHead` (`hidden_dim -> hidden_dim // 4 -> GELU -> 2 -> Sigmoid`), and `ArgumentHead` (`hidden_dim -> vocab_size`) with shape comments and type hints; verify individual head instantiation and output dimensions.
- [x] 1.2 Implement composite `NativeActionHead` in `jed_m2/heads.py` returning structured action outputs and providing inference `predict_action(...)` decoding; verify forward pass and backward gradient flow through all heads.

## 2. Unified LAM Block & Backbone Assembly

- [x] 2.1 Extend `MultiHeadLatentAttention` in `jed_m2/mla.py` with step mode and KV-caching support while preserving full backward compatibility with Phase 1 and 2 tests; verify single-step execution against sequence forward attention.
- [x] 2.2 Implement `JEDM2Block` in `jed_m2/model.py` interleaving RMSNorm -> MambaBlock -> RMSNorm -> MLA -> RMSNorm -> SparseTop2MoE with residual connections, returning hidden representations and MoE auxiliary loss for both sequence `forward` and recurrent `step` modes; verify block forward and step execution.
- [x] 2.3 Implement unified `JEDM2Model` in `jed_m2/model.py` stacking observation/token embeddings, 3 `JEDM2Block` layers, `JEPAWorldModel`, autoregressive LM head, and `NativeActionHead` with sequence forward and multi-step recurrent rollout `step(...)` methods; verify forward pass, output shapes, and backward gradient flow.

## 3. Module Exports, Verification Suite & Context Update

- [x] 3.1 Expose all new modules (`ActionGate`, `ActionTypeHead`, `CoordinateHead`, `ArgumentHead`, `NativeActionHead`, `ActionType`, `JEDM2Block`, `JEDM2Model`) in `jed_m2/__init__.py`; verify clean package imports.
- [x] 3.2 Implement and execute `verify_phase3.py` validating sequence forward pass, dual action output shapes, recurrent rollout `step(...)` execution, backward pass (asserting zero NaNs and Infinities), parameter scaling (~45M total / ~14M active), and peak VRAM allocation (< 2.0 GB) on CUDA in `bfloat16`; verify all tests pass.
- [x] 3.3 Update `PROJECT_CONTEXT.md` roadmap marking Phase 3 as complete with verified parameter and VRAM metrics.
