# Spec Delta

## MODIFIED Requirements

### Requirement: Parameter Scaling and Memory Budget Compliance
The system SHALL support parameter-matched configurations calibrated within $50\text{M} \pm 5\%$ total parameters for Nano architectures and within $273\text{M} \pm 5\%$ total parameters for Medium architectures, executing forward and backward passes under the 6.0 GB VRAM limit on CUDA in `bfloat16`.

#### Scenario: Hardware envelope verification
- **WHEN** the full `JEDM2Model` executes sequence forward and backward passes with sequence length 1024 on CUDA in `bfloat16`
- **THEN** total parameters remain within $45\text{M} \pm 5\text{M}$, active parameters remain within $14\text{M} \pm 3\text{M}$, peak VRAM remains below 2.0 GB, and all gradients are free of NaNs or Infinities

#### Scenario: Baseline parameter parity verification
- **WHEN** any of the four baseline models (Baseline A, B, C, D) executes sequence forward and backward passes on CUDA in `bfloat16`
- **THEN** total parameters remain within $50.0\text{M} \pm 5\%$ ($47.5\text{M} \le P \le 52.5\text{M}$ for A/B/C and $\le 53.5\text{M}$ with JEPA), peak allocated VRAM remains below 6.0 GB, and all gradients are free of NaNs or Infinities

#### Scenario: Medium model scaling parameter and memory verification
- **WHEN** the medium JED-M2 model executes sequence forward and backward passes with sequence length 2048 and gradient checkpointing on CUDA in `bfloat16`
- **THEN** total parameters remain within $273\text{M} \pm 5\%$ ($259.4\text{M} \le P \le 286.7\text{M}$), active parameters per token remain within $75\text{M} \pm 5\%$, peak VRAM remains strictly below 6.0 GB (expected $\le 3.0\text{ GB}$), and all gradients are free of NaNs or Infinities

## ADDED Requirements

### Requirement: Multi-Scale Architecture and Factory Provisioning
The system SHALL provide factory constructors (`create_jed_m2_medium`, `create_jed_m2_nano`) and configuration classes (`JEDM2MediumConfig`, `JEDM2Config`) supporting calibrated Nano (3 blocks, $d_{\text{model}}=512$, 8 experts) and Medium (6 blocks, $d_{\text{model}}=768$, 16 experts) model scales.

#### Scenario: Medium model instantiation via factory
- **WHEN** `create_jed_m2_medium()` is called without arguments
- **THEN** it returns a `JEDM2Model` initialized with 6 recurrent blocks, hidden dimension 768, 12 attention heads in MLA, 16 SwiGLU experts with top-2 routing, inner dimension 1536 in Mamba SSM, and latent dimension 384 in JEPA

#### Scenario: Multi-expert parameter accounting
- **WHEN** `count_parameters()` is called on a 16-expert MoE model configuration
- **THEN** it returns a dictionary with `total_params`, `trainable_params`, `active_params`, and `inactive_moe_params` accounting for 14 inactive experts per block and non-trainable JEPA target encoder parameters
