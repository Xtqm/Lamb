# Spec Delta

## ADDED Requirements

### Requirement: Provenance-Aware Real-Only Trajectory Selection
The system SHALL distinguish captured real interaction traces from procedural or reconstructed synthetic examples and SHALL expose a real-only dataset mode for controlled experiments.

#### Scenario: Real-only dataset load
- **WHEN** a training or evaluation run requests real-only mode
- **THEN** every returned sample has trace provenance identifying its source and version, source record, website identity, upstream split, and content checksum, and no procedural or reconstructed synthetic sample is returned

#### Scenario: Missing authentic trace data
- **WHEN** real-only mode is requested but no usable authentic traces are available
- **THEN** dataset initialization fails with an actionable error and does not fall back to procedural or synthetic data

#### Scenario: Mixed source inventory
- **WHEN** authentic traces and synthetic examples are both present in the configured data roots
- **THEN** the real-only dataset includes only authentic traces and reports the excluded synthetic record count

### Requirement: Website-Held-Out Split Manifests
The system SHALL support deterministic train, validation, and test manifests grouped by normalized website identity so that a website cannot occur in more than one split.

#### Scenario: Website-disjoint manifest generation
- **WHEN** a site-grouped split is requested with a fixed seed
- **THEN** the system assigns whole websites to train, validation, or test, writes a reproducible manifest, and verifies that the three website sets are pairwise disjoint

#### Scenario: Mind2Web category is not used as website identity
- **WHEN** a Mind2Web record contains both a `website` field and broad `domain` or `subdomain` category fields
- **THEN** the split groups by the normalized `website` identity and retains category fields only for stratification and reporting

#### Scenario: Duplicate and snapshot leakage check
- **WHEN** manifests are generated from multiple source records
- **THEN** the system detects duplicate task records and shared snapshot identifiers across splits, excludes or reports leakage before training, and records the disposition in the manifest summary

#### Scenario: Insufficient site diversity
- **WHEN** a run requests minimum site counts that the available corpus cannot satisfy
- **THEN** preparation fails before training and reports the available and required counts for each split

### Requirement: Dataset Diversity Audit
The system SHALL produce a machine-readable and human-readable audit of the dataset selected for each experiment.

#### Scenario: Audit a prepared real-data corpus
- **WHEN** train, validation, and test manifests are prepared
- **THEN** the audit reports per-split trajectory counts, distinct website counts, source counts, task-category coverage, action-type coverage, exclusions, and manifest/content checksums

#### Scenario: Enforce the ablation diversity floor
- **WHEN** the backbone ablation requests its default real-data profile
- **THEN** preparation requires at least 50 training websites and at least 10 distinct validation and 10 distinct test websites, unless the experiment explicitly records different thresholds before training

### Requirement: Faithful Real-Trace Supervision
The system SHALL preserve recorded interaction labels and SHALL NOT fabricate rationale, target, or coordinate supervision for real-only samples.

#### Scenario: Parse recorded interaction labels
- **WHEN** a captured interaction trace is converted into a training sample
- **THEN** the original operation label and argument are retained, and any mapping into model action classes is explicit and counted in the dataset audit

#### Scenario: Preserve Mind2Web SELECT operations
- **WHEN** a captured Mind2Web action has operation `SELECT` and the ablation model uses the seven-action configuration
- **THEN** the sample target is the distinct SELECT class and is not converted to CLICK

#### Scenario: Missing rationale annotation
- **WHEN** a real trace contains no recorded rationale text
- **THEN** the sample has no rationale target and its rationale loss mask is false

#### Scenario: Missing or ambiguous target geometry
- **WHEN** a real trace has no unambiguous labeled target element or valid target bounding box
- **THEN** the sample marks the relevant target or coordinate label unavailable, masks its training loss and evaluation metric, and does not substitute a fabricated box or nearest element

#### Scenario: Missing observation snapshot
- **WHEN** a real trace action lacks a captured observation snapshot
- **THEN** the action is excluded or explicitly marked unusable with a recorded reason and is not filled with a procedural page template
