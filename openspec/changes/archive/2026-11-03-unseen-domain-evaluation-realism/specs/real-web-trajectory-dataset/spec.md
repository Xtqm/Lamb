# Spec Delta: real-web-trajectory-dataset

## MODIFIED Requirements

### Requirement: Website-Held-Out Split Manifests
The system SHALL support deterministic train, validation, and test manifests grouped by normalized website identity so that a website cannot occur in more than one split, and SHALL drive both offline trajectory datasets and interactive DOM simulation sandboxes directly from verified manifests.

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

#### Scenario: Manifest-driven real environment simulation
- **WHEN** `RealTaskDOMEnvironment` is initialized with a manifest path and split identifier
- **THEN** the environment instantiates tasks exclusively from that manifest split without falling back to synthetic or mock task templates

#### Scenario: Generalization split classification
- **WHEN** real web tasks are loaded for evaluation
- **THEN** tasks are partitioned into official Mind2Web generalization splits: cross-task (unseen tasks on seen websites), cross-website (unseen websites in seen domains), and cross-domain (unseen domains)
