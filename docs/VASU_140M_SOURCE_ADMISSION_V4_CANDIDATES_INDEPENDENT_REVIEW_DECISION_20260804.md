# VASU-140M Source-Admission v4 Candidates Independent Review Decision

Status: accepted as review-ready candidate packages; non-authorizing.

Reviewer: GPT-5.5 independent review

Review date: 2026-08-04

Reviewed runtime commit: `778e590e11a379d8379b76e1ffdd1057f4bb00c0`

## Decisions

FineWeb-Edu extension candidate: **Accept for separately approved admission-record construction.**

Wikimedia English 2023-11-01 candidate: **Accept for separately approved admission-record construction.**

Neither decision admits a source. Both source packages remain `decision.state=pending`.

## Findings

### FineWeb-Edu Extension

Blocking: none.

High: none.

Medium: none.

Low: none.

### Wikimedia English

Blocking: none.

High: none.

Medium: none.

Low: none.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V4_CANDIDATE_CONSTRUCTION_AUDIT_20260804.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V4_CANDIDATE_REVIEW_PACKET_20260804.md`
- `configs/data/admissions/candidates/fineweb_edu_extension_2025_26.v4.candidate.json`
- `configs/data/admissions/candidates/wikipedia_en_20231101.v4.candidate.json`
- `vasu/data/vasu_140m_admission_candidates.py`
- `vasu/data/vasu_140m_source_admission.py`
- `vasu/data/vasu_140m_source_admission_v4.py`
- `tests/test_vasu_140m_admission_candidates.py`
- `configs/data/exclusions/vasu_140m_fineweb_semantic_quarantine_20260804.json`
- All ten prompt-inventory manifests referenced by each candidate package
- Source registry records in `configs/data/sources/fineweb_edu.json` and `configs/data/sources/wikimedia.json`
- Source manifest evidence under `data/manifests/pretrain/fineweb_extension_recovery_production.json` and `data/manifests/factual/wikimedia_likelihood_source_v1.json`
- Blocked admission evidence under `configs/data/admissions/blocked_evidence/`
- Accepted prompt-matrix and quarantine remediation decision evidence referenced by the packages

## Exact Package Identities

FineWeb-Edu extension:

- Package identity: `0b178c31742249c926eb9dd0b19cedadffe51c351d936a51e6821a204fc0b910`
- Recomputed identity: `0b178c31742249c926eb9dd0b19cedadffe51c351d936a51e6821a204fc0b910`
- Decision state: `pending`
- `training_authorized=false`
- `legal_evidence.unresolved_items=[]`

Wikimedia English:

- Package identity: `06bfd284b4cc3e64554ec42925e1c3b272f6051ffcf90591934b0cc8c2968446`
- Recomputed identity: `06bfd284b4cc3e64554ec42925e1c3b272f6051ffcf90591934b0cc8c2968446`
- Decision state: `pending`
- `training_authorized=false`
- `legal_evidence.unresolved_items=[]`

## Inventory-Matrix Validation

Both packages bind exactly ten unique production-candidate prompt inventories: five development and five sealed held-out manifests across arithmetic, factuality, manual review, repetition, and robustness.

Every referenced inventory manifest was present, hash-matched, and passed `validate_inventory_manifest_files`. Every referenced inventory has `fixture_only=false`; no suite freeze, evaluation authorization, or training authorization flag was enabled.

## FineWeb Quarantine Conclusion

FineWeb binds the mandatory semantic quarantine at `configs/data/exclusions/vasu_140m_fineweb_semantic_quarantine_20260804.json`.

The binding covers:

- Required flag: `true`
- File SHA-256: `f440bde55cb499a6294fc9739c65e0955e9d61460535f5dca0c9e761b78f3c58`
- Embedded canonical SHA-256: `cfd31cf8ea68d27994b1d85caba163de1141a9ead1a8bc87c90ed39c7e67837b`
- Quarantined unique source documents: 2,659

An adversarial no-quarantine probe failed closed with `ValueError: FineWeb admission requires its reviewed quarantine`.

## Wikimedia Zero-Candidate Conclusion

Wikimedia correctly carries no mandatory exclusion. The independent contamination evidence found zero Wikimedia candidates requiring quarantine, and the package preserves that result with an empty `mandatory_exclusions` list while retaining the shared prompt-matrix binding and all v3 admission invariants.

## Legal, Acquisition, and Provenance Conclusion

FineWeb-Edu extension legal evidence records ODC-By-1.0 obligations, attribution requirements, immutable source revision, source manifest identity, acquisition identity, lineage, quality, and deduplication evidence. The remaining underlying web-rights condition is preserved as a redistribution obligation, not silently dropped as a training authorization or release permission.

Wikimedia legal evidence records CC-BY-SA-3.0 and GFDL obligations, attribution and ShareAlike requirements, immutable source revision, source manifest identity, acquisition identity, lineage, quality, and deduplication evidence.

For both candidates, `legal_evidence.unresolved_items=[]` is justified by the current repository evidence for review-ready candidate construction. This does not authorize source admission, data redistribution, likelihood construction, release construction, or training.

## Commands and Results

- `git rev-parse HEAD`
  - `778e590e11a379d8379b76e1ffdd1057f4bb00c0`
- Independent Python validation using `validate_admission_package_v4_files`, `package_identity`, and `validate_inventory_manifest_files`
  - FineWeb passed validation; identity reproduced; ten unique inventories validated; no-quarantine probe failed closed.
  - Wikimedia passed validation; identity reproduced; ten unique inventories validated; mandatory exclusions count was zero.
- `python -m pytest tests\test_vasu_140m_admission_candidates.py tests\test_vasu_140m_source_admission.py tests\test_vasu_140m_source_admission_v4.py tests\test_vasu_140m_semantic_quarantine.py -q`
  - `29 passed in 0.65s`
- `python -m ruff check vasu\data\vasu_140m_admission_candidates.py scripts\build_vasu_140m_source_admission_v4_candidates.py tests\test_vasu_140m_admission_candidates.py`
  - `All checks passed!`
- `python -m json.tool` on both candidate packages
  - FineWeb: passed
  - Wikimedia: passed
- `git diff --check`
  - Exit code 0. Git reported line-ending conversion warnings for pre-existing modified documentation files only.
- `git status --short`
  - Pre-existing unrelated modified documentation files were present and preserved.

## Compatibility

The v4 candidate packages are additive evidence records. They do not alter existing source registries, prompt inventories, datasets, masks, tokenizer assets, model-family definitions, checkpoints, schedules, optimizer state, or training configuration. Existing v3 validation remains compatible.

## Non-Authorization

This acceptance permits only later, separately reviewed construction of approved admission records. It does not admit FineWeb or Wikimedia, construct likelihood inventories, publish a release, run evaluation, access or create checkpoints, create optimizer state, create authorization records, configure schedules, configure training, or train.
