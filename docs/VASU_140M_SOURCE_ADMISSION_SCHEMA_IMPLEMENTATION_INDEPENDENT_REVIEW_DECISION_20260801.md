# VASU-140M Source-Admission Schema Implementation Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5 independent review

## Decision

Accept the VASU-140M source-admission schema implementation.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a pre-existing dirty author-side worktree:
  `docs/CHANGELOG.md`, `docs/PROJECT_STATUS.md`, and `docs/ROADMAP.md` were
  already modified, and the source-admission implementation, tests, audit, and
  review packet were untracked. This review modified only this decision file.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_SOURCE_ADMISSION_SCHEMA_IMPLEMENTATION_AUDIT_20260801.md`
- `docs/VASU_140M_SOURCE_ADMISSION_SCHEMA_IMPLEMENTATION_REVIEW_PACKET.md`
- `vasu/data/vasu_140m_source_admission.py`
- `tests/test_vasu_140m_source_admission.py`
- `vasu/data/sources/schemas.py`
- `vasu/data/sources/registry.py`
- `vasu/data/sources/validation.py`
- `tests/test_data_source_registry.py`

## Rationale

The implementation is an additive metadata validator and does not mutate the
generic source registry. It enforces exact top-level and nested field sets,
canonical JSON package identity with `package_sha256` excluded from the hash
body, and fail-closed validation for modified-but-unhashed packages.

The implementation freezes the VASU-140M family, model-configuration SHA-256,
and tokenizer SHA-256. It requires a repository-relative source-registry JSON
path, source file SHA-256, source ID, and supported approval state. The
repository-bound validator verifies the referenced registry file hash, loaded
source ID, source approval state, and referenced evaluation inventory hashes.

The package contract requires immutable acquisition identity, SHA-256 raw hash
policy, legal URLs and obligations, explicit unresolved legal items, document
lineage, versioned quality policy, pinned evaluation inventories, and
cross-source deduplication evidence. Evaluation scanning and deduplication are
both required before split assignment. Approved packages require reviewer
identity, a timezone-aware review timestamp, and zero unresolved legal items.

Both `acquisition.authorized` and `training_authorized` are required to be
false. The implementation contains no acquisition, publication, schedule,
configuration, checkpoint, optimizer, or training entry point.

## Validation Results

- `python -m pytest tests\test_vasu_140m_source_admission.py tests\test_data_source_registry.py tests\test_vasu_140m_records.py -q`
  passed: 64 tests passed.
- `python -m ruff check vasu\data\vasu_140m_source_admission.py tests\test_vasu_140m_source_admission.py`
  passed.
- `python -m py_compile vasu\data\vasu_140m_source_admission.py` passed.
- `git diff --check` passed with line-ending warnings only for pre-existing
  modified documentation files.
- `git status --short` showed only pre-existing author-side documentation
  changes, the untracked implementation/test/review files, and this decision
  file after creation.
- An additional read-only adversarial probe confirmed that an approved package
  with a naive `decision.reviewed_at` timestamp is rejected.

## Compatibility Conclusion

The source-admission schema implementation is compatible with the existing
generic source registry. It adds a VASU-140M-specific package validator and
does not alter registry dataclasses, registry loading, registry validation,
mixture validation, existing dataset records, tokenizer assets, datasets,
masks, checkpoints, schedules, optimizer state, or exact-resume behavior.

No candidate source record, acquisition artifact, dataset release, checkpoint,
schedule, configuration, optimizer state, or training route was created by this
review.

## Non-Authorization

This acceptance does not authorize a source record, source discovery, registry
mutation, acquisition, data construction, release publication, configuration,
schedule, checkpoint, optimizer, authorization record, base pretraining,
instruction tuning, training, commit, or push.
