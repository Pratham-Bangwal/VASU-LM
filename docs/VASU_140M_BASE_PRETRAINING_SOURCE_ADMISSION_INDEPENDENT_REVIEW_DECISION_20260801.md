# VASU-140M Base-Pretraining Source-Admission Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5 independent review

## Decision

Accept the VASU-140M base-pretraining source-admission design.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a pre-existing dirty documentation worktree:
  `docs/CHANGELOG.md`, `docs/PROJECT_STATUS.md`, and `docs/ROADMAP.md` were
  already modified, and the source-admission design, audit, and review packet
  were untracked before this decision file was created. No registry,
  implementation, protected artifact, or production data path was modified by
  this review.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_READINESS.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_AUDIT_20260801.md`
- `docs/DATA_SOURCE_REGISTRY.md`
- `vasu/data/sources/schemas.py`
- `vasu/data/sources/validation.py`
- `vasu/data/sources/registry.py`
- `tests/test_data_source_registry.py`

## Rationale

The design correctly reuses the existing metadata-only source registry instead
of creating a parallel provenance authority. The existing schema and validation
support strict source identity fields, four explicit approval states
(`pending`, `approved`, `rejected`, `blocked`), fail-closed approval checks for
unknown policy values, reviewer identity, timezone-aware review timestamps,
unresolved approval notes, local path plans, strict field loading, and
manifest-to-registry identity comparison.

The design adds the VASU-140M-specific admission boundary that the generic
registry does not itself encode: every future candidate must start as a
`pending` registry record and must provide immutable source, license,
acquisition, document-lineage, quality/filtering, evaluation-isolation, and
cross-source deduplication evidence before approval. It explicitly blocks
unknown policy values, mutable revisions, missing license evidence, absent
document lineage, incompatible licenses, and unresolved evaluation
contamination risk.

The upstream readiness and data-release plans remain consistent with this
design: existing VASU-60M 257-token FineWeb binaries are not admissible as a
VASU-140M base-pretraining release input, the published instruction seed is
excluded from base-pretraining source selection, and source admission precedes
any acquisition or release construction.

## Validation Results

- `python -m pytest tests\test_data_source_registry.py tests\test_vasu_140m_records.py -q`
  passed: 53 tests passed.
- `python scripts\smoke_vasu_140m_record_spec.py | python -m json.tool > $null`
  passed.
- `git diff --check` passed with line-ending warnings only for pre-existing
  modified documentation files.
- `git status --short` before this decision showed only pre-existing
  documentation changes and untracked source-admission review documents.
- Targeted checks found no changes under `configs/data/sources` or
  `vasu/data/sources`.
- Targeted path checks found these base-pretraining protected paths absent:
  `data/processed/vasu_140m/base_pretraining`,
  `data/manifests/vasu_140m/base_pretraining`,
  `configs/training/vasu_140m_base_pretraining.json`,
  `configs/schedules/vasu_140m_base_pretraining.json`, and
  `checkpoints/vasu_140m`.

## Non-Authorization

This acceptance does not authorize source selection, registry mutation,
source discovery, acquisition, data processing, data construction, release
construction, configuration creation, schedule creation, checkpoint creation,
optimizer creation, authorization-record creation, base pretraining,
instruction tuning, training, commit, or push.
