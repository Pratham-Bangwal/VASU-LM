# VASU-140M Source Admission v4 Quarantine-Binding Independent Review Decision

Status: rejected by independent GPT-5.5 review; non-authorizing.

Review date: 2026-08-04

Reviewer: GPT-5.5 independent scientific reviewer

Runtime commit reviewed: `41d2bca2f732145166d82c343341602b572e4595`

Decision: Reject

## Scope

This review assessed whether the additive v4 source-admission validator
correctly preserves v3 admission invariants while making the accepted
prompt-matrix decision and mandatory FineWeb quarantine first-class,
hash-bound dependencies.

No source-admission package was created or persisted. No source was admitted,
no data was built, no checkpoint was accessed, and no evaluation or training
authority changed.

## Findings

### Blocking

- `validate_admission_package_v4_files` verifies the mandatory exclusion file
  byte SHA-256 but does not verify that the package's `canonical_sha256` equals
  the bound quarantine artifact's embedded `quarantine_sha256`. An independent
  in-memory probe changed `mandatory_exclusions[0].canonical_sha256` to
  `0000000000000000000000000000000000000000000000000000000000000000`,
  recomputed the v4 package identity, and validation unexpectedly passed
  against the real repository files. This means v4 does not yet enforce the
  audit claim that mandatory exclusions are bound by both file and canonical
  identities.

### High

- None beyond the blocking canonical-identity gap.

### Medium

- The v4 test coverage proves FineWeb cannot omit exclusions and proves v4
  package identity covers mutation, but it does not cover repository-file
  semantic verification of exclusion canonical identity.

### Low

- Existing v3 validation remains intact and the v4 layer correctly rejects
  missing FineWeb exclusions, rejected prompt-matrix decisions, optional
  exclusions, unsafe paths, and authorization changes in the probed cases.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_PROMPT_MATRIX_REJECTION_REMEDIATION_INDEPENDENT_REVIEW_DECISION_20260804.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V4_QUARANTINE_BINDING_AUDIT_20260804.md`
- `vasu/data/vasu_140m_source_admission.py`
- `vasu/data/vasu_140m_source_admission_v4.py`
- `tests/test_vasu_140m_source_admission.py`
- `tests/test_vasu_140m_source_admission_v4.py`
- `configs/data/exclusions/vasu_140m_fineweb_semantic_quarantine_20260804.json`

## Commands and Results

- `git rev-parse HEAD`:
  `41d2bca2f732145166d82c343341602b572e4595`
- `python -m pytest tests\test_vasu_140m_source_admission_v4.py tests\test_vasu_140m_source_admission.py tests\test_vasu_140m_semantic_quarantine.py -q`:
  `27 passed in 0.81s`
- `python -m ruff check vasu\data\vasu_140m_source_admission_v4.py tests\test_vasu_140m_source_admission_v4.py`:
  `All checks passed!`
- Independent in-memory validation probes:
  - baseline real-file v4 package: passed
  - missing FineWeb exclusion: `ValueError`
  - unsafe exclusion path: `ValueError`
  - optional exclusion: `ValueError`
  - rejected prompt-matrix decision: `ValueError`
  - authorization change: `ValueError`
  - false `canonical_sha256` with real file SHA and recomputed package identity:
    unexpected pass
- `git diff --check`:
  passed, with existing CRLF warnings on unrelated dirty documentation files.
- `git status --short`:
  showed the pre-existing dirty documentation files plus this decision document
  after creation.

## Compatibility Conclusion

The implementation is additive and preserves the v3 validator path for existing
v3 packages. However, v4 is not yet acceptable as a quarantine-binding gate
because repository-file validation does not prove the mandatory exclusion's
canonical identity.

## Eligibility

v4 is not yet eligible for constructing review-ready source-admission packages.
The exact blocker is the missing validation that each mandatory exclusion's
declared `canonical_sha256` matches the bound artifact's internal canonical
identity.

## Non-Authorization

This rejection does not admit any source and does not authorize likelihood
inventory construction, evaluation execution, release construction, checkpoint
access, optimizer creation, authorization records, training configuration, or
training.
