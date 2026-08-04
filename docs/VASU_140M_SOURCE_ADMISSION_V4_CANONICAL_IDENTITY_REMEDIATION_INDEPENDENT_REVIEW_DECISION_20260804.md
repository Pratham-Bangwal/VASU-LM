# VASU-140M Source Admission v4 Canonical-Identity Remediation Independent Review Decision

Status: accepted by independent GPT-5.5 review; non-authorizing.

Review date: 2026-08-04

Reviewer: GPT-5.5 independent scientific reviewer

Runtime commit reviewed: `d0feb26629c28c88fddfcea6ff21b677dd067313`

Decision: Accept

## Scope

This rereview assessed the remediation for the previously rejected
source-admission v4 quarantine-binding implementation. The prior blocker was
that file validation checked mandatory exclusion file bytes but did not verify
the exclusion artifact's embedded canonical quarantine identity.

No source-admission package was created or persisted. No source was admitted,
no likelihood inventory was constructed, no data release was built, no
checkpoint was accessed, and no evaluation or training authority changed.

## Findings

### Blocking

- None.

### High

- None.

### Medium

- v4 acceptance depends on later source-admission packages binding the reviewed
  FineWeb quarantine as a mandatory exclusion. This review accepts the validator
  implementation, not any concrete source admission.

### Low

- The v4 implementation remains additive: existing v3 packages and validators
  are unchanged and remain compatible.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V4_QUARANTINE_BINDING_INDEPENDENT_REVIEW_DECISION_20260804.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V4_QUARANTINE_BINDING_AUDIT_20260804.md`
- `vasu/data/vasu_140m_source_admission_v4.py`
- `tests/test_vasu_140m_source_admission_v4.py`
- `configs/data/exclusions/vasu_140m_fineweb_semantic_quarantine_20260804.json`

## Verification Conclusions

- Mandatory exclusion file bytes are hash-validated before artifact semantics
  are inspected.
- Mandatory exclusion JSON must parse successfully.
- Mandatory exclusion schema must be
  `vasu_140m_source_semantic_quarantine_v1`.
- The artifact's embedded `quarantine_sha256` must be a valid lowercase
  SHA-256 and must equal the admission binding's `canonical_sha256`.
- The prompt-matrix decision binding remains explicit, accepted, path-bound,
  and file-hash-bound.
- v3 legal, acquisition, lineage, inventory, likelihood, deduplication,
  decision, and authorization invariants remain enforced through the reused v3
  validator.

## Adversarial Probe

The original adversarial probe was reproduced against the real repository
files: change only `mandatory_exclusions[0].canonical_sha256` to
`0000000000000000000000000000000000000000000000000000000000000000`, recompute
the v4 package identity, retain the real artifact path and file SHA-256, and
run repository-file validation.

Result: validation now fails closed with
`ValueError: mandatory exclusion 0 canonical identity mismatch`.

Additional temporary-root probes:

- malformed exclusion JSON:
  `ValueError: mandatory exclusion 0 is not canonical JSON`
- unsupported exclusion schema:
  `ValueError: mandatory exclusion 0 schema is unsupported`
- missing embedded `quarantine_sha256`:
  `ValueError: mandatory exclusion 0 embedded quarantine_sha256 must be a lowercase SHA-256`
- invalid embedded SHA-256:
  `ValueError: mandatory exclusion 0 embedded quarantine_sha256 must be a lowercase SHA-256`
- embedded/admission canonical mismatch:
  `ValueError: mandatory exclusion 0 canonical identity mismatch`

## Commands and Results

- `git rev-parse HEAD`:
  `d0feb26629c28c88fddfcea6ff21b677dd067313`
- `python -m pytest tests\test_vasu_140m_source_admission_v4.py tests\test_vasu_140m_source_admission.py tests\test_vasu_140m_semantic_quarantine.py -q`:
  `28 passed in 1.01s`
- `python -m ruff check vasu\data\vasu_140m_source_admission_v4.py tests\test_vasu_140m_source_admission_v4.py`:
  `All checks passed!`
- `git diff --check`:
  passed, with existing CRLF warnings on unrelated dirty documentation files.
- `git status --short`:
  showed only the pre-existing dirty documentation files before this decision
  document was created.

## Compatibility Conclusion

The remediation preserves v3 compatibility and makes v4 suitable as an
additive quarantine-binding validator. It does not change source bytes,
inventories, datasets, checkpoints, tokenizer identities, release construction,
evaluation execution, or training state.

v4 is eligible for constructing review-ready source-admission packages, subject
to separate package construction and independent review. This decision does not
accept or admit any source.

## Non-Authorization

This acceptance does not authorize source admission, likelihood inventory
construction, evaluation execution, release construction, checkpoint access,
optimizer creation, authorization records, training configuration, or training.
