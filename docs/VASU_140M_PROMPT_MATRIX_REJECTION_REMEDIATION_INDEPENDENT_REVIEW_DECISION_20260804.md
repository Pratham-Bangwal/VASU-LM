# VASU-140M Prompt-Matrix Rejection Remediation Independent Review Decision

Status: accepted by independent GPT-5.5 review; non-authorizing.

Review date: 2026-08-04

Reviewer: GPT-5.5 independent scientific reviewer

Runtime commit reviewed: `21aad6a4ce7a21a36254ca7330ad599314d93caf`

Decision: Accept

## Scope

This review reassessed the prompt-matrix rejection remediation after the
short-answer contamination policy clarification and the FineWeb quarantine
artifact. The review did not open, read, hash, copy, or use the Age private
key. It did not decrypt held-out payloads and did not expose private prompts or
answers.

## Findings

### Blocking

- None.

### High

- None.

### Medium

- The accepted remediation depends on enforcing the quarantine artifact during
  later source-admission and release-construction gates. This review accepts
  the prompt matrix only for source-admission review, not for data construction.

### Low

- The working-tree copy of the quarantine JSON has CRLF bytes, while the
  committed Git blob has the required LF identity. This is documented and
  non-blocking because the authoritative committed artifact SHA-256 matches the
  required value and the quarantine's embedded canonical identity is unchanged.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_SHORT_ANSWER_CONTAMINATION_POLICY_REMEDIATION.md`
- `docs/VASU_140M_PRIVATE_CURATOR_SEALED_CANDIDATE_INDEPENDENT_REVIEW_DECISION_20260803.md`
- `docs/VASU_140M_BASE_V2_PROMPT_MATRIX_INDEPENDENT_REVIEW_DECISION_20260803.md`
- `docs/VASU_140M_PROMPT_MATRIX_REJECTION_REMEDIATION_20260804.md`
- `evaluation/results/vasu_140m_base_v2_semantic_contamination_independent_review_20260803.json`
- `configs/data/exclusions/vasu_140m_fineweb_semantic_quarantine_20260804.json`
- `vasu/data/vasu_140m_semantic_quarantine.py`
- `scripts/build_vasu_140m_semantic_quarantine.py`
- `tests/test_vasu_140m_semantic_quarantine.py`

## Verified Identities

- Runtime commit: `21aad6a4ce7a21a36254ca7330ad599314d93caf`
- Independent review embedded result SHA-256:
  `8c58321d1e129b60f96bea079d7503cda34106de61f47dcfa482640cfa0d4b34`
- Independent review file SHA-256:
  `01b5de7d7206edc8bad1af2a12aac5055b92011923110d962a4b60846a17431b`
- FineWeb source SHA-256:
  `c90f9e21d9b73324b9165cf1fb7ffbc274fbba5ac22b7cbe48abb6d7f1e`
- Quarantine embedded canonical SHA-256:
  `cfd31cf8ea68d27994b1d85caba163de1141a9ead1a8bc87c90ed39c7e67837b`
- Quarantine authoritative committed LF Git-blob SHA-256:
  `865c757133c0a7d34c9218d06f3ffdac3279fcd3fe5eaa6478c0f4f401d88faa`
- Quarantine Windows working-tree SHA-256:
  `f440bde55cb499a6294fc9739c65e0955e9d61460535f5dca0c9e761b78f3c58`

## Answer-Overlap Policy Conclusion

The 192 preserved normalized answer-only overlaps are non-leaking audit
evidence under the accepted short-answer policy. The preserved independent
review shows zero prompt, item-ID, semantic-family, and parent-document overlap.
An independent private/development answer-overlap recheck found no substantive
multi-token overlap: all overlapping normalized answers in that check were
single-token numeric arithmetic answers. No answer overlap therefore remains
blocking after the policy remediation.

## Quarantine Coverage Conclusion

The quarantine artifact covers every independently rejected FineWeb candidate:

- rejected quarantine candidates in independent review: 2,856
- candidates recorded in quarantine artifact: 2,856
- unique quarantined FineWeb source documents: 2,659
- missing rejected candidate IDs: 0
- unexpected quarantined candidate IDs: 0
- accepted low-risk semantic candidate IDs included in quarantine: 0
- stored ordinal commitment mismatches: 0

Disposition coverage:

- exact FineWeb arithmetic/factuality candidates quarantined: 2,847
- semantic FineWeb candidates quarantined: 9
- accepted low-risk semantic candidates excluded from quarantine: 304

The quarantine is bound to the exact independent review result and FineWeb
source identity, resolves source-record ordinals through opaque ordinal
commitments, deduplicates multiple candidates per source document, and keeps
`evaluation_run_authorized=false`, `release_build_permitted=false`, and
`training_authorized=false`.

## Validation Results

- `git rev-parse HEAD`:
  `21aad6a4ce7a21a36254ca7330ad599314d93caf`
- `python -m pytest tests\test_vasu_140m_semantic_quarantine.py tests\test_vasu_140m_contamination_scan.py tests\test_vasu_140m_source_admission.py -q`:
  `32 passed in 0.66s`
- `python -m ruff check vasu\data\vasu_140m_semantic_quarantine.py scripts\build_vasu_140m_semantic_quarantine.py tests\test_vasu_140m_semantic_quarantine.py`:
  `All checks passed!`
- `python -m json.tool configs\data\exclusions\vasu_140m_fineweb_semantic_quarantine_20260804.json > $null`:
  passed
- Independent temporary fail-closed probes:
  missing evidence, unexpected source, private-key exposure flag, held-out
  exposure flag, authorization change, and overwrite all failed closed.

## Compatibility Conclusion

The remediation is compatible with the existing source-admission path because
it adds a hash-bound source-document exclusion artifact without changing
inventories, sealed payloads, datasets, checkpoints, tokenizer identities,
evaluation execution, or training state.

The remediated five-development/five-held-out prompt matrix is now eligible for
source-admission review, provided the FineWeb quarantine remains mandatory and
likelihood inventories remain deferred until source admission establishes
document-level train exclusions.

## Non-Authorization

This acceptance does not authorize likelihood inventory construction, source
admission, production suite freezing, evaluation execution, data release
construction, checkpoint access or creation, optimizer creation, authorization
records, training configuration, or training.
