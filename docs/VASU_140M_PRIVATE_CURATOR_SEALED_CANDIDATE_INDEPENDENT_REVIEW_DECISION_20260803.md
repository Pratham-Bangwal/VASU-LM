# VASU-140M Private Curator Sealed Candidate Independent Review Decision

Status: rejected by independent GPT-5.5 review; non-authorizing.

Review date: 2026-08-04

Reviewer: GPT-5.5 independent reviewer and private held-out curator

Decision: Reject

## Scope

This review independently checked the five sealed private held-out candidate
inventories, detached sealing receipt, public provenance indexes, contamination
indexes, private-input correspondence, and repository plaintext boundary for the
VASU-140M base-model evaluation-v2 held-out curator package.

No Age secret key was opened, read, copied, moved, hashed, modified, or used.
No sealed repository payload was decrypted. No held-out prompt or answer text is
included in this decision.

## Findings

### Blocking

- The sealed candidate package is internally reproducible, but it cannot be
  accepted as part of the current production prompt matrix because the
  independent matrix review found 192 normalized answer overlaps between the
  development inventories and private held-out inventories.
- The independent source-contamination review found source candidates requiring
  quarantine before source-admission review: 2,847 exact FineWeb candidates and
  9 semantic FineWeb candidates requiring quarantine.

### High

- None beyond the blocking fail-closed conditions.

### Medium

- Exact answer-only matches are dominated by arithmetic-style outputs and are
  intentionally represented only by opaque commitments and quarantine decisions;
  they require later policy disposition without exposing held-out content.

### Low

- No low-severity defects were identified in the sealed-file structure,
  manifest validation, receipt binding, or plaintext boundary.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_PRIVATE_CURATOR_SEALING_IMPLEMENTATION_20260803.md`
- `docs/VASU_140M_PRIVATE_CURATOR_SEALING_EXECUTION_AUDIT_20260803.md`
- `evaluation/candidates/vasu_140m_base_v2_heldout_curator_v1/sealing_receipt.json`
- `configs/evaluation/vasu_140m_private_curator_intake_20260803.json`
- `configs/data/admissions/fineweb_edu_extension_2025_26.pending.json`
- `configs/data/admissions/wikipedia_en_20231101.pending.json`
- `configs/data/admissions/blocked_evidence/fineweb_edu_extension_2025_26.blocked.json`
- `configs/data/admissions/blocked_evidence/wikipedia_en_20231101.blocked.json`
- `scripts/seal_vasu_140m_private_curator_inputs.py`
- `evaluation/framework/vasu_140m_private_curator_sealing.py`
- `evaluation/framework/vasu_140m_private_curator_intake.py`
- Public sealed-candidate manifests, ciphertext files, provenance indexes, and
  contamination indexes under
  `evaluation/candidates/vasu_140m_base_v2_heldout_curator_v1/`
- Private JSONL inventories under `D:\VASU_PRIVATE_HELDOUT`, inspected only for
  independent correspondence and isolation checks.

## Verified Sealed Identities

| Dimension | Records | Inventory SHA-256 | Ciphertext SHA-256 |
| --- | ---: | --- | --- |
| arithmetic | 1,000 | `ccc2afe2dbf4414a523c4c3715b9e8defbdf146f9c39b0a98fcd98cc2279274f` | `dd26b6b07e0b0f1199d4f508c9e81f3958fcca891752c1e277efccb75ddeff8a` |
| factuality | 200 | `3d1b43edc6c01f7c9de4f67b2b65b1db3a8a74658d43f9a49d7a3691d62982e1` | `693134d7b4628e63c2415d3f28f93597ea0ec5a53a9fba3ac08071e6c96487cc` |
| manual_review | 60 | `e364c07f9ef0e3356bdb27f61cc761b95213a277e9598754ad278c792e9cf4b6` | `075113c520916dfdff0eeab1b9a6cb7b78cc81a1482bfbe071727084235a9daf` |
| repetition | 120 | `68a986f226e63f7c1015f3dcfebaf5a0444f4fb4d2743bd14b78fa2517706a0b` | `3b730912f87da0fc8f7651dd73d35c0161ee1a54c1b423ec0ab1a5c5076de571` |
| robustness | 120 | `09bfe0eb1cac11561ec82a44b45b113dafc55aff9bcce923a9b9c7824e922783` | `cc407cc22135a876ee06cff569a6fa14200f85d747b9dbab621a5e80096b7e63` |

Detached receipt identity:
`c14fe696143e618ffdf4f501f3ba90ab8fe4f48fab93e96bfe99cd3e2ce52d18`.

## Validation Results

- Sealed candidate manifests validated successfully without decryption.
- Public provenance and contamination indexes reproduced exactly from the
  private curator inputs in memory.
- Plaintext held-out payload search under
  `evaluation/candidates/vasu_140m_base_v2_heldout_curator_v1/` found zero
  plaintext payload files.
- Independent contamination evidence was recorded at
  `evaluation/results/vasu_140m_base_v2_semantic_contamination_independent_review_20260803.json`.

## Non-Authorization

This rejection does not authorize likelihood inventory construction, production
suite freezing, source admission, data-release construction, model execution,
checkpoint access or creation, evaluation execution, optimizer updates,
authorization records, training configuration, or training.
