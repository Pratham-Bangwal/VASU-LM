# VASU-140M Base V2 Prompt Matrix Independent Review Decision

Status: rejected by independent GPT-5.5 review; non-authorizing.

Review date: 2026-08-04

Reviewer: GPT-5.5 independent reviewer and private held-out curator

Decision: Reject

## Scope

This review assessed whether the five development inventories and five sealed
private held-out inventories form an acceptable production prompt-inventory
matrix for VASU-140M source-admission review.

No held-out plaintext was copied into the repository or included in this
decision. Repository evidence is limited to hashes, counts, categories,
decisions, and non-sensitive explanations.

## Findings

### Blocking

- Development/private held-out answer isolation is incomplete: the independent
  matrix check found 192 normalized answer overlaps. Prompt, item-ID,
  semantic-family, and parent-document overlaps were zero, but the requested
  answer-isolation requirement fails.
- Independent source-contamination review found 2,847 exact FineWeb candidates
  requiring quarantine and 9 semantic FineWeb candidates requiring quarantine.
  This blocks accepting the matrix for source-admission review until the
  quarantine set is resolved.

### High

- None beyond the blocking fail-closed conditions.

### Medium

- The exact candidate set is concentrated in arithmetic answer-style
  commitments; the review conservatively quarantines exact prompt-or-answer
  overlaps without exposing private answer text.

### Low

- The five development inventories are structurally valid as development-only
  inventories, and the five sealed held-out candidate inventories are
  reproducible and receipt-bound.

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
- Development inventories under
  `evaluation/fixtures/vasu_140m_assistant_authored_internal_v1/`
- Sealed held-out public manifests, ciphertexts, provenance indexes, and
  contamination indexes under
  `evaluation/candidates/vasu_140m_base_v2_heldout_curator_v1/`
- Source-contamination corpora:
  `data/interim/pretrain/fineweb_extension_recovered.jsonl.gz` and
  `data/processed/pretrain/factual/wikimedia_likelihood_source_v1/documents.jsonl`

## Inventory Identities

### Development Inventories

| Dimension | Records | Inventory SHA-256 |
| --- | ---: | --- |
| arithmetic | 1,000 | `fa9855f439220e811f5cf5de5978eeae358f8e6eeb3cecc33f61315c880c4eaf` |
| factuality | 200 | `eb84e4904ed2da8f8309bbb9df4d557490a6dc6f3c1c75ebae72a1f0749a9045` |
| manual_review | 60 | `955df5ef3e306b975756f076fe9572b0433127d993ca40cb4b4c7303d6e39bad` |
| repetition | 120 | `68ca327e0fe5b7d8e60420d5723478833ae77c50fea908f44121aa47691b629f` |
| robustness | 120 | `fc37fbc37e2019025406af8e6bdd3bb5d8748a904031e7c68b68bef13cdf75a2` |

### Sealed Held-Out Inventories

| Dimension | Records | Inventory SHA-256 | Ciphertext SHA-256 |
| --- | ---: | --- | --- |
| arithmetic | 1,000 | `ccc2afe2dbf4414a523c4c3715b9e8defbdf146f9c39b0a98fcd98cc2279274f` | `dd26b6b07e0b0f1199d4f508c9e81f3958fcca891752c1e277efccb75ddeff8a` |
| factuality | 200 | `3d1b43edc6c01f7c9de4f67b2b65b1db3a8a74658d43f9a49d7a3691d62982e1` | `693134d7b4628e63c2415d3f28f93597ea0ec5a53a9fba3ac08071e6c96487cc` |
| manual_review | 60 | `e364c07f9ef0e3356bdb27f61cc761b95213a277e9598754ad278c792e9cf4b6` | `075113c520916dfdff0eeab1b9a6cb7b78cc81a1482bfbe071727084235a9daf` |
| repetition | 120 | `68a986f226e63f7c1015f3dcfebaf5a0444f4fb4d2743bd14b78fa2517706a0b` | `3b730912f87da0fc8f7651dd73d35c0161ee1a54c1b423ec0ab1a5c5076de571` |
| robustness | 120 | `09bfe0eb1cac11561ec82a44b45b113dafc55aff9bcce923a9b9c7824e922783` | `cc407cc22135a876ee06cff569a6fa14200f85d747b9dbab621a5e80096b7e63` |

## Isolation Results

| Isolation check | Overlap count |
| --- | ---: |
| item IDs | 0 |
| semantic-family IDs | 0 |
| parent-document IDs | 0 |
| normalized prompts | 0 |
| normalized answers | 192 |

## Source-Contamination Results

Source hashes:

- FineWeb recovered source:
  `c90f9e21d9b73324b9165cf1fb7ffbc274fbba5ccba5ac22b7cbe48abb6d7f1e`
- Wikimedia likelihood source:
  `bc2a014fadcfca02cdebce55bc47c7780838a2ad5278f0cb2a63dc5679c7b7a4`

Scanned documents:

- FineWeb: 379,247
- Wikimedia: 20,000

Candidate dispositions:

| Source | Dimension | Candidate kind | Count | Disposition |
| --- | --- | --- | ---: | --- |
| FineWeb | arithmetic | exact | 2,846 | quarantine |
| FineWeb | factuality | exact | 1 | quarantine |
| FineWeb | manual_review | semantic | 9 | quarantine |
| FineWeb | manual_review | semantic | 214 | accept low-risk generic or partial overlap |
| FineWeb | robustness | semantic | 90 | accept low-risk generic or partial overlap |
| Wikimedia | all reviewed dimensions | exact/semantic | 0 | no candidates |

The complete non-plaintext evidence is recorded in
`evaluation/results/vasu_140m_base_v2_semantic_contamination_independent_review_20260803.json`.
Its result SHA-256 is
`8c58321d1e129b60f96bea079d7503cda34106de61f47dcfa482640cfa0d4b34`.

## Compatibility Conclusion

The development inventories and sealed held-out inventories are structurally
compatible with the existing evaluation-v2 inventory contract, and the sealed
candidate files remain decryptable only through the existing Age boundary.
However, the matrix is not acceptable for source-admission review because
answer isolation and contamination quarantine requirements are incomplete.

Likelihood inventories remain deferred until source admission and must be
created only after document-level train exclusions are known.

## Non-Authorization

This rejection does not authorize likelihood inventory construction, production
suite freezing, source admission, data-release construction, model execution,
checkpoint access or creation, evaluation execution, optimizer updates,
authorization records, training configuration, or training.
