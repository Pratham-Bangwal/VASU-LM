# VASU-140M Base-Model Evaluation v2 Inventory Contract Audit — 2026-08-01

Status: pre-commit fixture qualification evidence; non-authorizing.

## Root cause

Source-specific admission cannot honestly finish until every future evaluation
prompt has an immutable identity that training-data contamination scans can
consume. The accepted evaluation schema and scorer qualification did not yet
define how prompt payloads, expected answers, provenance, public contamination
commitments, or sealed held-out bytes are packaged. Treating a design document
or scorer fixture as that inventory would leave the production data gate
scientifically incomplete.

## Implementation

`evaluation/framework/vasu_140m_base_v2_inventory.py` adds one contract for
both fixture and future production-candidate inventories:

- exact family-compatible tokenizer, repository, suite, dimension, split,
  interface, scorer, generation-profile, artifact, and item commitments;
- NFC canonical content hashes that preserve meaningful whitespace, separately
  from NFC, whitespace-collapsed, case-folded contamination identities so
  capitalization and spacing changes cannot bypass an exact scan;
- plaintext development payload validation for likelihood, factuality,
  arithmetic, repetition, robustness, and manual-review tasks;
- explicit likelihood context/target boundaries and factual choice order;
- public provenance records with source, license, revision, citation, parent
  document, retrieval time, and authorship metadata;
- public hash-only contamination records supporting prompt and accepted-answer
  identities with exact word-span lengths, eight-word fragments, semantic
  fingerprints, and parent-document isolation;
- development-side reconciliation proving every prompt, accepted-answer, and
  rolling fragment commitment was derived from the bound plaintext payload,
  preventing a formally valid but unscannable or substituted commitment index;
- exact SHA-256, byte-count, record-count, unique-ID, task, and provenance
  reconciliation across payload, provenance, contamination, and commitments;
- plaintext development versus Age-X25519 sealed held-out access contracts;
- Age v1 header verification without decrypting or opening held-out content;
- repository path containment and immutable file verification; and
- explicit false suite-freeze, evaluation-run, and training authority.

The validator supports `fixture_only=false` production candidates, but no
inventory can itself claim that a suite is frozen or evaluation/training is
authorized. Those decisions remain separate hash-bound review gates.

## Fixture qualification

The smoke creates, validates, and removes six development and six opaque
held-out fixture inventories in system-temporary storage. Every evaluation
dimension is covered. All six attempted held-out openings are rejected. No
fixture prompt or opaque payload persists after the smoke.

### Exact identities

- Parent commit: `f2dbda855a4255c06ad082dd49c62cb0adab4e11`
- Implementation SHA-256:
  `1ead769715f72faa13be981bf92199d81db30952bc17a02812d51c8055b4905a`
- Tests SHA-256:
  `0ddb706517a313ddaf2d0a6d8b93c89352357ef1175df2fa157b618c372db606`
- Smoke SHA-256:
  `de52fe868e6ad0a78ed0d7cca9293ab504dc5cf6e16de037ef4b63ed9adfd44d`
- Frozen fixture SHA-256:
  `4dcdfaf5b04d4864e00d0244dde178135b987c37bc1a923280e2133ef7111dc5`
- Qualification SHA-256:
  `02f7461c3398f9643e8115cf341691e8107c141858d40532666ec243402248d7`

## Validation

- 30 focused inventory tests passed.
- Ruff passed for the implementation, tests, and smoke.
- The smoke reproduced the frozen fixture exactly.
- Inventory, scorer, schema, capability-framework, metrics, and comparison
  regressions passed together: 126 passed in 3.66 seconds on the accepted
  scoring-closeout lineage at `f2dbda8`.
- `git diff --check` must have no whitespace error; concurrent documentation
  may continue to show known CRLF conversion warnings.

## Remaining gates

The fixture validates the contract, not real encryption or prompt quality. A
future independent author/reviewer must create source-attributed development
content and genuinely Age-encrypted held-out content under a separate key.
Cryptographic encryption execution, recipient/key custody, semantic-duplicate
review, production contamination indexes, inventory acceptance decisions,
suite assembly, clean post-commit identity, and any model execution remain
separate gates.

## Compatibility and non-authorization

This implementation is additive. Existing model architecture, tokenizer,
datasets, masks, checkpoints, evaluation outputs, training configurations,
schedules, optimizer/scheduler state, and exact-resume behavior are unchanged.

No production inventory, protected prompt, held-out plaintext, suite manifest,
model or checkpoint execution, data acquisition, data release, configuration,
schedule, optimizer, authorization record, checkpoint, or training run was
created or authorized.
