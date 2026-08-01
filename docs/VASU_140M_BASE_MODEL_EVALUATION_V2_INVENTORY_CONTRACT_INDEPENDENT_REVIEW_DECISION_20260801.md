# VASU-140M Base-Model Evaluation v2 Inventory Contract Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the rebased VASU-140M evaluation-v2 inventory contract for a later
isolated inventory-contract commit and clean post-commit identity review.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree with unrelated
  modified files and many untracked VASU-140M packages. This was not treated
  as a blocker because the inventory contract's identities, fixture replay,
  and validation commands are self-contained.
- `git diff --check` reported only known line-ending warnings for existing
  modified documentation files and no whitespace errors.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/REPOSITORY_LINE_ENDING_REPRODUCIBILITY_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_postcommit_92656ac.json`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_REVIEW_PACKET.md`
- `evaluation/framework/vasu_140m_base_v2.py`
- `evaluation/framework/vasu_140m_base_v2_tasks.py`
- `evaluation/framework/vasu_140m_base_v2_statistics.py`
- `evaluation/framework/vasu_140m_base_v2_inventory.py`
- `tests/test_vasu_140m_base_v2_inventory.py`
- `scripts/smoke_vasu_140m_base_v2_inventory.py`
- `evaluation/fixtures/vasu_140m_base_v2_inventory_qualification_v1.json`

## Rationale

The contract correctly separates exact payload identity from contamination
identity. `content_text_sha256` hashes NFC-normalized content while preserving
meaningful whitespace; `normalized_text_sha256` NFC-normalizes, collapses
whitespace, and case-folds text for contamination scanning. Exact word-span
counts are stored with prompt and accepted-answer commitments, and eight-word
rolling fragments are required for contamination commitments.

Development inventories reconcile plaintext payloads back to task,
provenance, and contamination commitments. Likelihood payloads preserve
explicit context and target boundaries, factuality preserves choice identity
and order, and all six evaluation dimensions have development payload
coverage. The validator reconciles repository-relative artifact paths,
SHA-256, byte counts, record counts, item IDs, scorer identity, tokenizer
identity, split, access policy, provenance IDs, contamination IDs, and task
commitments.

Public provenance records require source name, URL, license name and URL,
source revision, citation, parent document, retrieval timestamp with timezone,
and authorship metadata. Contamination records are hash-only and reject
plaintext-like invalid hashes, duplicate commitments, too-short n-gram spans,
zero word counts, and substituted payload-derived commitments.

Held-out fixture payloads remain opaque Age-style bytes. The validator checks
only the Age v1 header and refuses `open_held_out=True` with a
`PermissionError`. The smoke validates six sealed held-out payloads without
decryption and records six rejected opening attempts.

The package is contract evidence only. It creates temporary synthetic
development and held-out fixture inventories, validates them, removes them,
and leaves no production inventory, real prompt, encryption key, held-out
plaintext, suite freeze, model/checkpoint execution, data acquisition,
configuration, schedule, optimizer, authorization record, checkpoint, or
training route.

## Commands and Exact Results

- `git status --short`
  showed the expected concurrent author-side modified files and untracked
  packages while preserving them.
- `git rev-parse HEAD`
  returned `f2dbda855a4255c06ad082dd49c62cb0adab4e11`.
- `git merge-base --is-ancestor f2dbda855a4255c06ad082dd49c62cb0adab4e11 HEAD`
  returned exit code 0; `base_is_ancestor=true`.
- `python -m pytest tests\test_vasu_140m_base_v2_inventory.py tests\test_vasu_140m_base_v2_scoring.py tests\test_vasu_140m_base_evaluation_v2.py tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q`
  passed: `126 passed in 3.15s`.
- `python -m ruff check evaluation\framework\vasu_140m_base_v2_inventory.py tests\test_vasu_140m_base_v2_inventory.py scripts\smoke_vasu_140m_base_v2_inventory.py`
  passed: `All checks passed!`.
- Frozen fixture replay:
  `$observed = (python scripts\smoke_vasu_140m_base_v2_inventory.py | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)`;
  `$expected = (Get-Content evaluation\fixtures\vasu_140m_base_v2_inventory_qualification_v1.json -Raw -Encoding utf8 | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)`;
  `if ($observed -ne $expected) { throw "inventory qualification fixture mismatch" }`
  passed: `inventory qualification fixture matched`.
- `git diff --check`
  completed with line-ending warnings only for existing modified
  documentation and no whitespace errors.

## Verified Identities

- Package base commit:
  `f2dbda855a4255c06ad082dd49c62cb0adab4e11`.
- Inventory implementation SHA-256:
  `1ead769715f72faa13be981bf92199d81db30952bc17a02812d51c8055b4905a`.
- Tests SHA-256:
  `0ddb706517a313ddaf2d0a6d8b93c89352357ef1175df2fa157b618c372db606`.
- Smoke SHA-256:
  `de52fe868e6ad0a78ed0d7cca9293ab504dc5cf6e16de037ef4b63ed9adfd44d`.
- Frozen fixture SHA-256:
  `4dcdfaf5b04d4864e00d0244dde178135b987c37bc1a923280e2133ef7111dc5`.
- Qualification SHA-256:
  `02f7461c3398f9643e8115cf341691e8107c141858d40532666ec243402248d7`.
- Development inventory identities:
  arithmetic `e99dfd13aa656c85d029490ee65504fa92a652b3b202e653f2e6de28c643869c`,
  factuality `4d1c29361fd30736ba4b41cd066046a7a121f24e07b5d9af35e7db443f4a3cdd`,
  likelihood `b811727c2d4c7c1873608b647f9dfd60f9ad645739d9911c4d441573e97122dd`,
  manual_review `17bbaf80f82b5da01fbdbc17c275564153629d19baa7b8da8b2793a7a8e54052`,
  repetition `d9a36a5b3d8ab07872e10d779ffe2a4ab14670e2d5225b4e8856860eef370417`,
  robustness `3447ac1d3082edeee5f9500a13e923d5042239c8efcb6bb1ebf5690b8b7cb993`.
- Held-out inventory identities:
  arithmetic `fcf3ec8783f1da7d4e86d595d6391ccf23a5bc62e7c2f6a258ca5075d2f10c88`,
  factuality `6114cbf5c868eb185376916927978302fa1a8b8084b50bfc2601a6ea04ddf44f`,
  likelihood `ba507ec2d7fa862cdeacfbcf5b71347bded42dce1998a0bfd598a089f6057004`,
  manual_review `0e14078123d9c77863d31d86016c58a84a74b4b8f6ba7eab79e75a241d00b178`,
  repetition `51522c3b1dd489ebda61397d97f43b39fb7584a677d593b075d137563041f399`,
  robustness `1354e56ceda81ead64bd43d71dae6e8ccf117777e55c0445341afc31d1b39478`.

## Fixture Reproducibility

The frozen inventory fixture reproduced exactly. The reproduced smoke reported
`dimension_count=6`, `development_payloads_validated=6`,
`sealed_payloads_validated_without_decryption=6`,
`held_out_opening_rejections=6`, `fixture_only=true`,
`prompt_content_persisted=false`, `held_out_opened=false`,
`production_suite_frozen=false`, `evaluation_run_authorized=false`,
`training_authorized=false`, and `temporary_artifacts_cleaned=true`.

## Held-Out Access Behavior

Held-out manifests require `state=sealed`, `payload_format=encrypted_jsonl`,
`encryption_algorithm=age-x25519`, a recipient fingerprint, and
`opening_authorized=false`. File validation checks the Age v1 header only and
does not decrypt or parse held-out payload content. Attempts to validate with
`open_held_out=True` fail closed with `PermissionError`, and the smoke records
one rejected opening attempt for each of the six held-out fixture inventories.

## Compatibility Assessment

The implementation is additive. It does not change model architecture,
tokenizer semantics, datasets, masks, checkpoints, optimizer or scheduler
state, training configurations, schedules, existing evaluation outputs, or
exact-resume behavior. It defines inventory packaging and validation contracts
for future separately reviewed suite-freezing and contamination gates.

## Non-Authorization

This acceptance authorizes only a later isolated inventory-contract commit and
clean post-commit identity review. It does not authorize production prompt
authoring, encryption or key creation, held-out opening, suite freezing,
source acquisition, data construction, model execution, checkpoint access or
creation, evaluation publication, configuration creation, schedule creation,
optimizer updates, authorization records, training, commit, or push.
