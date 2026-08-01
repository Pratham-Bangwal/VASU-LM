# VASU-140M Evaluation-v2 Inventory Post-Commit Identity Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-02

Reviewer: GPT-5.5 independent review

## Decision

Accept the clean post-commit identity of the VASU-140M evaluation-v2 inventory
contract at commit `858c8e9d1a8a977d2205065b3157e73ad34c3cf6`.

Acceptance covers only the lineage-preserving identity transition from the
accepted pre-commit package at parent
`f2dbda855a4255c06ad082dd49c62cb0adab4e11` to the isolated inventory-contract
commit. It does not authorize prompt authoring, held-out opening, suite
freezing, model execution, checkpoint access, evaluation publication, or
training.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The main repository worktree contained unrelated concurrent author-side
  modified and untracked packages before review. This was not treated as a
  blocker because all identity and behavior checks were performed in a clean
  detached worktree at the exact reviewed commit, and the main worktree was
  modified only by creating this decision document.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_POSTCOMMIT_IDENTITY_REVIEW_PACKET.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_POSTCOMMIT_IDENTITY_AUDIT_20260801.md`
- `.gitattributes`
- `evaluation/framework/vasu_140m_base_v2_inventory.py`
- `tests/test_vasu_140m_base_v2_inventory.py`
- `scripts/smoke_vasu_140m_base_v2_inventory.py`
- `evaluation/fixtures/vasu_140m_base_v2_inventory_qualification_v1.json`
- `evaluation/fixtures/vasu_140m_base_v2_inventory_qualification_postcommit_858c8e9.json`

## Commands and Exact Results

- `git show --format=%P -s 858c8e9d1a8a977d2205065b3157e73ad34c3cf6`
  returned `f2dbda855a4255c06ad082dd49c62cb0adab4e11`; the accepted package
  base is the direct parent of the reviewed commit.
- `git worktree add --detach
  C:\Users\acer\AppData\Local\Temp\vasu_inventory_postcommit_858c8e9_review
  858c8e9d1a8a977d2205065b3157e73ad34c3cf6` created a detached checkout at the
  exact reviewed commit.
- In the detached checkout, `git rev-parse HEAD` returned
  `858c8e9d1a8a977d2205065b3157e73ad34c3cf6`.
- In the detached checkout, `git status --short` was empty before validation.
- In the detached checkout, `git config --show-origin --get core.autocrlf`
  returned `file:C:/Program Files/Git/etc/gitconfig true`, confirming default
  Windows line-ending checkout behavior.
- `python -m pytest tests\test_repository_line_endings.py
  tests\test_vasu_140m_base_v2_inventory.py
  tests\test_vasu_140m_base_v2_scoring.py
  tests\test_vasu_140m_base_evaluation_v2.py
  tests\test_capability_framework.py tests\test_evaluation_metrics.py
  tests\test_evaluation_comparison.py -q` passed:
  `129 passed in 14.80s`.
- `python -m ruff check evaluation\framework\vasu_140m_base_v2_inventory.py
  tests\test_vasu_140m_base_v2_inventory.py
  scripts\smoke_vasu_140m_base_v2_inventory.py` passed:
  `All checks passed!`.
- Post-commit fixture comparison:
  `python scripts\smoke_vasu_140m_base_v2_inventory.py | ConvertFrom-Json |
  ConvertTo-Json -Depth 100 -Compress` matched
  `evaluation\fixtures\vasu_140m_base_v2_inventory_qualification_postcommit_858c8e9.json`
  exactly.
- `python scripts\smoke_vasu_140m_base_v2_inventory.py | python -m json.tool`
  emitted the expected post-commit report with
  `qualification_sha256=5a17ef13f059ba39bc2c4045d8f176585e52b7abc08e64e360bf8950fdef1e76`.
- Independent fixture-delta script reported exactly:
  `changed_top_level_fields=
  development_inventory_sha256s,held_out_inventory_sha256s,qualification_sha256,repository_commit`;
  `development_changes=6`; `held_out_changes=6`;
  `manifest_identity_deltas_from_repository_commit_only=12`.
- In the detached checkout, `git diff --check` produced no output.
- In the detached checkout, final `git status --short` was empty.
- `git worktree remove --force
  C:\Users\acer\AppData\Local\Temp\vasu_inventory_postcommit_858c8e9_review`
  removed the temporary worktree; `removed=True`.

## Verified Identities

- Reviewed commit:
  `858c8e9d1a8a977d2205065b3157e73ad34c3cf6`.
- Direct parent:
  `f2dbda855a4255c06ad082dd49c62cb0adab4e11`.
- Implementation SHA-256:
  `1ead769715f72faa13be981bf92199d81db30952bc17a02812d51c8055b4905a`.
- Tests SHA-256:
  `0ddb706517a313ddaf2d0a6d8b93c89352357ef1175df2fa157b618c372db606`.
- Smoke SHA-256:
  `de52fe868e6ad0a78ed0d7cca9293ab504dc5cf6e16de037ef4b63ed9adfd44d`.
- Accepted pre-commit fixture SHA-256:
  `4dcdfaf5b04d4864e00d0244dde178135b987c37bc1a923280e2133ef7111dc5`.
- Frozen post-commit fixture SHA-256:
  `aad4c03bdefde2afd00792fbe1adeafadf6bba9a6f8f5c0f3f8a9d40a123ce8e`.
- Pre-commit qualification SHA-256:
  `02f7461c3398f9643e8115cf341691e8107c141858d40532666ec243402248d7`.
- Post-commit qualification SHA-256:
  `5a17ef13f059ba39bc2c4045d8f176585e52b7abc08e64e360bf8950fdef1e76`.

## Exact Pre/Post Identity Comparison

Only these top-level fields differ between the accepted pre-commit fixture and
the post-commit fixture:

- `repository_commit`:
  `f2dbda855a4255c06ad082dd49c62cb0adab4e11` ->
  `858c8e9d1a8a977d2205065b3157e73ad34c3cf6`.
- `development_inventory_sha256s`: all six dimension identities changed.
- `held_out_inventory_sha256s`: all six dimension identities changed.
- `qualification_sha256`:
  `02f7461c3398f9643e8115cf341691e8107c141858d40532666ec243402248d7` ->
  `5a17ef13f059ba39bc2c4045d8f176585e52b7abc08e64e360bf8950fdef1e76`.

For each of the twelve generated inventory manifests, removing
`inventory_sha256` and `repository_commit` left pre-commit and post-commit
manifests identical. The twelve inventory identity changes therefore follow
solely from the repository commit embedded in each manifest.

Post-commit development inventory identities:

- arithmetic:
  `0be06d08facb9c9d425947e9600294defc93aed38e773e7ae80cab29e0b320e5`
- factuality:
  `5f69dd942f83e563e4bdc769543bc230ee80f3aee7c17a90d7ed28fa8f40328d`
- likelihood:
  `228964de231bbcf40dd5187685517a19a2885d72faf43416adf86eaf7bdf1c0e`
- manual_review:
  `2a2b3cf888dbf9c4896f17145a9a097b86c5ba3fafb6bcd38552bc1ae812da87`
- repetition:
  `e5607a3d1cf5e990f2cace27d6651de43ca2bd4561bca9637be1cf1843714f9e`
- robustness:
  `d04e3495045ac0ef9363f5c78966304c56bc5257e3e32d8b130f034eea0e25b1`

Post-commit held-out inventory identities:

- arithmetic:
  `ecf8322ef1d9a6b4a16be0394b2b8fff12daab75e8ac0dc4c940812af6f64ab9`
- factuality:
  `117b961872d9cb2d4509d2fe5d39e53057e1f66861a2c5a4da115f5e230a83dd`
- likelihood:
  `e9cdaf218accea9d1c1a6a68bd11ef6bb8b8f82dbfdfbdba26a5c563ac1c2def`
- manual_review:
  `2dbf95e7eccd7fb34ea998286d0611d969cbf0ceef2b03e51f341a39897f6aca`
- repetition:
  `289d32c899d2c6d4a2426cb5a3c6f5c38e17a2893ee340e354da65170a55a7a1`
- robustness:
  `2f315fd89de563ad647ba95a71e50783056997b3134e784806e0c2c591ff22ac`

## Held-Out Access Behavior

The post-commit smoke validated six development payloads and six sealed
held-out payloads. Held-out validation checked the Age v1 header and public
manifest commitments without decrypting or opening payload content. Six
explicit held-out opening attempts were rejected, and the report records
`held_out_opened=false`.

The reproduced report also records `prompt_content_persisted=false`,
`temporary_artifacts_cleaned=true`, `production_suite_frozen=false`,
`evaluation_run_authorized=false`, and `training_authorized=false`.

## Fixture Reproducibility

The post-commit fixture reproduced exactly from the clean detached checkout.
The frozen post-commit fixture SHA-256 is
`aad4c03bdefde2afd00792fbe1adeafadf6bba9a6f8f5c0f3f8a9d40a123ce8e`, and the
reproduced qualification identity is
`5a17ef13f059ba39bc2c4045d8f176585e52b7abc08e64e360bf8950fdef1e76`.

## Compatibility Assessment

The post-commit identity transition is compatible with existing model
architecture, tokenizer assets, datasets, masks, checkpoints,
optimizer/scheduler state, training configurations, schedules, existing
evaluation outputs, and exact-resume behavior. The package is additive and
defines inventory validation evidence only.

## Non-Authorization

This decision is non-authorizing. It does not authorize real prompt authoring,
encryption or key creation, held-out opening, suite freezing, source
acquisition, data construction, model execution, checkpoint creation or
access, evaluation publication, optimizer updates, authorization records,
training configuration, schedules, commits, pushes, or training.
