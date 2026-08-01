# VASU-140M Base-Model Evaluation v2 Inventory Construction Plan Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-02

Reviewer: GPT-5.5 independent review

## Decision

Accept the hardened VASU-140M evaluation-v2 inventory construction plan for a
later isolated construction-plan commit and clean post-commit identity review.

Acceptance covers only the non-authorizing plan and validator evidence. It
does not authorize prompt construction, held-out key creation, held-out
opening, source acquisition, data construction, model execution, checkpoint
access or creation, evaluation publication, optimizer updates, authorization
records, or training.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree containing
  unrelated modified files and untracked VASU-140M packages. This was not a
  blocker because the construction plan, validator, fixture replay, file
  hashes, and dependency checks are self-contained. The unrelated files were
  preserved.
- `git diff --check` reported only existing CRLF conversion warnings for
  modified documentation files and no whitespace errors.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONSTRUCTION_PLAN_REVIEW_PACKET.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONSTRUCTION_PLAN_AUDIT_20260801.md`
- `configs/evaluation/vasu_140m_base_v2_inventory_construction_plan.json`
- `evaluation/framework/vasu_140m_base_v2_inventory_plan.py`
- `tests/test_vasu_140m_base_v2_inventory_plan.py`
- `scripts/smoke_vasu_140m_base_v2_inventory_plan.py`
- `evaluation/fixtures/vasu_140m_base_v2_inventory_plan_qualification_v1.json`
- `evaluation/benchmarks/factual_cpt_v2.json`
- `vasu/data/arithmetic_v2.py`
- `evaluation/benchmarks/ultrachat_promotion_v1.json`

## Commands and Exact Results

- `git status --short` showed unrelated concurrent author-side modified and
  untracked files, including the reviewed untracked construction-plan package.
- `git rev-parse HEAD` returned
  `216013a9175a5cdff165765e60641ce5b1dae8c5`.
- `git merge-base --is-ancestor 216013a9175a5cdff165765e60641ce5b1dae8c5 HEAD`
  returned exit code 0; `base_is_ancestor=true`.
- `python -m pytest tests\test_vasu_140m_base_v2_inventory_plan.py -q`
  passed: `18 passed in 0.47s`.
- `python -m ruff check evaluation\framework\vasu_140m_base_v2_inventory_plan.py
  tests\test_vasu_140m_base_v2_inventory_plan.py
  scripts\smoke_vasu_140m_base_v2_inventory_plan.py`
  passed: `All checks passed!`.
- Frozen fixture replay:
  `$observed = python scripts\smoke_vasu_140m_base_v2_inventory_plan.py |
  ConvertFrom-Json`;
  `$frozen = Get-Content
  evaluation\fixtures\vasu_140m_base_v2_inventory_plan_qualification_v1.json
  -Raw | ConvertFrom-Json`;
  `if (($observed | ConvertTo-Json -Depth 20 -Compress) -ne ($frozen |
  ConvertTo-Json -Depth 20 -Compress)) { throw 'fixture mismatch' }`
  passed: `inventory plan fixture matched`.
- `python scripts\smoke_vasu_140m_base_v2_inventory_plan.py | python -m json.tool`
  produced valid JSON with `accepted_dependencies_bound=4`,
  `plan_sha256=4bd36c7d7b862e782614347f9c837e3fbaf12c7eaff0d80b7df146c6d71db784`,
  `likelihood_count_scope=per_admitted_source`,
  `prompt_inventory_count_required_before_source_admission=10`,
  `held_out_key_created=false`, `held_out_content_created=false`,
  `construction_authorized=false`, `evaluation_run_authorized=false`, and
  `training_authorized=false`.
- Independent dependency adversarial checks in temporary directories:
  missing decision document rejected as missing bound artifact; renamed
  decision document rejected as missing bound artifact; substituted decision
  document rejected as identity mismatch; mutated decision document rejected
  as identity mismatch.
- Package route scan with `rg` found only validation of false authorization
  flags and held-out security fields. It found no hidden construction,
  encryption-key creation, held-out opening, source acquisition, model
  execution, checkpoint operation, optimizer, or training route.
- `git diff --check` completed with CRLF conversion warnings only for
  `docs/CHANGELOG.md`, `docs/PROJECT_STATUS.md`, and `docs/ROADMAP.md`.

## Verified Hashes and Plan Identity

- Base commit:
  `216013a9175a5cdff165765e60641ce5b1dae8c5`.
- Plan identity:
  `4bd36c7d7b862e782614347f9c837e3fbaf12c7eaff0d80b7df146c6d71db784`.
- Plan file SHA-256:
  `9efd4b18c4b606d1d9d10a0028ac7f42c1d111522350951fffb54a0d926f6662`.
- Validator SHA-256:
  `c95c883ab318f5ad79e7208039ad37e03860b082a928471554cac229d12139b6`.
- Tests SHA-256:
  `a136f45319d49baf32efdab9b56e9880923fb0f8581421fada033fb9993cb888`.
- Smoke SHA-256:
  `6052d498d69424086adf4c06f4980add83a79bff5f82c5c405b426e518716b9f`.
- Fixture SHA-256:
  `a8ea22d607421a2d9fb9434523ba32243664dae9fc125bdc9ae6538aea038190`.
- Tokenizer SHA-256:
  `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a`.
- Inventory contract implementation SHA-256:
  `1ead769715f72faa13be981bf92199d81db30952bc17a02812d51c8055b4905a`.
- Task scorer implementation SHA-256:
  `7777bd7d6a488c21e3937b5b05e1e46153e7fddbcb67784c47dc9e53a5087bf5`.
- `evaluation/benchmarks/factual_cpt_v2.json` SHA-256:
  `17c356f0b3a5093511402b08f307b770b8cc4443b3ca3e1cceaa0eb7044115b9`.
- `vasu/data/arithmetic_v2.py` SHA-256:
  `197ba0f190ca2ce3098196edd639c5b3bc8007e8a6863e040011b519dcdde563`.
- `evaluation/benchmarks/ultrachat_promotion_v1.json` SHA-256:
  `68806800c0b8d524a1cdd4e83ccdd46d388aa9d86f8070a6fbbf3e04a8091bf1`.

## Dependency-Binding Assessment

The plan binds the four accepted decisions by exact path and SHA-256:

- Scoring qualification decision:
  `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_INDEPENDENT_REVIEW_DECISION_20260801.md`,
  `32450e710fabb27416dd466e6f05745283aae1b72fc0a7786a59116848d8139a`.
- Scoring post-commit identity decision:
  `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260801.md`,
  `6e4c2d2263c243772e5b66f49a44fd5d3eac8d7149947d6720d9005b3e7052d6`.
- Inventory-contract qualification decision:
  `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_INDEPENDENT_REVIEW_DECISION_20260801.md`,
  `110e8f5d8d3f3883995fb0ddd9479f90fd3a255b626b8aa53d354b67c1b506f7`.
- Inventory post-commit identity decision:
  `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260801.md`,
  `eb9a7512151df78ad11b7269e5366aa1edc3cece205d502928802f04ee83e2bd`.

The validator requires exactly these four dependency keys, repository-relative
Markdown decision paths under `docs/`, and exact file hashes. Temporary
adversarial checks confirmed missing, renamed, substituted, and mutated
decision documents fail validation before any construction stage.

## Count and Coverage Assessment

The plan covers all six VASU-140M base-evaluation v2 dimensions:

- likelihood: 512 development and 512 held-out records per admitted source;
- factuality: 200 development and 200 held-out records total;
- arithmetic: 1,000 development and 1,000 held-out records total;
- repetition: 120 development and 120 held-out records total;
- robustness: 120 development and 120 held-out pairs total;
- manual review: 60 development and 60 held-out records total.

Likelihood is correctly scoped per admitted source and deferred until
post-acquisition whole-document holdout, avoiding mixture-share masking.
The five prompt dimensions require ten prompt inventories before source
admission while staying independent from likelihood source acquisition. The
split policy requires semantic-family isolation, parent-document isolation,
development-before-held-out ordering, and no cross-split derivation. The
contamination policy requires prompt and answer hashes, eight-word fragments,
semantic review, and scanning before data splitting.

The source roles are appropriately constrained:

- `factual-cpt-v2-development-seed` is development seed/fact-family audit
  material only and cannot derive held-out prompts.
- `verified-arithmetic-v2-generator` can generate arithmetic prompts, with
  held-out derivation permitted only under disjoint family/range policy.
- `ultrachat-promotion-topic-seed` is topic/stratum seed material only;
  `prompt_reuse_permitted=false` and `held_out_derivation_permitted=false`
  prevent instruction prompts from being reused as base-model evaluation
  prompts or held-out derivation material.

## Held-Out Boundary Assessment

Held-out security is fail-closed and non-authorizing. The plan names
`age-x25519` as the future encryption algorithm but binds
`recipient_fingerprint=null`, `plaintext_repository_path_forbidden=true`,
`independent_curator_required=true`, `opening_authorized=false`, and
`key_creation_authorized=false`. The smoke fixture records that no held-out key
or held-out content was created.

## Compatibility

The package is additive. It changes no model architecture, tokenizer asset,
dataset, mask, checkpoint, optimizer or scheduler state, training
configuration, schedule, existing evaluation output, or exact-resume behavior.
It records a construction plan and validator only.

## Non-Authorization

This acceptance is non-authorizing. It may authorize only a later isolated
construction-plan commit and clean post-commit identity review. It does not
authorize prompt construction, held-out encryption-key creation, held-out
opening, source acquisition, data construction, model execution, checkpoint
creation or access, evaluation publication, optimizer updates, authorization
records, training configuration, schedules, commits, pushes, or training.
