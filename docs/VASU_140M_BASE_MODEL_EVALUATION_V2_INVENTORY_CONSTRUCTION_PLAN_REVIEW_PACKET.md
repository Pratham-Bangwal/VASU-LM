# VASU-140M Base Evaluation v2 Inventory Construction Plan Review Packet

Status: **independent GPT-5.5 review requested; non-authorizing.**

## Requested Decision

Accept or reject this construction plan's scientific coverage, exact accepted
dependency bindings, source roles, split policy, likelihood ordering,
contamination policy, and held-out boundary.

Acceptance does not authorize construction or key creation.

## Required Evidence

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
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
- all three source-catalog artifacts bound by the plan

## Review Questions

1. Are counts large enough for the declared per-dimension statistics while
   remaining practical on local hardware?
2. Does 512 likelihood records per admitted source avoid mixture-share
   masking, and is post-acquisition whole-document isolation ordered correctly?
3. Do the ten prompt inventories exist conceptually before source admission
   without creating a circular dependency?
4. Are factual, arithmetic, topic-seed, and held-out derivation roles honest?
5. Are instruction prompts prevented from leaking into the base-model suite?
6. Are semantic-family, parent-document, and derived-variant splits isolated?
7. Is the unkeyed Age boundary fail-closed and genuinely non-authorizing?
8. Does the validator bind every current source and implementation byte?
9. Does the plan bind the exact four accepted scoring/inventory decisions and
   fail when any dependency is missing, renamed, substituted, or mutated?

## Validation

```powershell
python -m pytest tests\test_vasu_140m_base_v2_inventory_plan.py -q
python -m ruff check evaluation\framework\vasu_140m_base_v2_inventory_plan.py tests\test_vasu_140m_base_v2_inventory_plan.py scripts\smoke_vasu_140m_base_v2_inventory_plan.py
$observed = python scripts\smoke_vasu_140m_base_v2_inventory_plan.py | ConvertFrom-Json
$frozen = Get-Content evaluation\fixtures\vasu_140m_base_v2_inventory_plan_qualification_v1.json -Raw | ConvertFrom-Json
if (($observed | ConvertTo-Json -Depth 20 -Compress) -ne ($frozen | ConvertTo-Json -Depth 20 -Compress)) { throw 'fixture mismatch' }
git diff --check
git status --short
```

## Decision Output

Create only:

`docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONSTRUCTION_PLAN_INDEPENDENT_REVIEW_DECISION_20260801.md`

Report decision, findings by severity, evidence, exact validation, frozen
identity, count/coverage assessment, dependency status, compatibility, and
explicit non-authorization. Do not create prompts or keys, modify code or
fixtures, run a model, commit, or push.
