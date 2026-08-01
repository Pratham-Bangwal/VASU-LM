# VASU-140M Base-Model Evaluation v2 Inventory Contract Review Packet

Review only the inventory contract and temporary fixture qualification. Do not
interpret acceptance as approval of any real prompt, provenance source,
held-out encryption, production inventory, model evaluation, or training.

Read completely:

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
- `evaluation/framework/vasu_140m_base_v2.py`
- `evaluation/framework/vasu_140m_base_v2_tasks.py`
- `evaluation/framework/vasu_140m_base_v2_statistics.py`
- `evaluation/framework/vasu_140m_base_v2_inventory.py`
- `tests/test_vasu_140m_base_v2_inventory.py`
- `scripts/smoke_vasu_140m_base_v2_inventory.py`
- `evaluation/fixtures/vasu_140m_base_v2_inventory_qualification_v1.json`

Confirm the exact identities in the audit and verify content versus
contamination normalization, case/whitespace-robust identity, likelihood target
boundaries, factual choice order, provenance requirements, multiple prompt
and accepted-answer commitments with exact scan-span word counts, eight-word
fragment commitments, development plaintext-to-commitment reconciliation,
exact byte/record/ID
reconciliation, production-candidate structural support, development/held-out
separation, Age-header-only scope, temporary cleanup, compatibility, and every
non-authorization flag.

Run:

```powershell
python -m pytest tests\test_vasu_140m_base_v2_inventory.py tests\test_vasu_140m_base_v2_scoring.py tests\test_vasu_140m_base_evaluation_v2.py tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q
python -m ruff check evaluation\framework\vasu_140m_base_v2_inventory.py tests\test_vasu_140m_base_v2_inventory.py scripts\smoke_vasu_140m_base_v2_inventory.py
$observed = (python scripts\smoke_vasu_140m_base_v2_inventory.py | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)
$expected = (Get-Content evaluation\fixtures\vasu_140m_base_v2_inventory_qualification_v1.json -Raw -Encoding utf8 | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)
if ($observed -ne $expected) { throw "inventory qualification fixture mismatch" }
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_INDEPENDENT_REVIEW_DECISION_20260801.md`.

Acceptance authorizes only an isolated commit and later clean post-commit
identity review. It does not authorize real prompt authoring, real held-out
encryption or opening, production inventory creation, suite freezing, model or
checkpoint execution, evaluation publication, source admission/acquisition,
data construction, configurations, schedules, optimizers, authorization
records, checkpoints, or training.
