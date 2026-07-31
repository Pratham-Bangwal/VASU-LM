# VASU-140M Base-Model Evaluation v2 Schema Remediation Review Packet

Review the current remediated schema identity without overwriting the earlier
implementation decision or either frozen fixture.

Read:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_IMPLEMENTATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_REMEDIATION_AUDIT_20260801.md`
- `evaluation/framework/vasu_140m_base_v2.py`
- `tests/test_vasu_140m_base_evaluation_v2.py`
- `scripts/smoke_vasu_140m_base_evaluation_v2_schema.py`
- `evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_qualification_rejected_2d73fcb.json`
- `evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_qualification_precommit.json`

Confirm that the rejected `2d73fcb` fixture remains separate, the existing
implementation decision binds only its stated earlier identity, the current
fixture reproduces at commit `51943a7a326b836755b2edaa7b7fd26ba0da366d`, and
the result schema fails closed on dimension-mode
mismatch, stage/split mismatch, incomplete greedy/sampled coverage, unbound
held-out access, result-to-suite mismatch, resume-identity mutation, aggregate
scores, and training authorization.

Run:

```powershell
python -m pytest tests\test_vasu_140m_base_evaluation_v2.py tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q
python -m ruff check evaluation\framework\vasu_140m_base_v2.py tests\test_vasu_140m_base_evaluation_v2.py scripts\smoke_vasu_140m_base_evaluation_v2_schema.py
$observed = (python scripts\smoke_vasu_140m_base_evaluation_v2_schema.py | ConvertFrom-Json | ConvertTo-Json -Depth 20 -Compress)
$expected = (Get-Content evaluation\fixtures\vasu_140m_base_evaluation_v2_schema_qualification_precommit.json -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 20 -Compress)
if ($observed -ne $expected) { throw "remediated qualification fixture mismatch" }
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_REMEDIATION_INDEPENDENT_REVIEW_DECISION_20260801.md`.

Acceptance authorizes only committing this remediated schema package and later
post-commit identity review. It does not freeze prompts, open held-out data,
run a model, open or create a checkpoint, publish evaluation results, construct
training data, create configs/schedules/optimizer state, authorize training,
train, commit, or push.
