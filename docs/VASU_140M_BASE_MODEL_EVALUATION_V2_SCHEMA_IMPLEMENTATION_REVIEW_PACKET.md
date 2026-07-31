# VASU-140M Base-Model Evaluation v2 Schema Implementation Review Packet

Review this as a pre-commit, schema-only implementation package.

Read:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_IMPLEMENTATION_AUDIT_20260801.md`
- `evaluation/framework/vasu_140m_base_v2.py`
- `tests/test_vasu_140m_base_evaluation_v2.py`
- `scripts/smoke_vasu_140m_base_evaluation_v2_schema.py`
- `evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_qualification_precommit.json`

Confirm strict identities, all six dimensions and both splits, sealed held-out
behavior, direct-likelihood separation, greedy/sampled bindings, contamination
ordering, bootstrap requirements, multi-shard/multi-metric reporting, no
aggregate score, resume identity, adversarial coverage, and non-authorization.

Run:

```powershell
python -m pytest tests\test_vasu_140m_base_evaluation_v2.py -q
python -m ruff check evaluation\framework\vasu_140m_base_v2.py tests\test_vasu_140m_base_evaluation_v2.py scripts\smoke_vasu_140m_base_evaluation_v2_schema.py
$observed = (python scripts\smoke_vasu_140m_base_evaluation_v2_schema.py | ConvertFrom-Json | ConvertTo-Json -Depth 20 -Compress)
$expected = (Get-Content evaluation\fixtures\vasu_140m_base_evaluation_v2_schema_qualification_precommit.json -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 20 -Compress)
if ($observed -ne $expected) { throw "qualification fixture mismatch" }
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260801.md`.

Acceptance authorizes only committing the reviewed implementation and later
post-commit identity review. It does not freeze real prompts or held-out data,
implement scorers, run a model, open a checkpoint, publish results, construct
training data, or authorize training.
