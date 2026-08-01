# VASU-140M Base-Model Evaluation v2 Scoring Qualification Review Packet

Review only the prompt-free scorer and statistics qualification package. Do
not interpret acceptance as freezing production prompts, opening held-out
content, authorizing model evaluation, or authorizing training.

Read completely:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_REMEDIATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_REMEDIATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_AUDIT_20260801.md`
- `evaluation/framework/vasu_140m_base_v2.py`
- `evaluation/framework/vasu_140m_base_v2_tasks.py`
- `evaluation/framework/vasu_140m_base_v2_statistics.py`
- `tests/test_vasu_140m_base_v2_scoring.py`
- `scripts/smoke_vasu_140m_base_v2_scoring.py`
- `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json`

Confirm the exact identities in the audit and verify strict fields, unique
prompt/text identities, complete dimension/mode coverage, direct-likelihood
semantics, visible factual ties, arithmetic error taxonomy, token-level
degeneration metrics, paired robustness, deterministic bootstrap intervals,
complete strata, blank deterministic manual sampling, compact-to-full report
binding, absence of an aggregate capability score, and all non-authorization
flags.

Run:

```powershell
python -m pytest tests\test_vasu_140m_base_v2_scoring.py tests\test_vasu_140m_base_evaluation_v2.py tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q
python -m ruff check evaluation\framework\vasu_140m_base_v2_tasks.py evaluation\framework\vasu_140m_base_v2_statistics.py tests\test_vasu_140m_base_v2_scoring.py scripts\smoke_vasu_140m_base_v2_scoring.py
$observed = (python scripts\smoke_vasu_140m_base_v2_scoring.py --evidence-only | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)
$expected = (Get-Content evaluation\fixtures\vasu_140m_base_v2_scoring_qualification_v1.json -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)
if ($observed -ne $expected) { throw "scoring qualification fixture mismatch" }
python scripts\smoke_vasu_140m_base_v2_scoring.py | python -m json.tool > $null
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_INDEPENDENT_REVIEW_DECISION_20260801.md`.

Acceptance authorizes only an isolated commit and later clean post-commit
identity review. It does not freeze a production suite, create or reveal
development/held-out prompts, open held-out content, invoke a model, access or
create a checkpoint, publish an evaluation result, construct data, create a
configuration/schedule/optimizer/authorization record, or start training.
