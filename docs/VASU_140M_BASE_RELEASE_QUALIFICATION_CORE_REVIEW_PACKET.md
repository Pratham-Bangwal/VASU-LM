# VASU-140M Base-Release Qualification Core Review Packet

Review only the fixture-qualified, non-publishing qualification core. Do not
interpret acceptance as source admission, production data qualification,
publication authorization, or training authorization.

Read:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_BUILDER_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_FOUNDATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_FOUNDATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_RELEASE_QUALIFICATION_CORE_AUDIT_20260801.md`
- `vasu/data/vasu_140m_base_records.py`
- `vasu/data/vasu_140m_base_release.py`
- `tests/test_vasu_140m_base_release.py`
- `scripts/smoke_vasu_140m_base_release_qualification.py`
- `evaluation/fixtures/vasu_140m_base_release_qualification_fixture_v1.json`

Confirm the exact identities in the audit, strict semantic evidence envelopes,
fixture/production scope separation, recursive file binding, streaming
source-separated output, complete serialized token/mask/lineage audits,
source/split isolation, deterministic two-pass identity, disk and scratch
safety, failure preservation, report integrity, additive compatibility, and
all non-authorization flags.

Run:

```powershell
python -m pytest tests\test_vasu_140m_base_release.py tests\test_vasu_140m_base_records.py tests\test_vasu_140m_records.py tests\test_vasu_140m_release_plan.py -q
python -m ruff check vasu\data\vasu_140m_base_release.py tests\test_vasu_140m_base_release.py scripts\smoke_vasu_140m_base_release_qualification.py
$observed = (python scripts\smoke_vasu_140m_base_release_qualification.py | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)
$expected = (Get-Content evaluation\fixtures\vasu_140m_base_release_qualification_fixture_v1.json -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)
if ($observed -ne $expected) { throw "base-release qualification fixture mismatch" }
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_RELEASE_QUALIFICATION_CORE_INDEPENDENT_REVIEW_DECISION_20260801.md`.

Acceptance authorizes only an isolated commit and later clean post-commit
identity review. It does not approve a source, validate real source evidence,
construct or publish production data, create an authorization envelope,
schedule, config, optimizer, checkpoint, or start training.
