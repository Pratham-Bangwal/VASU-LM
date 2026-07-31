# VASU-140M Base-Pretraining Release Foundation Review Packet

Review the full-loss record implementation and production-builder design as
one pre-commit, non-authorizing foundation package.

Read:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_513_TOKEN_DATA_MASK_SPECIFICATION.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN_INDEPENDENT_REVIEW_DECISION_20260731.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_BUILDER_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_FOUNDATION_AUDIT_20260801.md`
- `vasu/data/vasu_140m_records.py`
- `vasu/data/vasu_140m_base_records.py`
- `tests/test_vasu_140m_records.py`
- `tests/test_vasu_140m_base_records.py`
- `scripts/smoke_vasu_140m_record_spec.py`
- `scripts/smoke_vasu_140m_base_records.py`
- `evaluation/fixtures/vasu_140m_513_record_spec_v1.json`
- `evaluation/fixtures/vasu_140m_base_text_record_fixture_v1.json`

Confirm full-loss masking, chunk/document boundaries, complete lineage,
document-level split isolation, revision consistency, deterministic packing,
real-tokenizer evidence, additive compatibility, streaming/source-separated
builder design, complete audits, two-pass replay, detached two-person one-build
authorization, atomicity/recovery, and non-authorization.

Run:

```powershell
python -m pytest tests\test_vasu_140m_base_records.py tests\test_vasu_140m_records.py -q
python -m ruff check vasu\data\vasu_140m_base_records.py tests\test_vasu_140m_base_records.py scripts\smoke_vasu_140m_base_records.py
python scripts\smoke_vasu_140m_base_records.py | python -m json.tool > $null
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_PRETRAINING_RELEASE_FOUNDATION_INDEPENDENT_REVIEW_DECISION_20260801.md`.

Acceptance authorizes only committing the reviewed foundation and later
implementing the production builder for separate review. It does not authorize
source admission/acquisition, production qualification/publication, schedules,
configs, checkpoints, optimizer state, or training.

