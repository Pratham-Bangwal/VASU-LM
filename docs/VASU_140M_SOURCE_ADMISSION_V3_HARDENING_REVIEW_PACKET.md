# VASU-140M Source-Admission v3 Hardening Review Packet

Status: **future second-opinion packet; author-side validation is complete and non-authorizing.**

## Requested Decision

Accept or reject the v3 source-admission hardening. Acceptance means the
registry semantics and complete evaluation-matrix requirements correctly close
the v2 gaps. It does not approve FineWeb-Edu, Wikipedia, acquisition, a data
release, or training.

## Required Evidence

Read:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V2_REMEDIATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V3_HARDENING_AUDIT_20260801.md`
- `vasu/data/vasu_140m_source_admission.py`
- `tests/test_vasu_140m_source_admission.py`
- `scripts/smoke_vasu_140m_source_admission_v3.py`
- `evaluation/fixtures/vasu_140m_source_admission_v3_qualification_v1.json`
- `configs/data/sources/fineweb_edu.json`
- `configs/data/sources/wikimedia.json`
- `vasu/data/sources/registry.py`
- `vasu/data/sources/schemas.py`
- `vasu/data/sources/validation.py`

Independently inspect the primary-source URLs listed in the audit.

## Required Questions

1. Was the v2 semantic mismatch real and significant?
2. Does v3 compare all critical legal, policy, revision, and access fields with
   the selected hash-bound registry record?
3. Does approval require exactly 10 unique five-prompt-dimension/two-split
   inventory bindings while correctly deferring corpus-likelihood splits until
   acquisition?
4. Does file validation reject metadata mismatch and `fixture_only=true`?
5. Can pending/blocked dossiers honestly represent incomplete evidence without
   becoming eligible for approval?
6. Is the schema-version compatibility conclusion accurate?
7. Are FineWeb-Edu and Wikipedia correctly left blocked rather than approved?
8. Is there any path to acquisition, publication, authorization, or training?

## Required Commands

```powershell
python -m pytest tests\test_vasu_140m_source_admission.py tests\test_data_source_registry.py -q
python -m ruff check vasu\data\vasu_140m_source_admission.py tests\test_vasu_140m_source_admission.py scripts\smoke_vasu_140m_source_admission_v3.py
$observed = python scripts\smoke_vasu_140m_source_admission_v3.py | ConvertFrom-Json
$frozen = Get-Content evaluation\fixtures\vasu_140m_source_admission_v3_qualification_v1.json -Raw | ConvertFrom-Json
if (($observed | ConvertTo-Json -Depth 20 -Compress) -ne ($frozen | ConvertTo-Json -Depth 20 -Compress)) { throw 'fixture mismatch' }
git diff --check
git status --short
```

## Decision Output

Create only:

`docs/VASU_140M_SOURCE_ADMISSION_V3_HARDENING_INDEPENDENT_REVIEW_DECISION_20260801.md`

Report decision, severity-grouped findings, evidence, exact validation results,
fixture reproducibility, compatibility, candidate-source status, and explicit
non-authorization. Do not modify code, tests, fixtures, registry files, shared
docs, data, checkpoints, schedules, configs, authorization records, or training
state. Do not commit or push.
