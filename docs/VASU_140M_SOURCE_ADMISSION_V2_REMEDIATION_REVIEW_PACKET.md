# VASU-140M Source-Admission v2 Remediation Review Packet

Review the versioned correction to the previously accepted VASU-140M
source-admission validator.

## Required evidence

Read:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_SOURCE_ADMISSION_SCHEMA_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V2_REMEDIATION_AUDIT_20260801.md`
- `vasu/data/vasu_140m_source_admission.py`
- `vasu/data/sources/__init__.py`
- `vasu/data/sources/registry.py`
- `tests/test_vasu_140m_source_admission.py`
- `tests/test_data_source_registry.py`
- `configs/data/sources/fineweb_edu.json`
- `configs/data/sources/wikimedia.json`

## Decision questions

Confirm that:

1. the v1 state conflation and bundled-record failure are real defects;
2. v2 clearly separates generic registry approval from the VASU-140M
   admission decision;
3. VASU-140M approval still requires a generically approved source plus every
   model/release-specific reviewer and evidence gate;
4. v1 packages and ambiguous or duplicate source-ID resolution fail closed;
5. the public multi-record loader preserves existing strict registry behavior;
6. the real bundled FineWeb-Edu extension record is hash-bound and resolves
   exactly once; and
7. no source admission, registry mutation, acquisition, data release,
   checkpoint, configuration, authorization, or training route was created.

## Validation commands

```powershell
python -m pytest tests\test_vasu_140m_source_admission.py tests\test_data_source_registry.py tests\test_vasu_140m_records.py -q
python -m ruff check vasu\data\vasu_140m_source_admission.py vasu\data\sources\registry.py vasu\data\sources\__init__.py tests\test_vasu_140m_source_admission.py
git diff --check
git status --short
```

Create only
`docs/VASU_140M_SOURCE_ADMISSION_V2_REMEDIATION_INDEPENDENT_REVIEW_DECISION_20260801.md`.

Acceptance authorizes only committing this remediation and later preparing
separate pending admission packages. It does not authorize source selection as
final, registry mutation, acquisition, data construction, publication,
configuration, schedule, checkpoint, optimizer, authorization record, base
pretraining, instruction tuning, training, push, or any source-specific
approval.
