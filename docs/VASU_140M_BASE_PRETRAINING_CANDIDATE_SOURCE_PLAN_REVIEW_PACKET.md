# VASU-140M Candidate Source Plan Review Packet

Review the proposed FineWeb-Edu-extension plus bounded-Wikipedia strategy.

Read `AGENTS.md`, `docs/PROJECT_STATUS.md`, the accepted source-admission
design and implementation decisions,
`docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN.md`,
`docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN.md`,
`docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN_AUDIT_20260801.md`,
`docs/DATA_SOURCE_REGISTRY.md`, `configs/data/sources/fineweb_edu.json`,
`configs/data/sources/wikimedia.json`, and the referenced recovery/index docs.

Confirm that direct binary reuse and incomplete original-source provenance are
rejected; the two candidates are independently admitted; legal, contamination,
deduplication, split, measurement, and no-replacement gates are explicit; and
no acquisition or release is authorized.

Run:

```powershell
python -m pytest tests\test_data_source_registry.py tests\test_vasu_140m_source_admission.py -q
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN_INDEPENDENT_REVIEW_DECISION_20260801.md`.
Acceptance authorizes only later source-specific admission packages. It does
not authorize registry mutation, acquisition, processing, publication,
configuration, checkpoints, or training.
