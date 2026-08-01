# VASU-140M Real-Data Exact-Resume Contract Implementation Review Packet

Status: **future second-opinion packet; author-side validation is complete and non-authorizing.**

## Requested Decision

Accept or reject the additive exact-resume contract implementation. Acceptance
means only that the schemas, identity checks, exact comparison logic, mutation
defense, adversarial result contract, tests, and frozen fixture are suitable as
the foundation for a later real-data workload runner.

Acceptance does not claim that the real workload has run or that the full
real-data exact-resume readiness gate has passed.

## Required Evidence

Review these files directly:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_REAL_DATA_EXACT_RESUME_DESIGN.md`
- `docs/VASU_140M_REAL_DATA_EXACT_RESUME_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_REAL_DATA_EXACT_RESUME_CONTRACT_IMPLEMENTATION_AUDIT_20260801.md`
- `vasu/training/vasu_140m_real_data_resume.py`
- `tests/test_vasu_140m_real_data_resume.py`
- `scripts/smoke_vasu_140m_real_data_resume_contract.py`
- `evaluation/fixtures/vasu_140m_real_data_resume_contract_qualification_v1.json`
- `scripts/qualify_vasu_140m_exact_resume.py`
- `vasu/training/trainer.py`
- `vasu/training/resume_state.py`
- `vasu/training/resumable_sampler.py`
- `vasu/training/checkpoint.py`

## Review Questions

1. Does the specification bind the exact family, release/source bytes,
   schedule, evaluation-development identity, workload, optimizer/scheduler,
   runtime, interruptions, outputs, review evidence, and non-authorization?
2. Are post-validation byte mutation and path-link attacks rejected before a
   future load?
3. Do the state digests and result validator fail closed on missing, changed,
   or internally inconsistent state?
4. Is exact equality required without a hidden tolerance?
5. Does the adversarial result contract cover every design-required failure?
6. Is the package additive and non-executing, with no hidden CUDA, optimizer,
   checkpoint, production-data, or training route?
7. Are any findings severe enough to block a clean commit and later runner
   implementation?

## Required Validation

Run:

```powershell
python -m pytest tests\test_vasu_140m_real_data_resume.py tests\test_vasu_140m_exact_resume_qualification.py tests\test_resumable_training.py tests\test_checkpoint_io.py tests\test_model_family_checkpoint_identity.py -q
python -m ruff check vasu\training\vasu_140m_real_data_resume.py tests\test_vasu_140m_real_data_resume.py scripts\smoke_vasu_140m_real_data_resume_contract.py
$observed = python scripts\smoke_vasu_140m_real_data_resume_contract.py | ConvertFrom-Json
$frozen = Get-Content evaluation\fixtures\vasu_140m_real_data_resume_contract_qualification_v1.json -Raw | ConvertFrom-Json
if (($observed | ConvertTo-Json -Depth 20 -Compress) -ne ($frozen | ConvertTo-Json -Depth 20 -Compress)) { throw 'fixture mismatch' }
git diff --check
git status --short
```

Confirm that the four protected paths listed in the audit remain absent.

## Decision Output

Create only:

`docs/VASU_140M_REAL_DATA_EXACT_RESUME_CONTRACT_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260801.md`

Report findings by severity, evidence examined, exact command results, fixture
reproducibility, compatibility, and explicit non-authorization. Do not modify
implementation, tests, fixtures, shared status documents, data, checkpoints,
configs, schedules, authorization records, or training state. Do not commit or
push.
