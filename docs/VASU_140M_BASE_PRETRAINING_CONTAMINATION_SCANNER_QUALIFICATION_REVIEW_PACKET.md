# VASU-140M Base-Pretraining Contamination Scanner Qualification Review Packet

Status: **review packet for a future second opinion; author-side validation is complete and non-authorizing.**

## Requested Decision

Accept or reject the fixture-qualified pre-split contamination scanner. Review
whether it correctly turns public inventory commitments and independently
reviewed semantic candidates into immutable per-document eligibility evidence.
Acceptance does not approve any production inventory, source, acquisition,
release, model execution, or training.

## Required Evidence

Read completely:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONTRACT_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_CONSTRUCTION_PLAN_REPRODUCIBILITY_FIX_AUDIT_20260802.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_INVENTORY_BUILDER_QUALIFICATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_CONTAMINATION_SCANNER_QUALIFICATION_AUDIT_20260801.md`
- `evaluation/framework/vasu_140m_base_v2_inventory.py`
- `vasu/data/vasu_140m_contamination_scan.py`
- `tests/test_vasu_140m_contamination_scan.py`
- `scripts/smoke_vasu_140m_contamination_scan.py`
- `evaluation/fixtures/vasu_140m_contamination_scan_qualification_v1.json`
- `vasu/data/vasu_140m_source_admission.py`

If the inventory-contract decision file is absent or does not accept the exact
contract identity in its audit, reject this package as dependency-ineligible.

## Required Questions

1. Does rolling word-span matching detect prompts and short answers embedded in
   longer documents?
2. Is NFC, whitespace, and case normalization consistent between inventory
   commitments and scanning?
3. Are eight-word fragments reconciled without weakening exact matching?
4. Must every semantic candidate have exactly one valid human decision, and
   are the exact candidate and decision lists hash-bound into the report?
5. Are unknown items/documents, duplicate identities, non-finite similarity,
   source substitution, malformed counts, and report mutation rejected?
6. Does the report reconcile findings, semantic rejections, outcomes, and all
   counts while retaining only source-document hashes?
7. Is scanning unambiguously pre-split and non-authorizing?
8. Is there any acquisition, publication, model-execution, checkpoint, or
   training route?

## Required Commands

```powershell
python -m pytest tests\test_vasu_140m_contamination_scan.py tests\test_vasu_140m_base_v2_inventory.py -q
python -m ruff check vasu\data\vasu_140m_contamination_scan.py tests\test_vasu_140m_contamination_scan.py scripts\smoke_vasu_140m_contamination_scan.py
$observed = python scripts\smoke_vasu_140m_contamination_scan.py | ConvertFrom-Json
$frozen = Get-Content evaluation\fixtures\vasu_140m_contamination_scan_qualification_v1.json -Raw | ConvertFrom-Json
if (($observed | ConvertTo-Json -Depth 20 -Compress) -ne ($frozen | ConvertTo-Json -Depth 20 -Compress)) { throw 'fixture mismatch' }
git diff --check
git status --short
```

## Decision Output

Create only:

`docs/VASU_140M_BASE_PRETRAINING_CONTAMINATION_SCANNER_QUALIFICATION_INDEPENDENT_REVIEW_DECISION_20260801.md`

Report the decision, severity-grouped findings, evidence examined, exact
validation results, fixture reproducibility, dependency status, compatibility,
and explicit non-authorization. Do not modify implementation, tests, fixtures,
shared docs, data, registries, checkpoints, configurations, schedules,
authorization records, or training state. Do not commit or push.
