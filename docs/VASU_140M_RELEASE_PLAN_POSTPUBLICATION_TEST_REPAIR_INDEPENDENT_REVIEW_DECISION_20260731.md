# VASU-140M Release-Plan Post-Publication Test Repair Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-07-31

Reviewer: GPT-5.5 independent review

## Decision

Accept the VASU-140M release-plan post-publication test repair.

The repair preserves the fail-closed default behavior while adding an explicit,
read-only replay mode for the immutable pre-publication qualification evidence
after the one authorized instruction-seed publication made planned output paths
legitimately present.

## Findings With Severity

- Blocking: none.
- High: none.
- Medium: none.
- Low: existing unrelated modified documentation remains in
  `docs/CHANGELOG.md`, `docs/PROJECT_STATUS.md`, and `docs/ROADMAP.md`. This
  does not affect the repair conclusion because focused tests, explicit replay,
  default failure behavior, Ruff, and diff checks all pass.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_RELEASE_PLAN_POSTPUBLICATION_TEST_REPAIR_AUDIT_20260731.md`
- `docs/VASU_140M_RELEASE_PLAN_POSTPUBLICATION_TEST_REPAIR_REVIEW_PACKET.md`
- `docs/VASU_140M_INSTRUCTION_SEED_V1_PUBLICATION_INDEPENDENT_REVIEW_DECISION_20260731.md`
- `vasu/data/vasu_140m_release_plan.py`
- `scripts/smoke_vasu_140m_release_plan.py`
- `tests/test_vasu_140m_release_plan.py`

## Validation Results

- `python -m pytest tests\test_vasu_140m_records.py tests\test_vasu_140m_release_plan.py -q`:
  passed, `26 passed`.
- `python scripts\smoke_vasu_140m_release_plan.py --allow-existing-planned-outputs | python -m json.tool > $null`:
  passed.
- `python -m ruff check vasu\data\vasu_140m_release_plan.py scripts\smoke_vasu_140m_release_plan.py tests\test_vasu_140m_release_plan.py`:
  passed, `All checks passed!`.
- `git diff --check`: passed with no whitespace errors; Git reported CRLF
  conversion warnings for existing modified files.
- `git status --short`: inspected. Existing modified docs and repair files are
  present, plus this decision document after acceptance.
- `python scripts\smoke_vasu_140m_release_plan.py`: failed as required with
  `ValueError: planned production output already exists:
  data/processed/vasu_140m/instruction_seed_v1`.

## Repair Review Conclusion

The repaired behavior satisfies the requested checks:

- default qualification still rejects existing planned output paths;
- the post-publication replay is explicit through
  `allow_existing_planned_outputs=True` or
  `--allow-existing-planned-outputs`;
- replay remains read-only and reproduces only the immutable pre-publication
  report;
- replay matches frozen report SHA-256
  `8147215c2f58044aa60374f39e772e13fce3913090ea5c2d8acc915aeeca6fcb`;
- replay still reports `release_build_permitted=false`,
  `training_authorized=false`, and `training_permitted=false`;
- tests prove both fail-closed default qualification and explicit historical
  replay;
- no overwrite, republish, source-selection, data-construction, checkpoint, or
  training route was introduced.

## Non-Authorization Confirmation

This acceptance authorizes only retaining the post-publication test repair. It
does not authorize publication, source selection, data construction,
configuration creation, schedule creation, optimizer creation, checkpoint
creation, authorization records, base pretraining, instruction tuning, commit,
or push. No publication occurred, no data was constructed, no protected
artifact was created, no training occurred, no authorization changed, no commit
was created, and no push was performed.
