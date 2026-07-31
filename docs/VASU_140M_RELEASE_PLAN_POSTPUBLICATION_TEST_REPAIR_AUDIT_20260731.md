# VASU-140M Release-Plan Post-Publication Test Repair Audit — 2026-07-31

## Root cause

The instruction-seed release-plan qualification was correctly frozen before
the one authorized publication. Its repository tests later continued to invoke
the default qualification against the live repository, where the planned
output paths now exist. Five tests therefore failed with `planned production
output already exists` even though the intended no-overwrite gate was working.

## Repair

`validate_release_plan` remains fail-closed by default. The smoke helper now
requires an explicit `allow_existing_planned_outputs=True` opt-in to replay the
immutable pre-publication qualification after the authorized publication. The
CLI equivalent is `--allow-existing-planned-outputs`.

That opt-in does not write files, make a build permissible, change the frozen
report, or claim that outputs are currently absent. Tests now prove both
properties: default qualification rejects the published output, and explicit
inspection reproduces the exact frozen report.

## Validation

- `python -m pytest tests\\test_vasu_140m_release_plan.py -q` — 7 passed.
- `python scripts\\smoke_vasu_140m_release_plan.py --allow-existing-planned-outputs | python -m json.tool > $null` — passed.
- `python -m ruff check vasu\\data\\vasu_140m_release_plan.py scripts\\smoke_vasu_140m_release_plan.py tests\\test_vasu_140m_release_plan.py` — passed.
- `git diff --check` — passed.

## Compatibility and authority

No data, masks, tokenizer assets, checkpoints, manifests, receipts, schedules,
or evaluation results were modified. Dataset, tokenizer, checkpoint, and
exact-resume compatibility are unchanged. The repair does not authorize
publication, source selection, configuration, optimizer creation, or training.

## Independent decision

GPT-5.5 independently accepted the repair on 2026-07-31 in
`VASU_140M_RELEASE_PLAN_POSTPUBLICATION_TEST_REPAIR_INDEPENDENT_REVIEW_DECISION_20260731.md`.
Acceptance covers only this regression repair and does not authorize another
publication, source selection, data construction, configuration, checkpoints,
or training.
