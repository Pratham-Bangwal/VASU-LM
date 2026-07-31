# VASU-140M Release-Plan Post-Publication Test Repair Review Packet

Review the repair for the instruction-seed release-plan tests after the one
authorized publication made planned output paths legitimately present.

Read `AGENTS.md`, `docs/PROJECT_STATUS.md`,
`docs/VASU_140M_RELEASE_PLAN_POSTPUBLICATION_TEST_REPAIR_AUDIT_20260731.md`,
`docs/VASU_140M_INSTRUCTION_SEED_V1_PUBLICATION_INDEPENDENT_REVIEW_DECISION_20260731.md`,
`vasu/data/vasu_140m_release_plan.py`,
`scripts/smoke_vasu_140m_release_plan.py`, and
`tests/test_vasu_140m_release_plan.py`.

Confirm all of the following:

1. Default plan qualification still rejects an existing planned output path.
2. The exceptional replay mode is explicit, read-only, and bounded to
   historical pre-publication evidence.
3. Replay still matches the exact frozen report identity.
4. Tests prove both the fail-closed default and historical replay behavior.
5. No path permits overwrite, another publication, source selection, data
   construction, checkpoint creation, or training.

Run:

```powershell
python -m pytest tests\test_vasu_140m_release_plan.py -q
python scripts\smoke_vasu_140m_release_plan.py --allow-existing-planned-outputs | python -m json.tool > $null
python -m ruff check vasu\data\vasu_140m_release_plan.py scripts\smoke_vasu_140m_release_plan.py tests\test_vasu_140m_release_plan.py
git diff --check
git status --short
```

Write only
`docs/VASU_140M_RELEASE_PLAN_POSTPUBLICATION_TEST_REPAIR_INDEPENDENT_REVIEW_DECISION_20260731.md`.
State accept/reject, findings by severity, evidence examined, command results,
and a non-authorization confirmation. Acceptance authorizes only this repair;
it does not authorize publication, source selection, data construction,
configuration, schedules, checkpoints, or training.
