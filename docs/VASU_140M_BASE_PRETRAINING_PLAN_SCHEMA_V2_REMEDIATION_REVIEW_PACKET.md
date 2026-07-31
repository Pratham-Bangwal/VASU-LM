# VASU-140M Base-Pretraining Plan Schema v2 Remediation Review Packet

Review this as a corrected future-plan schema only. Do not inherit the v1
decision: the schema ID and implementation bytes have changed.

Read:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_READINESS.md`
- `docs/VASU_140M_BASE_CHECKPOINT_SELECTION_AND_TRAINING_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_AND_PREFLIGHT_SCHEMA.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_V2_REMEDIATION_AUDIT_20260801.md`
- `vasu/training/vasu_140m_base_plan.py`
- `tests/test_vasu_140m_base_plan.py`
- `vasu/training/config_schema.py`
- `vasu/training/experiment_governance.py`

Confirm that v2 closes every audit finding, is fail-closed against v1 and
unknown fields, remains additive for existing VASU-31M/60M systems, and
cannot instantiate or launch a real plan. Verify gate-path uniqueness,
gate-to-selected-artifact equality, comparison honesty, source manifest
binding, exact accounting, bounded runtime, final evidence, conservative
safety limits, output isolation, pending review, canonical plan identity, and
repository-file validation.

Expected pre-commit identities:

- parent commit:
  `483f4b9b65bf27b87ef603129c0a9f9e7de79959`;
- implementation SHA-256:
  `28761e278326e81685d57bcc119a63cc72c50676bb27cc6eff6f99f388d35a47`;
- test SHA-256:
  `5aa2bedba1e753ab3daee867f27db1a3cd97d7f18cc940b6f8542a4d35f24e35`;
- schema/design SHA-256:
  `c60f7ca28a79f82b7743920662def0fe672dfbc038317cd0770360f9c82be6e3`.

Run:

```powershell
python -m pytest tests\test_vasu_140m_base_plan.py -q
python -m ruff check vasu\training\vasu_140m_base_plan.py tests\test_vasu_140m_base_plan.py
git diff --check
git status --short
```

Create only
`docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_V2_REMEDIATION_INDEPENDENT_REVIEW_DECISION_20260801.md`.

Acceptance authorizes only an isolated commit of the v2 schema package and a
later clean post-commit identity review. It does not authorize creating a real
plan/config/schedule/envelope, selecting source shares or hyperparameters,
allocating a model or optimizer, creating a checkpoint, or training.
