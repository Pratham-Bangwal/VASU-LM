# VASU-140M Base-Pretraining Plan Schema v2 Remediation Audit — 2026-08-01

Status: pre-commit remediation evidence; non-authorizing.

## Root cause

The independently accepted pre-commit v1 schema bound useful identities and
accounting, but an author-side adversarial audit found that it could still
admit an unsafe or scientifically ambiguous future plan. In particular, the
selected release and evaluation suite did not have to equal their accepted
gate artifacts; the first-lineage comparison was implicit; schedule and
release could be the same file; source manifests and runtime bounds were not
bound; validation/checkpoint intervals could exceed the run; final validation
and checkpointing were optional; disk and thermal limits were weakly
constrained; and outputs could use arbitrary repository paths.

Because v1 had already received an independent decision, the corrected
contract uses schema ID `vasu_140m_base_pretraining_plan_v2`. The earlier
decision remains historical and cannot accept the new bytes or schema.

## Remediation

Version 2 now requires:

- globally unique artifact and decision paths across all four gates;
- exact equality between the selected release/evaluation suite and their
  accepted gate artifacts;
- a separately bound schedule artifact;
- a per-source manifest hash in addition to token, mask, and lineage hashes;
- an explicit `pre_update_random_initialization` comparison with no claimed
  matched trained parent;
- validation and checkpoint intervals no greater than the update budget;
- final validation and a final atomic checkpoint;
- exact optimizer-update/input-position runtime caps, a coherent throughput
  range, a slow-bound-compatible wall-time cap, and stop-at-budget behavior;
- at least 10 GB free disk, mandatory telemetry, warning below abort, abort no
  higher than 88 C, and critical no higher than 95 C; and
- isolated absent output directories under `checkpoints/vasu_140m/`,
  `logs/vasu_140m/`, and `evaluation/results/vasu_140m/`.

## Validation evidence

- `python -m pytest tests\test_vasu_140m_base_plan.py -q`
- `python -m pytest tests\test_vasu_140m_base_plan.py tests\test_config_schema.py tests\test_experiment_governance.py tests\test_platform_preflight.py -q`
- `python -m ruff check vasu\training\vasu_140m_base_plan.py tests\test_vasu_140m_base_plan.py`
- `git diff --check`

The focused suite contains 17 passing adversarial tests after remediation; the
combined governance/preflight regression set contains 23 passing tests.

Frozen pre-commit identities:

- parent commit:
  `483f4b9b65bf27b87ef603129c0a9f9e7de79959`;
- implementation SHA-256:
  `28761e278326e81685d57bcc119a63cc72c50676bb27cc6eff6f99f388d35a47`;
- test SHA-256:
  `5aa2bedba1e753ab3daee867f27db1a3cd97d7f18cc940b6f8542a4d35f24e35`;
- schema/design SHA-256:
  `c60f7ca28a79f82b7743920662def0fe672dfbc038317cd0770360f9c82be6e3`.

No real plan, source acquisition, data release, schedule, model allocation,
optimizer, output directory, checkpoint, authorization envelope, or training
run was created.
