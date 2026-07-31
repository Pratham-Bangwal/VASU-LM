# VASU-140M Base-Pretraining Plan Schema v2 Remediation Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the hardened VASU-140M base-pretraining plan schema v2 for an isolated
schema commit and later clean post-commit identity review.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree with unrelated
  modified documentation and untracked VASU-140M schema files. This was not
  treated as a blocker because the v2 package is internally consistent,
  schema-only, and the requested focused tests and identity checks pass.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_READINESS.md`
- `docs/VASU_140M_BASE_CHECKPOINT_SELECTION_AND_TRAINING_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_AND_PREFLIGHT_SCHEMA.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_V2_REMEDIATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_V2_REMEDIATION_REVIEW_PACKET.md`
- `vasu/training/vasu_140m_base_plan.py`
- `tests/test_vasu_140m_base_plan.py`
- `vasu/training/config_schema.py`
- `vasu/training/experiment_governance.py`

## Rationale

Version 2 uses schema ID `vasu_140m_base_pretraining_plan_v2`, so it supersedes
the previously accepted v1 schema and cannot inherit the old v1 acceptance.
The prior decision remains historical evidence for the old bytes only.

The validator requires exactly the four readiness gates and globally unique
gate artifact and decision paths, including a same-path case-insensitive
collision check. Gate artifacts must be independently accepted. The selected
base-data release must exactly match the accepted `base_data_release`
artifact, and the selected evaluation suite must exactly match the accepted
`base_evaluation_v2` artifact.

The schedule is separately bound by path, SHA-256, and record count, and it is
rejected if it reuses the release manifest path. Each source now binds
manifest, token, mask, and lineage SHA-256 values, plus train-record and
supervised-target counts.

The first-lineage comparison is explicit and honest:
`pre_update_random_initialization` with `matched_parent=false`. The plan still
requires fresh random initialization and rejects any parent checkpoint, so it
does not claim Candidate A or any other trained checkpoint as a VASU-140M
parent.

The accounting remains exact across schedule records, source records,
processed records, microbatches, accumulation groups, optimizer updates,
processed input positions, supervised-target budget, and cosine scheduler
totals. The runtime envelope binds minimum and maximum throughput, wall-time
cap, optimizer-update cap, input-position cap, and `stop_at_budget=true`.

Validation and checkpoint intervals must fit within the run, and final
validation plus final checkpoint are mandatory. Safety hardening requires at
least 10 GB free disk, disk checks before launch and checkpoint, required
telemetry, ordered thermal limits, abort temperature no higher than 88 C,
critical temperature no higher than 95 C, atomic checkpoints with sidecars,
and all required hard-stop causes.

Outputs are restricted to distinct, non-overlapping, initially absent
directories under `checkpoints/vasu_140m/`, `logs/vasu_140m/`, and
`evaluation/results/vasu_140m/`. Review state must remain
`pending_independent_review`, and `training_authorized` must remain false.
The package validates future plan structure only and does not instantiate or
launch a real plan.

## Commands and Exact Results

- `git rev-parse HEAD`
  returned `483f4b9b65bf27b87ef603129c0a9f9e7de79959`.
- `git status --short`
  showed the known concurrent modified documentation and untracked schema
  package files.
- `python -m pytest tests\test_vasu_140m_base_plan.py tests\test_config_schema.py tests\test_experiment_governance.py tests\test_platform_preflight.py -q`
  passed: 23 passed in 0.72s.
- `python -m ruff check vasu\training\vasu_140m_base_plan.py tests\test_vasu_140m_base_plan.py`
  passed: all checks passed.
- `git diff --check`
  completed with CRLF line-ending warnings only for existing modified files.
- `Get-FileHash vasu\training\vasu_140m_base_plan.py -Algorithm SHA256`
  returned `28761e278326e81685d57bcc119a63cc72c50676bb27cc6eff6f99f388d35a47`.
- `Get-FileHash tests\test_vasu_140m_base_plan.py -Algorithm SHA256`
  returned `5aa2bedba1e753ab3daee867f27db1a3cd97d7f18cc940b6f8542a4d35f24e35`.
- `Get-FileHash docs\VASU_140M_BASE_PRETRAINING_PLAN_AND_PREFLIGHT_SCHEMA.md -Algorithm SHA256`
  returned `c60f7ca28a79f82b7743920662def0fe672dfbc038317cd0770360f9c82be6e3`.

Additional absence checks found no:

- `configs/training/vasu_140m_base_pretraining.json`
- `configs/schedules/vasu_140m_base_pretraining.json`
- `configs/training/vasu_140m_base_pretraining`
- `configs/schedules/vasu_140m_base_pretraining`
- `checkpoints/vasu_140m/base_pretraining`
- `data/processed/vasu_140m/base_pretraining`
- `data/manifests/vasu_140m/base_pretraining`
- `authorizations/vasu_140m_base_pretraining`

## Exact Identity Comparison

| Identity | Expected | Observed | Result |
| --- | --- | --- | --- |
| Parent commit | `483f4b9b65bf27b87ef603129c0a9f9e7de79959` | `483f4b9b65bf27b87ef603129c0a9f9e7de79959` | Match |
| Implementation SHA-256 | `28761e278326e81685d57bcc119a63cc72c50676bb27cc6eff6f99f388d35a47` | `28761e278326e81685d57bcc119a63cc72c50676bb27cc6eff6f99f388d35a47` | Match |
| Test SHA-256 | `5aa2bedba1e753ab3daee867f27db1a3cd97d7f18cc940b6f8542a4d35f24e35` | `5aa2bedba1e753ab3daee867f27db1a3cd97d7f18cc940b6f8542a4d35f24e35` | Match |
| Design SHA-256 | `c60f7ca28a79f82b7743920662def0fe672dfbc038317cd0770360f9c82be6e3` | `c60f7ca28a79f82b7743920662def0fe672dfbc038317cd0770360f9c82be6e3` | Match |

## Compatibility Conclusion

The v2 schema is additive for existing VASU-31M and VASU-60M systems. It does
not change existing configuration schemas, governance helpers, launchers,
tokenizer assets, datasets, masks, schedules, checkpoints, optimizer state,
authorization records, or training code. It intentionally supersedes only the
uninstantiated VASU-140M v1 future-plan schema.

## Non-Authorization

This acceptance authorizes only committing the schema-only v2 remediation and
later preparing a clean post-commit identity review. It does not authorize a
real experiment plan, training configuration, data schedule, authorization
envelope, source acquisition, data release construction, source-share or
hyperparameter selection, model allocation, optimizer creation, checkpoint
creation, production output creation, training, commit, or push.
