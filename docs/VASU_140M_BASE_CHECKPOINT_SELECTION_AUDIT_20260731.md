# VASU-140M Base Checkpoint Selection Audit — 2026-07-31

## Scope and conclusion

This read-only audit evaluated whether an existing repository checkpoint can
be selected as the parent for the published VASU-140M instruction seed v1
release. Result: **no compatible parent exists; no checkpoint was selected.**

## Evidence

- Repository `HEAD` was clean before this audit.
- `vasu/model/families.py` defines `vasu_60m_v1` with 58,337,792 parameters
  and `vasu_140m_v1` with 137,841,408 parameters.
- The recursive checkpoint inventory contains many VASU-60M and legacy
  artifacts, but no VASU-140M checkpoint directory or `.pt` file.
- `docs/PROJECT_STATUS.md`, `docs/CHECKPOINTS.md`, and `docs/ROADMAP.md`
  identify Candidate A and FineWeb step-200,000 as VASU-60M artifacts.
- The VASU-140M preflight reports the required family/configuration identity
  and retains `training_authorized=false`.

## Validation performed

```text
python scripts/preflight_vasu_140m.py                 passed
python -m pytest tests/test_model_families.py \
  tests/test_model_family_checkpoint_identity.py -q   28 passed
python -m ruff check vasu/config.py vasu/model/families.py \
  vasu/model/checkpoint_identity.py scripts/preflight_vasu_140m.py  passed
```

## Compatibility and authorization

No checkpoint, tokenizer, dataset, mask, manifest, schedule, optimizer state,
or release artifact was changed. Existing VASU-31M/60M checkpoints remain
loadable only in their matching families. The published VASU-140M release is
unchanged. No training configuration, schedule, authorization record,
checkpoint selection, optimizer state, or training run was created.

The required decision is documented in
`VASU_140M_BASE_CHECKPOINT_SELECTION_AND_TRAINING_PLAN.md`. GPT-5.5
independently accepted the conclusion in
`VASU_140M_BASE_CHECKPOINT_SELECTION_INDEPENDENT_REVIEW_DECISION_20260731.md`.
