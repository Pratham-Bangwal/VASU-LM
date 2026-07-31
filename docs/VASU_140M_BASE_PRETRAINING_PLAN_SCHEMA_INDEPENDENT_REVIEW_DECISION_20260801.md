# VASU-140M Base-Pretraining Plan Schema Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the future VASU-140M base-pretraining plan schema and final-preflight
design for an isolated schema commit.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree with unrelated
  modified and untracked VASU-140M packages. This was not treated as a blocker
  because the schema is internally non-authorizing and does not depend on those
  unrelated packages being accepted.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_READINESS.md`
- `docs/VASU_140M_BASE_CHECKPOINT_SELECTION_AND_TRAINING_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_AND_PREFLIGHT_SCHEMA.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_PLAN_SCHEMA_REVIEW_PACKET.md`
- `vasu/training/vasu_140m_base_plan.py`
- `tests/test_vasu_140m_base_plan.py`
- `vasu/training/config_schema.py`
- `vasu/training/experiment_governance.py`

## Rationale

The schema binds the exact `vasu_140m_v1` family, family SHA-256
`72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b`,
model-configuration SHA-256
`29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059`,
tokenizer SHA-256
`04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a`,
repository commit, file-bound evidence, and canonical plan SHA-256.

All four readiness gates are mandatory and must have independently accepted
artifact and decision bindings: CUDA/AMP, base-data release, base-evaluation
v2, and real-data exact resume. The first-lineage requirement is correctly
fail-closed: initialization must be `fresh_random`, with no parent checkpoint.
This rejects VASU-31M/60M parents, Candidate A, converted weights, partial
loads, or shape-tolerant loading as a VASU-140M base parent.

The validator reconciles source records, schedule record count, processed
records, microbatches, accumulation groups, optimizer updates, processed input
positions, supervised-target budget, and scheduler total updates. Optimizer
and scheduler parameters are exact-field validated, constrained to AdamW plus
cosine schedule, and bound to implementation identity where applicable.

Development and held-out evaluations remain separate. Both split inventories
must cover likelihood, factuality, arithmetic, repetition, robustness, and
manual review exactly once. Development is required before training, while
held-out opening before training is rejected.

The safety contract requires disk checks, required ordered thermal telemetry,
atomic checkpoints with sidecars, and fail-closed abort causes for CUDA OOM,
non-finite values, checkpoint failure, validation failure, resume divergence,
thermal limit, and disk limit. Output checkpoint, log, and result directories
must be distinct, repository-relative, and absent before launch.

Review state must remain pending and `training_authorized` must remain false.
The documented final-preflight and detached authorization design still require
a later exact human approval and do not make the package launchable.

## Commands and Exact Results

- `python -m pytest tests\test_vasu_140m_base_plan.py -q`
  passed: 14 passed in 0.44s.
- `python -m ruff check vasu\training\vasu_140m_base_plan.py tests\test_vasu_140m_base_plan.py`
  passed: all checks passed.
- `git diff --check`
  completed with CRLF line-ending warnings only for pre-existing modified
  files.
- `git status --short`
  showed concurrent modified and untracked author-side packages, including
  this schema package.

Additional identity and absence checks:

- Module SHA-256:
  `7e762c8cb95b3014d24dfea9cf5093e73950718301a447541306806c2913e447`
- Test SHA-256:
  `b4a7b02f38e63b75ad2e085891a2e74f0d02adfc209871b1eba8fbf9e25d9640`
- Schema document SHA-256:
  `25c0700d5e48e5d288800bf240ef7be3cfc5ae19b70b00c8250fa43c2b562171`
- Audit document SHA-256:
  `43decc34a2b3556a135ddd5c180d83d61c1f33f61d79a2c2fa357b290e6545b1`
- Review packet SHA-256:
  `f6a301cd43db0d2246d9b0c3bba32d12b698f3e9a9a4b5dac0c83a91ecf11f8f`
- Checked absent:
  `configs/training/vasu_140m_base_pretraining.json`,
  `configs/schedules/vasu_140m_base_pretraining.json`,
  `configs/training/vasu_140m_base_pretraining`,
  `configs/schedules/vasu_140m_base_pretraining`,
  `checkpoints/vasu_140m/base_pretraining`,
  `data/manifests/vasu_140m/base_pretraining`,
  `data/processed/vasu_140m/base_pretraining`, and
  `authorizations/vasu_140m_base_pretraining`.

## Compatibility Conclusion

The package is additive. It does not alter existing VASU-31M/60M
configuration schemas, experiment-governance helpers, tokenizer assets,
datasets, masks, schedules, checkpoints, optimizer state, authorization
records, or training launchers. It creates a strict future-plan validator but
does not instantiate a real experiment plan, configuration, schedule, or
runner path.

## Non-Authorization

This acceptance authorizes only committing this reviewed schema. A real
VASU-140M base-pretraining plan remains prohibited until all four production
readiness gates have separately accepted evidence. No actual plan,
configuration, schedule, authorization envelope, model, optimizer, checkpoint,
or training run was created. This decision does not authorize source
acquisition, data release construction, configuration creation, schedule
creation, authorization-envelope creation, model allocation, optimizer-state
creation, checkpoint creation, training, commit, or push.
