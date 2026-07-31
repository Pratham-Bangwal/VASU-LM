# VASU-140M Base-Model Evaluation v2 Schema Implementation Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the pre-commit VASU-140M base-model evaluation v2 schema implementation
for an isolated commit and later clean post-commit identity review.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree with unrelated
  modified source-admission files and untracked VASU-140M packages. This was
  not treated as a package blocker because the schema implementation is
  internally self-contained and the frozen qualification reproduced exactly.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_IMPLEMENTATION_REVIEW_PACKET.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_IMPLEMENTATION_AUDIT_20260801.md`
- `evaluation/framework/vasu_140m_base_v2.py`
- `tests/test_vasu_140m_base_evaluation_v2.py`
- `scripts/smoke_vasu_140m_base_evaluation_v2_schema.py`
- `evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_qualification_precommit.json`

## Rationale

The schema implementation enforces exact fields for suite and result
manifests and binds the frozen VASU-140M family, model configuration,
tokenizer, checkpoint, repository, runtime, command, result, and resume
identities. It covers likelihood, factuality, arithmetic, repetition,
robustness, and manual review as separate dimensions across development and
held-out splits.

The held-out policy remains sealed: default validation verifies development
files only, refuses explicit held-out opening, and records no held-out access.
Likelihood and factuality inventories use the `direct_likelihood` interface
and are rejected if they bind generation profiles. Generative dimensions bind
both greedy and sampled profiles. Contamination is required before training
split assignment. Results support multiple shards and metrics without a
single aggregate capability score.

The adversarial coverage rejects unknown fields, unsafe paths, mutations,
non-finite statistics, wrong-family checkpoints, and authorization changes.
The synthetic qualification did not invoke a model, open a checkpoint, freeze
real prompts, publish results, open held-out inventories, or authorize
training.

## Identity Comparison

Reproduced reviewed identities:

- Module SHA-256:
  `b3d889168196b3c880eb99f9868f14105d6b4b3b5cc84010add96ffb77997c49`
- Test SHA-256:
  `313126c35430372832a9f19f463d15e01a3b054ad0a7f6c15027cf7813174319`
- Smoke SHA-256:
  `e675ab631742377f1c457dd73ffb22ca98e5b34ecd3497c98e06c998b1ae5fb0`
- Repository commit:
  `2e5a492e0b4c958b0a811fac4deadb90841abd25`
- Synthetic suite SHA-256:
  `16bb1dba2100e9d93a4eff1d2dde4b71d85ac8ac25733fd3c0f8819a083ab527`
- Synthetic result SHA-256:
  `e694adeb23e4667e7a5dcc4705ee74d47bee2bc480294ff8aa86b7cb03710de0`
- Qualification SHA-256:
  `679f13b8ebaf8f05cc97dff8151385643d996ad72f8f2b80b529e70a61416a96`

## Commands and Exact Results

- `python -m pytest tests\test_vasu_140m_base_evaluation_v2.py -q`
  passed: 20 passed in 0.31s.
- `python -m ruff check evaluation\framework\vasu_140m_base_v2.py tests\test_vasu_140m_base_evaluation_v2.py scripts\smoke_vasu_140m_base_evaluation_v2_schema.py`
  passed: all checks passed.
- Frozen qualification comparison passed:
  `$observed` from
  `python scripts\smoke_vasu_140m_base_evaluation_v2_schema.py` matched
  `evaluation\fixtures\vasu_140m_base_evaluation_v2_schema_qualification_precommit.json`
  after canonical JSON compression.
- Direct smoke inspection reported `held_out_files_opened=false`,
  `real_prompts_frozen=false`, `model_invoked=false`,
  `checkpoint_opened=false`, `result_published=false`,
  `training_authorized=false`, and qualification SHA-256
  `679f13b8ebaf8f05cc97dff8151385643d996ad72f8f2b80b529e70a61416a96`.
- `git diff --check` completed with CRLF line-ending warnings only for
  pre-existing modified files.
- `git status --short` showed concurrent modified source-admission files and
  unrelated untracked author-side packages.

## Compatibility Conclusion

The schema implementation is additive. It does not alter existing evaluation
framework consumers, tokenizer assets, model architecture, datasets, masks,
published releases, checkpoints, schedules, configurations, optimizer state,
or training code. Any real suite freezing, held-out opening, model run, or
post-commit identity binding remains separately review-gated.

## Non-Authorization

This acceptance authorizes only committing the reviewed schema implementation
and preparing a separate clean post-commit identity review. It does not
authorize prompt freezing, held-out inventory opening, model execution,
checkpoint opening or creation, source acquisition, production dataset
publication, schedule creation, training configuration, optimizer-state
creation, authorization-record creation, training, commit, or push.
