# VASU-140M Base-Model Evaluation v2 Schema Remediation Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the remediated VASU-140M base-model evaluation v2 schema package for an
isolated remediation commit and later clean post-commit identity review.

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
  because the remediated schema package is internally consistent,
  non-authorizing, and the requested focused tests and fixture replay pass.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_IMPLEMENTATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_REMEDIATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_REMEDIATION_REVIEW_PACKET.md`
- `evaluation/framework/vasu_140m_base_v2.py`
- `tests/test_vasu_140m_base_evaluation_v2.py`
- `scripts/smoke_vasu_140m_base_evaluation_v2_schema.py`
- `evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_qualification_rejected_2d73fcb.json`
- `evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_qualification_precommit.json`

## Rationale

The first rejection remains preserved as a separate historical artifact. The
rejected fixture
`evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_qualification_rejected_2d73fcb.json`
is still present with SHA-256
`443ee3a5decdf2c491a609613d9851ebb031c67d412ca53baf76f0f02e20439e`
and records the rejected commit-bound qualification
`8870976dc9f7c76e5da3eacdb374eedc1a4b1a2fa74af0d41789c4ff2be38b14`
at commit `2d73fcb766c76dffb7ceb8b615d7162b79262176`. This review did not
overwrite that fixture or edit the preserved rejection evidence.

The remediated qualification reproduces exactly at commit
`51943a7a326b836755b2edaa7b7fd26ba0da366d`. The current precommit fixture has
SHA-256 `4080eddbfe2d5c9355ee8de438a5594053e6b59b990a1d7196534eccfed908deb`
and qualification SHA-256
`d82bb1408d5b28ff9599a535c4a94c6b0a9c8e6db638bc03be48556787e178d8`.

The remediation addresses the result-schema defect identified after the first
review. Likelihood and factuality are restricted to `direct_likelihood` result
mode, so they cannot be mislabeled as generated results. Arithmetic,
repetition, and robustness require separate greedy and sampled result
coverage. Output shards and dimension reports must use the declared
`evaluation_stage`, so development and held-out results cannot be mixed.

Development results must carry the canonical empty held-out authorization.
Held-out results require explicit authorization with decision ID, path, and
SHA-256. Dimension-report split and mode coverage must match output-shard
coverage, while duplicate metric reports remain rejected. Result manifests
bind the exact suite and tokenizer through `validate_result_against_suite`.
Resume identity now includes evaluation stage and held-out authorization
identity.

The validator still fails closed on aggregate score fields, wrong-family
checkpoints, non-finite statistics, reversed timestamps, unhashed mutations,
unsafe or unknown fields, and `training_authorized=true`.

The synthetic qualification remains prompt-free and non-executing: it verifies
development file bindings only, does not open held-out files, does not freeze
real prompts, does not invoke a model, does not open a checkpoint, does not
publish results, and does not authorize training.

## Commands and Exact Results

- `python -m pytest tests\test_vasu_140m_base_evaluation_v2.py tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q`
  passed: 67 passed in 2.99s.
- `python -m ruff check evaluation\framework\vasu_140m_base_v2.py tests\test_vasu_140m_base_evaluation_v2.py scripts\smoke_vasu_140m_base_evaluation_v2_schema.py`
  passed: all checks passed.
- Remediated fixture replay passed exactly:
  `$observed` from
  `python scripts\smoke_vasu_140m_base_evaluation_v2_schema.py` matched
  `evaluation\fixtures\vasu_140m_base_evaluation_v2_schema_qualification_precommit.json`
  after canonical JSON compression.
- `git diff --check`
  completed with CRLF line-ending warnings only for existing modified files.
- `git status --short`
  showed the known concurrent author-side modified and untracked packages,
  with no commit or push performed.

## Exact Identities

- Current repository commit:
  `51943a7a326b836755b2edaa7b7fd26ba0da366d`
- Implementation SHA-256:
  `b3d889168196b3c880eb99f9868f14105d6b4b3b5cc84010add96ffb77997c49`
- Test SHA-256:
  `313126c35430372832a9f19f463d15e01a3b054ad0a7f6c15027cf7813174319`
- Smoke SHA-256:
  `e675ab631742377f1c457dd73ffb22ca98e5b34ecd3497c98e06c998b1ae5fb0`
- Preserved rejected fixture SHA-256:
  `443ee3a5decdf2c491a609613d9851ebb031c67d412ca53baf76f0f02e20439e`
- Current fixture SHA-256:
  `4080eddbfe2d5c9355ee8de438a5594053e6b59b990a1d7196534eccfed908deb`
- Synthetic suite SHA-256:
  `1f977b37b68249967164ac8cc0faeded0d8b999a6dca1dd11d7622198fd8e79a`
- Synthetic result SHA-256:
  `df62dfc9407abf2598de1632144e4ab93e69cd0257000d0f467372cd44a20285`
- Current qualification SHA-256:
  `d82bb1408d5b28ff9599a535c4a94c6b0a9c8e6db638bc03be48556787e178d8`

## Compatibility Conclusion

The remediation is additive and pre-commit. It tightens the synthetic
evaluation-v2 result schema without changing existing evaluation outputs,
tokenizer assets, datasets, masks, model architecture, checkpoints, schedules,
training configurations, optimizer state, or exact-resume state. Future real
suite freezing, held-out opening, model execution, result publication, and
post-commit identity binding remain separately review-gated.

## Non-Authorization

This acceptance authorizes only committing the remediated schema package and
later preparing a clean post-commit identity review. It does not authorize
freezing real prompts, opening held-out data, invoking a model, opening or
creating checkpoints, publishing evaluation results, constructing data,
creating configurations, creating schedules, creating optimizer state,
creating authorization records, authorizing training, training, commit, or
push.
