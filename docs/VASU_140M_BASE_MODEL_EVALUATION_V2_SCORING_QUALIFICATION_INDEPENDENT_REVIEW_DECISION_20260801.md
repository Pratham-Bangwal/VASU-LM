# VASU-140M Base-Model Evaluation v2 Scoring Qualification Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the VASU-140M evaluation-v2 scoring qualification for an isolated
commit and later clean post-commit identity review.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree with unrelated
  modified files and multiple untracked VASU-140M packages. This was not
  treated as a scoring-package blocker because the package identities,
  validation commands, and frozen fixture replay are self-contained.
- `git diff --check` reported CRLF conversion warnings for existing modified
  files only; it reported no whitespace errors.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_REMEDIATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCHEMA_REMEDIATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_REVIEW_PACKET.md`
- `evaluation/framework/vasu_140m_base_v2.py`
- `evaluation/framework/vasu_140m_base_v2_tasks.py`
- `evaluation/framework/vasu_140m_base_v2_statistics.py`
- `tests/test_vasu_140m_base_v2_scoring.py`
- `scripts/smoke_vasu_140m_base_v2_scoring.py`
- `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json`

## Rationale

The task contract uses exact fields and prompt/text hashes only. It rejects
unknown fields, duplicate task IDs, duplicate prompt/text identities,
duplicate robustness pairs, missing required strata, wrong task dimensions,
and observation modes that do not match the task dimension.

The scorer covers all six required evaluation dimensions without model
execution. Likelihood uses direct token log-likelihoods with finite,
non-positive per-token validation and token-weighted loss/perplexity.
Factuality scores multiple-choice likelihood directly and reports raw and
length-normalized predictions while surfacing ties instead of silently
breaking them. Arithmetic preserves strict correct, incorrect, malformed,
unanswered, truncation, and prompt-leakage outcomes. Repetition records
token-level degeneration metrics, including unique-token ratio, repeated
n-gram rates, EOS, empty output, truncation, and terminal loops. Robustness is
paired and reports consistency, empty-output visibility, and paired
correctness deltas. Manual review produces blank deterministic review packets
and rejects injected human judgments.

The statistics layer keeps dimension/mode summaries separate, reports complete
strata, uses deterministic bootstrap confidence intervals, supports paired
robustness comparisons, and deliberately provides no aggregate capability
score. The qualification requires all nine required dimension/mode pairs,
binds compact evidence to the complete report, and validates all
non-authorization flags.

The smoke fixture is prompt-free and contains synthetic hashes and
observations only. It does not use a real production prompt inventory, does
not open held-out content, does not invoke a model, does not open or create a
checkpoint, and does not create datasets, schedules, optimizers, or training
state.

## Commands and Exact Results

- `git status --short`
  showed a concurrent author-side worktree with unrelated modified files and
  untracked VASU-140M packages, including this scoring qualification package.
- `python -m pytest tests\test_vasu_140m_base_v2_scoring.py tests\test_vasu_140m_base_evaluation_v2.py tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q`
  passed: `96 passed in 3.10s`.
- `python -m ruff check evaluation\framework\vasu_140m_base_v2_tasks.py evaluation\framework\vasu_140m_base_v2_statistics.py tests\test_vasu_140m_base_v2_scoring.py scripts\smoke_vasu_140m_base_v2_scoring.py`
  passed: `All checks passed!`.
- Frozen compact fixture replay:
  `$observed = (python scripts\smoke_vasu_140m_base_v2_scoring.py --evidence-only | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)`;
  `$expected = (Get-Content evaluation\fixtures\vasu_140m_base_v2_scoring_qualification_v1.json -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)`;
  `if ($observed -ne $expected) { throw "scoring qualification fixture mismatch" }`
  passed: `scoring qualification fixture matched`.
- `python scripts\smoke_vasu_140m_base_v2_scoring.py | python -m json.tool > $null`
  passed: full scoring qualification JSON was valid.
- `git diff --check`
  completed with CRLF conversion warnings only for existing modified files and
  no whitespace errors.

## Verified Identities

- Parent commit / repository commit:
  `eafd4d708415116663a5c1b1909cc94a6e7b00b3`.
- Existing suite/result schema implementation SHA-256:
  `b3d889168196b3c880eb99f9868f14105d6b4b3b5cc84010add96ffb77997c49`.
- Task/scorer implementation SHA-256:
  `7777bd7d6a488c21e3937b5b05e1e46153e7fddbcb67784c47dc9e53a5087bf5`.
- Statistics/qualification implementation SHA-256:
  `5d4c07d01fe7b833a2b6daf06fb562061ed0731160ffbd0e031eba3527568d71`.
- Test SHA-256:
  `c71190f99822d1c5a887aa64bc1b22c5c2b57f465124ee34053d7ec2a8a62b34`.
- Smoke SHA-256:
  `6e43664e5eb1d8f39916ff63f1425ebbb3f339ee69ec6dceb0ec21dcb979e074`.
- Frozen compact fixture SHA-256:
  `0ff1801e11d96036a7c645b752754955eda0e4dc5a640c7c57ad1f2e918cfe26`.
- Full qualification SHA-256:
  `5fbb82d7eabd6dc025b595776c1dafa814c117785af6f7e1821663b82d59c32c`.
- Compact evidence SHA-256:
  `0790f5ef1fd301968f8fbb3e430e32c213128b63365ca7719139fa33e21e0f1b`.
- Task inventory SHA-256:
  `82447a39cbbabda6259735befdba1b0719073c4daeb387801305329e554383ef`.
- Observation inventory SHA-256:
  `2dc6f263355270d8fe1d19fc71bddc319779068d5a52b5ae0baa0db167121599`.
- Score rows SHA-256:
  `7ef49b1ad20d0b01e56d6a505beb637a7f9ac5383e13c28eb09cfa2b5a3b0323`.

The frozen compact fixture reproduced exactly.

## Compatibility Assessment

The package is additive. It does not change model architecture, tokenizer
assets, datasets, masks, checkpoints, optimizer or scheduler state, training
configuration, existing evaluation outputs, or exact-resume behavior. It
qualifies prompt-free scorer/statistics behavior only; real inventory
construction, suite freezing, held-out opening, model execution, result
publication, and post-commit identity remain separate review gates.

## Non-Authorization

This acceptance authorizes only a later isolated commit and clean post-commit
identity review. It does not authorize prompt construction, production suite
freezing, held-out encryption or opening, source acquisition, data
construction, model execution, checkpoint access or creation, evaluation
publication, configuration creation, schedule creation, optimizer updates,
authorization records, training, commit, or push.
