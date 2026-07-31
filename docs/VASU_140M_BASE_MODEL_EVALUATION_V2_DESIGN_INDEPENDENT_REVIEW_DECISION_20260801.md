# VASU-140M Base-Model Evaluation Suite v2 Design Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5 independent review

## Decision

Accept the VASU-140M base-model evaluation suite v2 design for later
implementation review.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a pre-existing dirty author-side worktree:
  `docs/CHANGELOG.md`, `docs/PROJECT_STATUS.md`, and `docs/ROADMAP.md` were
  already modified. Several unrelated VASU-140M planning documents were also
  untracked. This review modified only this decision file.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_READINESS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_DESIGN_REVIEW_PACKET.md`
- `evaluation/suites/vasu_capability_v1.json`
- `evaluation/framework/schemas.py`
- `evaluation/framework/generation.py`
- `evaluation/framework/runner.py`
- `evaluation/framework/reporting.py`
- `evaluation/framework/registry.py`
- `evaluation/framework/scoring.py`
- `evaluation/metrics.py`
- `evaluation/comparison.py`
- `tests/test_capability_framework.py`
- `tests/test_evaluation_metrics.py`
- `tests/test_evaluation_comparison.py`

## Rationale

The design is specifically scoped to decoder-only base-model evaluation through
raw continuation and direct likelihood interfaces. It explicitly excludes
assistant roles, instruction templates, and instruction-tuned checkpoints as
direct controls.

The design separates likelihood, factuality, arithmetic,
repetition/degeneration, robustness, and manual review. It requires
development and held-out inventory isolation, held-out sealing until the
declared decision point, and prompt-contamination scanning before data
splitting. It also requires direct likelihood scoring for choices and target
continuations rather than inferring likelihood from generated text.

The identity model is adequate for a future implementation: suite manifests
must bind schema/version, repository commit, family/configuration, tokenizer,
prompt and dataset files, scorer code, generation settings, seeds, and
canonical manifest hash. Result manifests must additionally bind checkpoint,
environment, device, precision, command, timestamps, and output shards.
Interrupted evaluation is specified to resume only under identical suite,
checkpoint, tokenizer, scorer, and runtime identities.

The statistical reporting requirements are sound: paired tasks use paired
differences, accuracy and rates require deterministic bootstrap 95 percent
confidence intervals, strata remain separately reported, and metrics must not
be collapsed into a single misleading capability score. Random initialization
and VASU-60M references are allowed only as labeled calibration points, not as
matched scientific parents.

The existing `vasu_capability_v1` suite is correctly treated as infrastructure
evidence rather than reused as the VASU-140M base-model contract. It contains
instruction-style prompts and promotion gates that are not sufficient for this
base-model evaluation design. Existing framework code provides useful
building blocks for deterministic generation, identity-bound resume checks,
atomic reporting, separated objective/heuristic/human reporting, and paired
bootstrap comparison, but the v2 suite, schemas, prompts, and qualification
remain future review-gated work.

## Validation Results

- `python -m pytest tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q`
  passed: 47 tests passed.
- `git diff --check` passed with line-ending warnings only for pre-existing
  modified documentation files.
- `git status --short` showed pre-existing modified documentation files and
  untracked planning documents, plus this decision file after creation.
- Targeted path checks found no frozen VASU-140M base-model v2 suite, held-out
  inventory, model result directory, base-pretraining data release, 140M
  checkpoint directory, training configuration, or schedule at the expected
  paths.

## Compatibility Conclusion

The design is documentation-only and changes no model architecture, tokenizer,
dataset, mask, checkpoint, schedule, optimizer state, evaluation result, or
existing evaluation framework behavior. Existing `vasu_capability_v1` tooling
and tests remain compatible. Future implementation must be additive and
separately reviewed before any suite or prompt inventory is frozen.

## Non-Authorization

This acceptance authorizes only later implementation review. It does not
freeze prompts, open held-out inventories, run a model, construct data, create
a data release, create a checkpoint, create a configuration, create a schedule,
create optimizer state, create an authorization record, authorize training,
commit, or push.
