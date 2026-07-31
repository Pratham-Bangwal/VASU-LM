# VASU-140M Base-Pretraining Release Foundation Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the VASU-140M full-loss base-record and release-builder foundation for
an isolated commit and later production-builder implementation review.

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
  not treated as a package blocker because the foundation is internally
  non-authorizing and its tests/fixture identities are self-contained.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_FOUNDATION_REVIEW_PACKET.md`
- `docs/VASU_140M_513_TOKEN_DATA_MASK_SPECIFICATION.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN_INDEPENDENT_REVIEW_DECISION_20260731.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_BUILDER_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_FOUNDATION_AUDIT_20260801.md`
- `vasu/data/vasu_140m_records.py`
- `vasu/data/vasu_140m_base_records.py`
- `tests/test_vasu_140m_records.py`
- `tests/test_vasu_140m_base_records.py`
- `scripts/smoke_vasu_140m_record_spec.py`
- `scripts/smoke_vasu_140m_base_records.py`
- `evaluation/fixtures/vasu_140m_513_record_spec_v1.json`
- `evaluation/fixtures/vasu_140m_base_text_record_fixture_v1.json`

## Rationale

The foundation correctly separates base-pretraining full-loss text from the
existing response-masked instruction record contract. `BaseTextChunk` retains
source, revision, parent document, document hash, transformation, chunk ID,
chunk index, split, text hash, tokens, and stored mask. Content rejects PAD,
UNK, BOS, and EOS, appends EOS, masks only the first chunk token, supervises
later content and EOS, and relies on `stored_mask[1:]` so artificial
cross-chunk and PAD targets remain unsupervised.

The stateful stream validator rejects duplicate chunk IDs, duplicate exact
text, parent documents crossing splits, source-revision drift, and duplicate
parent chunk indexes. The streaming packer uses the accepted 513-token packer
without corpus-wide packed-record materialization and preserves split-local
packing. The frozen base-text fixture uses the real tokenizer and reproduced
report SHA-256
`7c4e4710cc70bb39124f8b1445516aac801401bc2c263175b6472dba125aa578`.

The accompanying builder design is source-separated, streaming, two-pass,
atomic, non-overwriting, detached two-person one-build authorization gated,
and explicitly excludes schedules, configurations, checkpoints, optimizer
state, and training.

## Commands and Exact Results

- `python -m pytest tests\test_vasu_140m_base_records.py tests\test_vasu_140m_records.py -q`
  passed: 34 passed in 0.91s.
- `python -m ruff check vasu\data\vasu_140m_base_records.py tests\test_vasu_140m_base_records.py scripts\smoke_vasu_140m_base_records.py`
  passed: all checks passed.
- `python scripts\smoke_vasu_140m_base_records.py | python -m json.tool > $null`
  passed; JSON was valid. Direct smoke inspection reported
  `fixture_only=true`, `source_discovered=false`, `data_acquired=false`,
  `production_release_created=false`, `training_authorized=false`, and report
  SHA-256 `7c4e4710cc70bb39124f8b1445516aac801401bc2c263175b6472dba125aa578`.
- `git diff --check` completed with CRLF line-ending warnings only for
  pre-existing modified files.
- `git status --short` showed concurrent modified source-admission files and
  unrelated untracked author-side packages.

## Compatibility Conclusion

The foundation is additive. It does not modify the frozen instruction record
contract, tokenizer identity, model architecture, existing VASU-31M/60M data,
published instruction release, checkpoints, schedules, training
configuration, or optimizer state. A future production builder remains
separately review-gated.

## Non-Authorization

This acceptance authorizes only committing the reviewed foundation and later
implementing the production builder for separate review. It does not authorize
source admission, source acquisition, production dataset publication, schedule
creation, training configuration, checkpoint creation, optimizer-state
creation, authorization-record creation, training, commit, or push.
