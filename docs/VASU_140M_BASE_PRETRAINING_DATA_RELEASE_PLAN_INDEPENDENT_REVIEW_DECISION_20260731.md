# VASU-140M Base-Pretraining Data-Release Plan Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-07-31

Reviewer: GPT-5.5 independent review

## Decision

Accept the VASU-140M base-pretraining data-release plan.

The plan is source-agnostic, selects no source, constructs no data, and
correctly preserves the VASU-140M data compatibility boundary. It rejects
automatic reuse of VASU-60M 257-token FineWeb binaries and the published
VASU-140M instruction seed, and it requires a separately reviewed,
provenance-bound, full-loss 513-token base-pretraining release before any
future experiment can be proposed.

## Findings With Severity

- Blocking: none.
- High: none.
- Medium: none.
- Low: `tests/test_vasu_140m_release_plan.py` has five historical failures
  because it still requires `data/processed/vasu_140m/instruction_seed_v1` to
  be absent. That path now exists due to the one authorized instruction-seed
  publication. The failures match the expected post-publication evidence
  described in the prompt and audit; no unrelated failure was observed.
- Low: the worktree already contains modified `docs/CHANGELOG.md` and
  `docs/PROJECT_STATUS.md`, plus untracked author-side data-release plan
  files. This does not affect the plan review because the plan is
  non-authorizing and protected base-pretraining outputs remain absent.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_READINESS.md`
- `docs/VASU_140M_513_TOKEN_DATA_MASK_SPECIFICATION.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN_AUDIT_20260731.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN_REVIEW_PACKET.md`
- `docs/DATASET.md`
- `data/processed/pretrain/fineweb_manifest.json`

## Command Results

- `python -m pytest tests\test_vasu_140m_records.py -q`: passed,
  `19 passed`.
- `python scripts\smoke_vasu_140m_record_spec.py | python -m json.tool`:
  passed. The smoke reported `vasu_140m_v1`, record width `513`, fixture-only
  evidence, deterministic rebuild, split isolation, shifted-mask alignment,
  `production_release_created=false`, and `training_authorized=false`.
- `python -m pytest tests\test_vasu_140m_release_plan.py -q`: expected
  post-publication result, `5 failed, 2 passed`. All five failures raise
  `ValueError: planned production output already exists:
  data/processed/vasu_140m/instruction_seed_v1`. No unrelated failure was
  observed.
- `git diff --check`: passed with no whitespace errors; Git reported CRLF
  conversion warnings for existing modified docs.
- `git status --short`: inspected. Existing modified docs and untracked
  author-side data-release plan files are present, plus this decision document
  after acceptance.

## Plan Review Conclusion

The plan satisfies the requested checks:

- selects no source and creates no data;
- rejects automatic reuse of VASU-60M 257-token FineWeb binaries;
- rejects use of the published VASU-140M instruction seed as base-pretraining
  text;
- requires source provenance, license, immutable revision or snapshot, raw
  content hashes, acquisition details, retention constraints, and permitted
  use before processing;
- requires pinned contamination inventories, exact and word-ngram comparison,
  quarantine of collisions, cross-source deduplication, cross-split
  deduplication, and document-level split isolation;
- requires full-loss 513-token `uint16` records with shifted full-loss masks,
  PAD/boundary masking, split-local packing, decoded round trips, complete
  serialized mask audits, and deterministic rebuild comparison;
- requires atomic, non-overwriting publication into a new versioned
  `vasu_140m/base_pretraining/*` output family;
- requires a separate builder implementation review and one-build publication
  authorization.

The existing `data/processed/pretrain/fineweb_manifest.json` is a 256-token
VASU-60M-era FineWeb view and is not sufficient for VASU-140M base-pretraining
release construction.

## Protected Artifact Check

The following protected or training-related paths were absent at review time:

- `data/processed/vasu_140m/base_pretraining`
- `data/manifests/vasu_140m/base_pretraining`
- `configs/training/vasu_140m_base_pretraining.json`
- `configs/schedules/vasu_140m_base_pretraining.json`
- `checkpoints/vasu_140m`

## Non-Authorization Confirmation

This acceptance authorizes only retaining the source-agnostic data-release
plan. It does not authorize source discovery, acquisition, processing, data
release construction, experiment configuration, schedules, optimizer creation,
authorization records, checkpoint creation, base pretraining, instruction
tuning, commit, or push. No data was constructed, no protected artifact was
created, no training occurred, no authorization changed, no commit was
created, and no push was performed.
