# VASU-140M Authorization Protocol v2 Implementation Independent Review Decision

Status: accepted as pre-commit implementation evidence; non-authorizing.

Review date: 2026-07-30

Reviewer: GPT-5.5 independent review

## Decision

Accept the VASU-140M authorization protocol v2 implementation as eligible for
commit and later post-commit runtime-identity review.

This acceptance does not authorize creation of an authorization envelope,
production publication, protected artifact creation, checkpoint selection,
schedule creation, training configuration, optimizer creation, or training.

## Findings With Severity

- Blocking: none.
- High: none.
- Medium: none.
- Low: the frozen v2 qualification is intentionally pre-commit evidence bound
  to repository commit `16e0022059aa70405e0e3bc553215840de596ab5`. After this
  accepted implementation is committed, a separate clean-runtime post-commit
  qualification and independent identity review remain mandatory before any
  human approval envelope is eligible.
- Low: the test suite verifies external canonical and in-repository envelope
  rejection, while linked-envelope rejection was confirmed by direct code
  inspection of `load_detached_authorization`. A future hardening pass may add
  an explicit symlink/junction envelope test, but the implementation currently
  rejects symlink envelope paths before and after resolution.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_AUTHORIZATION_PROTOCOL_V2_IMPLEMENTATION_REVIEW_PACKET.md`
- `docs/VASU_140M_AUTHORIZATION_PROTOCOL_V2_IMPLEMENTATION_AUDIT_20260730.md`
- `vasu/data/vasu_140m_authorization_protocol.py`
- `tests/test_vasu_140m_authorization_protocol.py`
- `scripts/smoke_vasu_140m_authorization_protocol.py`
- `evaluation/fixtures/vasu_140m_authorization_protocol_v2_qualification_precommit.json`
- `docs/VASU_140M_PRODUCTION_RELEASE_AUTHORIZATION_PROTOCOL.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_AUTHORIZATION_PROTOCOL_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_BUILDER_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `vasu/data/vasu_140m_production_release.py`
- `tests/test_vasu_140m_production_release.py`
- `evaluation/fixtures/vasu_140m_instruction_seed_v1_production_qualification_postcommit_c014716.json`

## Rationale

Independent inspection found that the v2 gate is correctly isolated from the
accepted production builder, preserving the builder SHA-256
`0efb0189b4b0c5a8621da9053d56db57d95ebf5b0f1bb59753952ec8afa5d1e2`.
The gate loads only detached canonical JSON from outside the repository,
rejects exact-field deviations, validates the canonical self-hash, and binds a
safe authorization ID to the fixed receipt path.

`validate_authorization_v2` fail-closes on wrong schema, release, scope,
runtime commit, accepted anchor, builder path and SHA-256, gate path and
SHA-256, qualification SHA-256, assignment SHA-256, external-manifest-template
SHA-256, output paths, overwrite permission, training flag, dates, dirty
runtime state, missing ancestry, consumed receipt, and existing protected
outputs. It also verifies the on-disk builder and gate file hashes before
publication.

The lock design uses detached atomic `open("xb")` creation and binds lock
contents to authorization ID and authorization SHA-256. Stale recovery is
opt-in and requires a self-hashed matching lock older than 24 hours with an
inactive process owner. The v2 transaction reuses the accepted builder's staged
artifact validation and preserves pre-publication cleanup, disk-failure
rejection, open-handle failure cleanup, mutation detection, post-rename
quarantine, and final receipt binding.

The persisted receipt directly records the v2 `authorization_sha256`,
`authorization_protocol_sha256`, runtime commit, unchanged builder SHA-256,
qualification SHA-256, final external-manifest SHA-256, completion state, and
`training_authorized=false`.

No API in the reviewed module creates an authorization envelope, approves
publication, selects a checkpoint, creates a schedule or training
configuration, creates optimizer state, or starts training.

## Commands Run And Results

- `python -m pytest tests\test_vasu_140m_authorization_protocol.py tests\test_vasu_140m_production_release.py -q`:
  passed, `30 passed`.
- `python -m ruff check vasu\data\vasu_140m_authorization_protocol.py tests\test_vasu_140m_authorization_protocol.py scripts\smoke_vasu_140m_authorization_protocol.py`:
  passed, `All checks passed!`.
- `python scripts\smoke_vasu_140m_authorization_protocol.py | python -m json.tool > $null`:
  passed and emitted valid JSON.
- Exact smoke-vs-fixture comparison:
  passed, observed report equals
  `evaluation/fixtures/vasu_140m_authorization_protocol_v2_qualification_precommit.json`.
- `git diff --check`:
  passed with no whitespace errors; Git reported only CRLF conversion warnings
  for existing modified docs.
- `git status --short`:
  inspected. Existing modified docs remain, and the v2 implementation packet,
  audit, smoke, tests, fixture, and this decision document are untracked.

## Identity Verification

- authorization-gate SHA-256:
  `d135889b89b424e7a3253adddec0b1cfd09b49efdb6831f048934f2468d9acac`;
- test SHA-256:
  `ccda470cff74a2911cd7cbb4d5d993b0f6f7797b8da32c519cff341beebcc101`;
- smoke SHA-256:
  `cbb2cd2ebb9692c4615ea16020d11cd5b0b7d2ad86c1d5facc1120af7a90506a`;
- frozen qualification SHA-256:
  `f04cf10e31c40027f9e0e97d21822845ac2833945fef45bad0884009380a0d6e`;
- unchanged builder SHA-256:
  `0efb0189b4b0c5a8621da9053d56db57d95ebf5b0f1bb59753952ec8afa5d1e2`.

The frozen qualification also records
`authorization_envelope_created=false`,
`production_release_created=false`, `publication_authorized=false`, and
`training_authorized=false`.

## Protected Path Confirmation

Confirmed absent after review:

- `data/processed/vasu_140m/instruction_seed_v1`
- `data/manifests/vasu_140m/instruction_seed_v1.json`
- `data/manifests/vasu_140m/authorization_receipts`

## Non-Authorization Confirmation

This decision does not authorize creation of an authorization envelope,
production publication, production release construction, protected artifact
creation, external-manifest publication, receipt creation against `D:\VASU`,
checkpoint selection, schedule creation, training configuration, optimizer
creation, or training. This review did not create a real authorization
envelope, invoke publication against `D:\VASU`, create protected artifacts,
train, commit, or push.
