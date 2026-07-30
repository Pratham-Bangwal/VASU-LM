# VASU-140M Production Release Authorization Protocol Independent Review Decision

Status: accepted as a protocol design; non-authorizing.

Review date: 2026-07-30

Reviewer: GPT-5.5 independent review

## Decision

Accept the detached two-identity one-build authorization protocol.

Acceptance covers only the protocol design. It does not authorize implementation
of the protocol, creation of an authorization envelope, production publication,
checkpoint selection, schedule creation, training configuration, optimizer
creation, or training.

## Findings With Severity

- Blocking: none.
- High: none.
- Medium: none.
- Low: the review packet lists
  `evaluation/fixtures/vasu_140m_instruction_seed_v1_production_postcommit_qualification.json`,
  which is absent. The intended post-commit evidence is present at
  `evaluation/fixtures/vasu_140m_instruction_seed_v1_production_qualification_postcommit_c014716.json`
  and matches the accepted post-commit decision, so this is a packet path typo
  rather than a protocol blocker.
- Low: this is specification-only acceptance. A future implementation review
  must still prove the v2 detached-envelope loader, rejection of in-repository
  envelopes, exact runtime commit enforcement, exclusive locking, stale-lock
  recovery, concurrent-use failure, and all crash and mutation cases in code.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_AUTHORIZATION_PROTOCOL.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_AUTHORIZATION_PROTOCOL_REVIEW_PACKET.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_AUTHORIZATION_PROTOCOL_AUDIT_20260730.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_BUILDER_DESIGN.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_BUILDER_DESIGN_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_BUILDER_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `vasu/data/vasu_140m_production_release.py`
- `tests/test_vasu_140m_production_release.py`
- `scripts/smoke_vasu_140m_postcommit_qualification.py`
- `evaluation/fixtures/vasu_140m_instruction_seed_v1_production_qualification_postcommit_c014716.json`

## Rationale

The protocol correctly removes circular Git identity by keeping the future
human approval envelope detached from tracked Git state. It separately binds
the accepted implementation anchor commit, exact implementation-file SHA-256,
and a future exact clean runtime commit. The ancestry requirement prevents the
runtime commit from bypassing the reviewed implementation history, while the
implementation SHA-256 prevents unreviewed code changes on a descendant commit.

The proposed envelope is appropriately narrow: fixed release ID, fixed scope,
fixed output paths, fixed qualification and assignment identities, fixed
manifest-template identity, explicit human approver, bounded expiration,
canonical self-hash, overwrite disabled, and `training_authorized=false`.
Receipt presence and authorization SHA-256 binding make successful authority
single-use. Unknown fields are rejected in v2, preventing silent scope
expansion.

The fail-closed requirements cover the important construction hazards for a
future implementation review: dirty or wrong runtime `HEAD`, wrong anchor,
wrong implementation hash, mismatched qualification, source/assignment/template
drift, existing protected paths, reuse, mutation between preflight and
publication, incomplete publication, Windows path traversal through links or
junctions, disk failure, locking, stale-lock recovery, and concurrent envelope
use.

The design keeps review, human approval, and explicit publication invocation
separate. Independent acceptance is not represented as approval, and the
protocol explicitly states that a valid envelope alone is insufficient without
a separate explicit instruction to invoke publication.

## Commands Run And Results

- `git status --short`: inspected; unrelated documentation was already
  modified and the protocol documents were untracked before this decision.
- `Get-FileHash docs\VASU_140M_PRODUCTION_RELEASE_AUTHORIZATION_PROTOCOL.md -Algorithm SHA256`:
  matched `6334ae0adc53b9a1e9009735b47e3571ff3a7f198b6c4006302ea1450f0b01df`.
- `Get-FileHash docs\VASU_140M_PRODUCTION_RELEASE_AUTHORIZATION_PROTOCOL_REVIEW_PACKET.md -Algorithm SHA256`:
  matched `2e15fd0fa3d579fd5ef4aa8a4a793bbb2205d5c9b39f043367d9cc370d452218`.
- `Get-FileHash vasu\data\vasu_140m_production_release.py -Algorithm SHA256`:
  matched `0efb0189b4b0c5a8621da9053d56db57d95ebf5b0f1bb59753952ec8afa5d1e2`.
- `git rev-parse HEAD`: returned
  `6ef4d0c34cdbae776877e94ca420afefff3e5b5d`.
- `git merge-base --is-ancestor c014716ec38ef8f08842356fc016359dc5a233d7 HEAD`:
  passed; the accepted implementation anchor is an ancestor of current `HEAD`.
- `python scripts\smoke_vasu_140m_postcommit_qualification.py | python -m json.tool > $null`:
  passed; the read-only post-commit qualification reproduced the frozen
  c014716 evidence.
- Protected path check: `data/processed/vasu_140m/instruction_seed_v1`,
  `data/manifests/vasu_140m/instruction_seed_v1.json`, and
  `data/manifests/vasu_140m/authorization_receipts` are absent.
- `git diff --check`: passed with no whitespace errors; Git reported only CRLF
  conversion warnings for existing modified docs.

## Qualification Identity Confirmation

The frozen post-commit qualification evidence retains:

- qualification SHA-256:
  `ed6f64b9d5cb97925bc68186b4ec403de6e25dc484e33cb103ed05ee977f5cd6`;
- implementation SHA-256:
  `0efb0189b4b0c5a8621da9053d56db57d95ebf5b0f1bb59753952ec8afa5d1e2`;
- assignment SHA-256:
  `59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394`;
- decoded round trips: 996;
- split counts: 898 train, 48 development, 50 evaluation;
- `production_release_created=false`;
- `release_build_permitted=false`;
- `training_authorized=false`.

## Non-Authorization Confirmation

This decision does not authorize creation of an authorization envelope,
production publication, production release construction, external-manifest
publication, receipt creation, checkpoint selection, schedule creation,
training configuration, optimizer creation, or training. This review did not
implement the protocol, create an envelope, invoke the publisher, create
protected artifacts, train, commit, or push.
