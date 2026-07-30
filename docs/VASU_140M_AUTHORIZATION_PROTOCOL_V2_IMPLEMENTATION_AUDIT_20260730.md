# VASU-140M Authorization Protocol v2 Implementation Audit

Date: 2026-07-30

Status: implemented; independent review pending; non-authorizing

## Root cause and architecture

Implementing v2 inside the accepted production builder would change its exact
reviewed SHA-256. The gate is therefore isolated in
`vasu/data/vasu_140m_authorization_protocol.py`. A future envelope must bind
both the unchanged builder and the separate gate SHA-256.

The gate owns detached canonical loading, exact runtime validation, exclusive
locking, conservative stale-lock recovery, and the outer transaction that
writes a consumed receipt directly bound to v2 authority. It reuses the
accepted builder's deterministic manifest and staged-byte validation
primitives. It does not create envelopes.

## Implemented guarantees

- Envelope must be canonical UTF-8 JSON, a regular non-link file, and outside
  the repository.
- Exact field set rejects missing, extra, or scope-expanding fields.
- Authorization ID syntax prevents receipt-path traversal.
- Human approver, non-future approval, non-expired approval, and a maximum
  seven-day validity window are required.
- Clean exact runtime `HEAD`, accepted anchor ancestry, unchanged builder hash,
  exact gate hash, qualification hash, assignment hash, template hash, and
  fixed output paths are required.
- Training and overwrite flags must remain false.
- Detached `open("xb")` locking prevents concurrent consumption.
- Stale recovery is opt-in and requires a self-hashed matching lock older than
  24 hours whose recorded process is no longer active.
- Disk exhaustion fails before publication.
- Staged bytes are revalidated before and after an injected mutation point and
  after directory rename.
- Pre-rename failure removes staging; post-rename failure preserves quarantine.
- The consumed receipt directly binds the v2 authorization SHA-256 and gate
  SHA-256.

## Validation evidence

Combined focused suite:

`30 passed`

This covers the 15 accepted builder tests and 15 v2 tests, including canonical
loading, in-repository rejection, exact schemas, dates, clean/wrong runtime
identity, ancestry, code mutation, unsafe IDs, prepared-identity drift,
consumed receipts, concurrent locks, stale-lock recovery, live/mismatched
locks, successful receipt binding, open-handle failure, disk exhaustion,
pre-publication mutation, and post-rename quarantine.

Ruff passed for the implementation and tests. The smoke report reproduced
qualification SHA-256
`f04cf10e31c40027f9e0e97d21822845ac2833945fef45bad0884009380a0d6e`.

The complete repository suite passed with 1,156 tests passed, 8 skipped, and
16 existing CPU-only `pin_memory` warnings. Ruff also passed for the smoke
script, the frozen report reproduced exactly, the earlier post-commit builder
smoke remained green, and `git diff --check` found no whitespace errors.

The frozen report is pre-commit evidence bound to parent `HEAD`
`16e0022059aa70405e0e3bc553215840de596ab5`. A separately reviewed post-commit
identity qualification remains mandatory after implementation acceptance and
commit.

## Compatibility

- Accepted production-builder file and SHA-256: unchanged.
- Release bytes, tokenizer, datasets, masks, checkpoints, and optimizer state:
  unchanged.
- Existing v1 builder tests and historical evidence: retained.
- Exact resume: unaffected.
- Training authority: none.

## Current gate

GPT-5.5 independently accepted the exact pre-commit implementation on
2026-07-30. Acceptance authorizes only an evidence-preserving implementation
commit. It does not authorize creation of an envelope or production
publication. A separate post-commit identity review remains mandatory before
human authorization is eligible.

The reviewer recorded one optional hardening item: linked-envelope rejection
is implemented and was confirmed by code inspection, but an explicit
symlink/junction test may be added only as a separately reviewed future change.
