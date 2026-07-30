# VASU-140M Fixture Release Construction Independent Review Decision

Status: accepted as fixture-only construction evidence; non-authorizing.

Review date: 2026-07-30

Reviewer: GPT-5.5 independent review

## Decision

Accept the fixture-only transactional construction layer as evidence for a
later, separately gated production release-builder design.

This acceptance does not permit construction of the 996-example production
release, source discovery, schedule generation, training configuration,
checkpoint selection, authorization, or model training.

## Materials Reviewed

- `vasu/data/vasu_140m_fixture_release.py`
- `tests/test_vasu_140m_fixture_release.py`
- `vasu/data/vasu_140m_records.py`
- `configs/data/releases/vasu_140m_instruction_seed_v1.plan.json`
- `docs/VASU_140M_INSTRUCTION_SEED_RELEASE_INDEPENDENT_REVIEW_DECISION.md`
- `docs/VASU_140M_FIXTURE_RELEASE_CONSTRUCTION_REVIEW_PACKET.md`
- `docs/VASU_140M_FIXTURE_RELEASE_CONSTRUCTION_AUDIT_20260730.md`
- `docs/PROJECT_STATUS.md`

## Rationale

Independent inspection found the fixture layer appropriately bounded to
caller-supplied logical examples with a hard 30-example cap. The construction
path rejects the planned production release directory and production manifest
containment, requires a fixture-named non-existing destination with an existing
parent, writes all token, mask, and manifest artifacts into an isolated staging
directory, and publishes only by renaming the completed staging directory into
place. Existing destinations, wrong plan identity, wrong independent-decision
identity, oversized fixtures, split leakage through the reused record-contract
validator, and artifact tampering fail closed.

The fixture manifest is bound to the accepted instruction seed plan SHA-256
`8da8cc92ef913bb567c96ec47986f849b84eb85ac2d11f3ad245fb1639361e16` and the
accepted plan decision SHA-256
`4522d1c36a73700da9f3eec7cdd87561fded1a8f8661fa0ef88ab18a89058890`. Published
artifacts are checked by file size, SHA-256, local filename binding, and token
and mask layout. The reused 513-token record contract preserves `uint16[513]`
tokens, `uint8[513]` stored masks, prompt and PAD exclusion, response and EOS
supervision, split isolation, and trainer alignment through `stored_mask[1:]`.

Injected pre-publication failure removes the staging directory and leaves no
destination. The layer has no source discovery, no production plan execution,
no schedule builder, no training configuration, no checkpoint or optimizer
selection, and no training entry point. It changes no architecture, tokenizer
asset, checkpoint tensor key, production dataset, existing mask, schedule, or
training configuration.

## Validation

- Focused tests: `python -m pytest tests\test_vasu_140m_fixture_release.py -q`
  passed, 7 tests.
- Ruff: `python -m ruff check vasu\data\vasu_140m_fixture_release.py
  tests\test_vasu_140m_fixture_release.py` passed.
- Git diff check: `git diff --check` reported no whitespace errors; only
  pre-existing CRLF conversion warnings on unrelated modified docs.

## Non-Authorization Statement

No production data, logical manifest, token binary, mask binary, source
selection, schedule, training configuration, checkpoint, optimizer update,
authorization record, commit, or push was created by this review.
