# VASU-140M Authorization Protocol v2 Implementation Review Packet

Date: 2026-07-30

Requested reviewer: GPT-5.5 independent review

Decision requested: accept or reject the implementation

## Scope

Review the new detached authorization gate, its adversarial tests, smoke
qualification, and frozen pre-commit evidence:

- `vasu/data/vasu_140m_authorization_protocol.py`;
- `tests/test_vasu_140m_authorization_protocol.py`;
- `scripts/smoke_vasu_140m_authorization_protocol.py`;
- `evaluation/fixtures/vasu_140m_authorization_protocol_v2_qualification_precommit.json`;
- `docs/VASU_140M_AUTHORIZATION_PROTOCOL_V2_IMPLEMENTATION_AUDIT_20260730.md`;
- the accepted protocol and independent decision;
- the unchanged production builder and its accepted implementation evidence.

## Identities

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

## Required review questions

1. Does the loader reject in-repository, linked, malformed, and non-canonical
   envelopes?
2. Are all v2 fields exact and unknown fields rejected?
3. Are runtime `HEAD`, clean state, anchor ancestry, builder identity, gate
   identity, qualification, assignment, paths, dates, scope, and training
   prohibition bound fail-closed?
4. Does the detached atomic lock prevent concurrent use, and is stale recovery
   restricted to a matching, old, inactive, self-hashed lock?
5. Does the persisted receipt directly bind the v2 authorization and gate
   SHA-256 values?
6. Do disk, mutation, open-handle, post-rename, staging, and reuse failures
   remain safe?
7. Does the separate module preserve the accepted production-builder SHA-256
   and frozen release identities?
8. Is any API capable of creating an authorization envelope or authorizing
   training?

## Required validation

Run:

```powershell
python -m pytest tests\test_vasu_140m_authorization_protocol.py tests\test_vasu_140m_production_release.py -q
python -m ruff check vasu\data\vasu_140m_authorization_protocol.py tests\test_vasu_140m_authorization_protocol.py scripts\smoke_vasu_140m_authorization_protocol.py
python scripts\smoke_vasu_140m_authorization_protocol.py | python -m json.tool > $null
git diff --check
git status --short
```

Confirm protected production paths remain absent.

## Required response

Return a decision, findings by severity, evidence examined, commands and
results, exact reproducibility of the frozen qualification, and explicit
confirmation that acceptance does not authorize an envelope, publication,
checkpoint selection, training configuration, or training.

If accepted, create only:

`docs/VASU_140M_AUTHORIZATION_PROTOCOL_V2_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260730.md`

Do not create an envelope, invoke publication, create protected artifacts,
train, commit, or push.
