# VASU-140M Base Checkpoint Selection Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-07-31

Reviewer: GPT-5.5 independent review

## Decision

Accept the VASU-140M base-checkpoint selection decision.

The repository supports the fail-closed conclusion that no compatible
`vasu_140m_v1` base checkpoint exists. No checkpoint is selected. VASU-60M
Candidate A must not be selected, converted, partially loaded, or treated as a
VASU-140M parent.

## Findings With Severity

- Blocking: none.
- High: none.
- Medium: none.
- Low: the working tree contains existing modified planning documents and the
  untracked base-checkpoint selection packet/audit/plan. This does not affect
  the review conclusion because the required validation commands pass and the
  checkpoint inventory contains no VASU-140M artifact.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/CHECKPOINTS.md`
- `docs/ROADMAP.md`
- `docs/VASU_140M_IMPLEMENTATION_READINESS.md`
- `docs/VASU_140M_BASE_CHECKPOINT_SELECTION_AND_TRAINING_PLAN.md`
- `docs/VASU_140M_BASE_CHECKPOINT_SELECTION_AUDIT_20260731.md`
- `docs/VASU_140M_BASE_CHECKPOINT_SELECTION_REVIEW_PACKET.md`
- `vasu/model/families.py`
- `vasu/model/checkpoint_identity.py`
- `scripts/preflight_vasu_140m.py`
- `tests/test_model_families.py`
- `tests/test_model_family_checkpoint_identity.py`
- Recursive checkpoint inventory under `checkpoints/`
- Published VASU-140M instruction release hashes under
  `data/processed/vasu_140m/instruction_seed_v1` and
  `data/manifests/vasu_140m/instruction_seed_v1.json`

## Command Results

- `python scripts\preflight_vasu_140m.py`: passed. The report identified
  `vasu_140m_v1`, config fingerprint
  `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059`,
  family fingerprint
  `72f5a98d7d3f307c8bebf4fd6c21b1b8642262a37f59070d436c0de54a3af99b`,
  parameter count `137841408`, `passed=true`, and
  `training_authorized=false`.
- `python -m pytest tests\test_model_families.py tests\test_model_family_checkpoint_identity.py -q`:
  passed, `28 passed`.
- `python -m ruff check vasu\config.py vasu\model\families.py vasu\model\checkpoint_identity.py scripts\preflight_vasu_140m.py`:
  passed, `All checks passed!`.
- `git diff --check`: passed with no whitespace errors; Git reported only CRLF
  conversion warnings for existing modified docs.
- `git status --short`: inspected. Existing modified docs and the
  base-checkpoint selection packet/audit/plan are present.

## Checkpoint Inventory Conclusion

The recursive checkpoint inventory contains legacy checkpoints, VASU-31M
checkpoints, and many `checkpoints/vasu_60m/...` artifacts, including Candidate
A, Candidate C, Candidate D, FineWeb milestones, instruction checkpoints, and
factual-CPT checkpoints. No checkpoint directory or `.pt` file identified as
VASU-140M was present, and `checkpoints/vasu_140m` does not exist.

Candidate A is explicitly wrong-family:

- Candidate A family: `vasu_60m_v1`
- Candidate A parameter count: `58,337,792`
- Required family: `vasu_140m_v1`
- Required parameter count: `137,841,408`

A direct family-identity check rejects a VASU-60M identity when validated as
`vasu_140m_v1` with a family mismatch. The tests also prove wrong destination
configuration is rejected before state loading and matching loads are strict.

## Published Release Unchanged

The published VASU-140M instruction release remains unchanged. Verified
published identities include:

- external manifest file SHA-256:
  `b5a88ac62cd1a8ba2d449873673b935ec439e3e61b7bef10c0c6314aec67260a`
- logical records SHA-256:
  `95ec1692e229a8b1288d985cceff5dad4c8b1ac5df1e62975740b071721a140b`
- train tokens:
  `f0a7f73847c234e3b63819dd752e0626c261fe3ef196d16a2b33bcb77e30d1f2`
- train mask:
  `bafe1b658aedb96e2c5c698a59cd62d4ca78df725155b5ff7084b043180a40fe`
- development tokens:
  `f648721b94b3b59dd60b2b777ca5ba6c8a0e2733449b29bfd88f1c149fa9610c`
- development mask:
  `b55bcad2b097c5935b9cdd3ed23c37038d14b6ff5ce1a83a3f29e1337ee1b99a`
- evaluation tokens:
  `bb4ebee650f2630f77959991b433bf5b7608113dc44508b9cc33d1189f16ee05`
- evaluation mask:
  `56c9937d3d4741ece4cda79ae1aa30c2ed50b1381736d93f7bdb8bfa5270c6cf`

No publication artifact was modified by this review.

## Non-Authorization Confirmation

This review does not authorize checkpoint creation, base pretraining,
instruction tuning, training configuration creation, schedule creation,
optimizer creation, authorization records, training, commits, or pushes. No
checkpoint, training config, schedule, optimizer state, authorization record,
or training run was created by this review.
