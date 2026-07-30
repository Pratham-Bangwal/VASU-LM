# VASU-140M Instruction Seed v1 Publication Review Packet

Date: 2026-07-30

Requested reviewer: GPT-5.5 independent review

Decision requested: accept or reject the completed publication evidence

## Review scope

Review:

- `AGENTS.md`;
- `docs/PROJECT_STATUS.md`;
- `docs/VASU_140M_INSTRUCTION_SEED_V1_PUBLICATION_AUDIT_20260730.md`;
- the accepted authorization protocol, implementation, post-commit identity,
  and detached runtime-review evidence supplied in chat;
- `vasu/data/vasu_140m_production_release.py`;
- `vasu/data/vasu_140m_authorization_protocol.py`;
- `data/processed/vasu_140m/instruction_seed_v1`;
- `data/manifests/vasu_140m/instruction_seed_v1.json`;
- `data/manifests/vasu_140m/authorization_receipts/vasu-140m-instruction-seed-v1-20260730-001.json`;
- detached envelope:
  `C:\Users\acer\.vasu\authorizations\vasu_140m_instruction_seed_v1_20260730_001.json`.

## Required checks

1. Recompute the envelope self-hash and confirm its authority, exact runtime,
   qualification, paths, expiry, and `training_authorized=false`.
2. Run `validate_published_release` and confirm `publication_status=complete`.
3. Recompute every release, manifest, and receipt hash in the audit.
4. Confirm 996 decoded logical examples and exact 898/48/50 split isolation.
5. Confirm all token/mask layout and supervision invariants.
6. Confirm the receipt directly binds the v2 authorization and runtime.
7. Confirm the authorization is consumed and no publication lock remains.
8. Confirm no checkpoint, schedule, training configuration, optimizer state,
   or training authority was created.

## Required response

Return:

- decision: accept or reject;
- findings grouped as blocking, high, medium, and low;
- evidence examined;
- commands and results;
- exact artifact identities and counts;
- explicit non-training-authorization confirmation.

If accepted, create only:

`docs/VASU_140M_INSTRUCTION_SEED_V1_PUBLICATION_INDEPENDENT_REVIEW_DECISION_20260730.md`

Do not modify release artifacts, create another envelope, republish, select a
checkpoint, train, commit, or push.
