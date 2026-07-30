# VASU-140M Instruction Seed v1 Publication Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-07-31

Reviewer: GPT-5.5 independent review

## Decision

Accept the completed VASU-140M instruction-seed v1 production publication.

This acceptance recognizes that one authorized, hash-bound publication
completed and was consumed by its receipt. It does not authorize checkpoint
selection, schedule creation, training configuration, optimizer creation, or
training.

## Findings With Severity

- Blocking: none.
- High: none.
- Medium: none.
- Low: `docs/VASU_140M_INSTRUCTION_SEED_V1_PUBLICATION_REVIEW_PACKET.md`
  requests a `20260730` decision document, while the review request explicitly
  requests this `20260731` decision path. This document follows the newer
  explicit review instruction.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_INSTRUCTION_SEED_V1_PUBLICATION_AUDIT_20260730.md`
- `docs/VASU_140M_INSTRUCTION_SEED_V1_PUBLICATION_REVIEW_PACKET.md`
- `docs/VASU_140M_PRODUCTION_RELEASE_AUTHORIZATION_PROTOCOL_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `docs/VASU_140M_AUTHORIZATION_PROTOCOL_V2_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `docs/VASU_140M_AUTHORIZATION_PROTOCOL_V2_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260730.md`
- Detached runtime eligibility decision supplied in chat
- `vasu/data/vasu_140m_production_release.py`
- `vasu/data/vasu_140m_authorization_protocol.py`
- `data/processed/vasu_140m/instruction_seed_v1`
- `data/manifests/vasu_140m/instruction_seed_v1.json`
- `data/manifests/vasu_140m/authorization_receipts/vasu-140m-instruction-seed-v1-20260730-001.json`
- `C:\Users\acer\.vasu\authorizations\vasu_140m_instruction_seed_v1_20260730_001.json`

## Commands And Results

- Published-release validator:
  `publication_status= complete`,
  `manifest_sha256= 0eaec0131cb644e8eddf8ccedbcec3a42d7c1fa2754750b887aa1b9a443ce390`,
  `training_authorized= False`.
- `python -m pytest tests\test_vasu_140m_authorization_protocol.py -q`:
  passed, `15 passed`.
- `python -m pytest tests\test_vasu_140m_production_release.py -q -k "not real_source_qualification_matches_frozen_review_identity"`:
  passed, `14 passed, 1 deselected`.
- `git diff --check`:
  passed with no whitespace errors; Git reported only CRLF conversion warnings
  for existing modified docs.
- `git status --short`:
  inspected. Existing modified docs and untracked publication evidence are
  present.
- Canonical self-hash checks:
  external manifest, internal manifest, receipt, and detached envelope all
  matched their reported canonical hashes.

## Release Identities

- authorization ID:
  `vasu-140m-instruction-seed-v1-20260730-001`
- authorization SHA-256:
  `9eaf4dce720b11a3389ef373556c899ca2559c3e0bbe95993c058ebf9bad09fd`
- runtime commit:
  `5cf41881aed610c075f01d24efe57799a5dabb60`
- qualification SHA-256:
  `60399e56ba30cc9904dfab9009194f424ea75d386f85598fa6b5a297815f7321`
- external manifest SHA-256:
  `0eaec0131cb644e8eddf8ccedbcec3a42d7c1fa2754750b887aa1b9a443ce390`
- external manifest file SHA-256:
  `b5a88ac62cd1a8ba2d449873673b935ec439e3e61b7bef10c0c6314aec67260a`
- receipt SHA-256:
  `6a03032d5d3613f0bcdbff6219686e994a4993d3028a02f273fb242ac22bcdbf`
- receipt file SHA-256:
  `bab6c8cce24cfb30f41f5e406151ff356680b0a1ea92b8f5f887f60fa8c88730`
- internal manifest SHA-256:
  `c0ba23b5ebfcc1eafe632c08a28f4d924362b55acbfde5bc2826689c92220e04`
- internal manifest file SHA-256:
  `80895494b16882327509863067424e1d564a2e817ed86d94bdaf63622e04aaa8`
- logical records SHA-256:
  `95ec1692e229a8b1288d985cceff5dad4c8b1ac5df1e62975740b071721a140b`
- production builder SHA-256:
  `0efb0189b4b0c5a8621da9053d56db57d95ebf5b0f1bb59753952ec8afa5d1e2`
- authorization protocol SHA-256:
  `d135889b89b424e7a3253adddec0b1cfd09b49efdb6831f048934f2468d9acac`
- assignment SHA-256:
  `59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394`

## Artifact Identities And Counts

- train: 898 examples, 76 packed records, 37,217 used tokens,
  17,362 supervised tokens.
- development: 48 examples, 4 packed records, 1,944 used tokens,
  825 supervised tokens.
- evaluation: 50 examples, 5 packed records, 2,020 used tokens,
  937 supervised tokens.
- decoded logical examples: 996.

Artifact hashes:

- `train.tokens.bin`:
  `f0a7f73847c234e3b63819dd752e0626c261fe3ef196d16a2b33bcb77e30d1f2`
- `train.mask.bin`:
  `bafe1b658aedb96e2c5c698a59cd62d4ca78df725155b5ff7084b043180a40fe`
- `development.tokens.bin`:
  `f648721b94b3b59dd60b2b777ca5ba6c8a0e2733449b29bfd88f1c149fa9610c`
- `development.mask.bin`:
  `b55bcad2b097c5935b9cdd3ed23c37038d14b6ff5ce1a83a3f29e1337ee1b99a`
- `evaluation.tokens.bin`:
  `bb4ebee650f2630f77959991b433bf5b7608113dc44508b9cc33d1189f16ee05`
- `evaluation.mask.bin`:
  `56c9937d3d4741ece4cda79ae1aa30c2ed50b1381736d93f7bdb8bfa5270c6cf`

## Integrity Conclusions

- One authorized publication completed.
- Release status is `complete`.
- The consumed receipt directly binds the v2 authorization SHA-256,
  authorization protocol SHA-256, runtime commit, qualification SHA-256,
  external manifest SHA-256, and `training_authorized=false`.
- The detached envelope self-hash is valid and the authorization is consumed
  by the existing receipt for the same authorization ID.
- No publication lock remains beside the detached envelope.
- `validate_published_release` revalidated token/mask sizes, binary masks,
  record widths, prompt exclusion, response/EOS supervision, record
  boundaries, and PAD tails.
- No VASU-140M checkpoint path, instruction-seed training configuration,
  instruction-seed schedule path, or optimizer-state path was found at the
  checked publication-specific locations.
- `training_authorized` remains false in the envelope, external manifest,
  internal manifest, receipt, and accepted plan.

## Non-Authorization Confirmation

This decision does not authorize checkpoint selection, schedule creation,
training configuration, optimizer creation, or training. This review did not
modify release artifacts, create another envelope, republish, train, commit,
or push.
