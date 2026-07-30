# VASU-140M Instruction Seed v1 Publication Audit

Date: 2026-07-30

Status: independently accepted; training remains unauthorized

## Authority

- human approver: Pratham Sharma;
- authorization ID:
  `vasu-140m-instruction-seed-v1-20260730-001`;
- authorization SHA-256:
  `9eaf4dce720b11a3389ef373556c899ca2559c3e0bbe95993c058ebf9bad09fd`;
- exact runtime commit:
  `5cf41881aed610c075f01d24efe57799a5dabb60`;
- authorization validity: 2026-07-30 through 2026-08-06;
- scope: one production construction;
- training authorized: false.

The detached envelope remains at
`C:\Users\acer\.vasu\authorizations\vasu_140m_instruction_seed_v1_20260730_001.json`.

## Publication result

The transactional publisher returned `release_complete=true`.

- qualification SHA-256:
  `60399e56ba30cc9904dfab9009194f424ea75d386f85598fa6b5a297815f7321`;
- external manifest SHA-256:
  `0eaec0131cb644e8eddf8ccedbcec3a42d7c1fa2754750b887aa1b9a443ce390`;
- external manifest file SHA-256:
  `b5a88ac62cd1a8ba2d449873673b935ec439e3e61b7bef10c0c6314aec67260a`;
- receipt SHA-256:
  `6a03032d5d3613f0bcdbff6219686e994a4993d3028a02f273fb242ac22bcdbf`;
- receipt file SHA-256:
  `bab6c8cce24cfb30f41f5e406151ff356680b0a1ea92b8f5f887f60fa8c88730`;
- internal manifest SHA-256:
  `c0ba23b5ebfcc1eafe632c08a28f4d924362b55acbfde5bc2826689c92220e04`;
- internal manifest file SHA-256:
  `80895494b16882327509863067424e1d564a2e817ed86d94bdaf63622e04aaa8`;
- logical records SHA-256:
  `95ec1692e229a8b1288d985cceff5dad4c8b1ac5df1e62975740b071721a140b`.

## Split evidence

| Split | Examples | Packed records | Used tokens | Supervised tokens |
|---|---:|---:|---:|---:|
| Train | 898 | 76 | 37,217 | 17,362 |
| Development | 48 | 4 | 1,944 | 825 |
| Evaluation | 50 | 5 | 2,020 | 937 |

Artifact hashes:

- train tokens:
  `f0a7f73847c234e3b63819dd752e0626c261fe3ef196d16a2b33bcb77e30d1f2`;
- train mask:
  `bafe1b658aedb96e2c5c698a59cd62d4ca78df725155b5ff7084b043180a40fe`;
- development tokens:
  `f648721b94b3b59dd60b2b777ca5ba6c8a0e2733449b29bfd88f1c149fa9610c`;
- development mask:
  `b55bcad2b097c5935b9cdd3ed23c37038d14b6ff5ce1a83a3f29e1337ee1b99a`;
- evaluation tokens:
  `bb4ebee650f2630f77959991b433bf5b7608113dc44508b9cc33d1189f16ee05`;
- evaluation mask:
  `56c9937d3d4741ece4cda79ae1aa30c2ed50b1381736d93f7bdb8bfa5270c6cf`.

## Validation

`validate_published_release` passed and `publication_status` returned
`complete`. This revalidated external/internal manifest hashes, receipt
binding, exact artifact sets, token/mask lengths, binary masks, record
boundaries, prompt exclusion, response/EOS supervision, and PAD tails.

The detached lock was removed after success. Existing-output checks now reject
qualification or a second construction before publication can be attempted,
and the consumed receipt permanently occupies the authorization ID.

## Artifact locations

- release:
  `data/processed/vasu_140m/instruction_seed_v1`;
- external manifest:
  `data/manifests/vasu_140m/instruction_seed_v1.json`;
- consumed receipt:
  `data/manifests/vasu_140m/authorization_receipts/vasu-140m-instruction-seed-v1-20260730-001.json`.

## Non-authorization

Publication created only the reviewed dataset release, manifest, and receipt.
It did not select a checkpoint, create a schedule or training configuration,
create optimizer state, or authorize training.

## Independent acceptance and packet erratum

GPT-5.5 independently accepted the completed publication on 2026-07-31. The
accepted decision is
`VASU_140M_INSTRUCTION_SEED_V1_PUBLICATION_INDEPENDENT_REVIEW_DECISION_20260731.md`.

The immutable review packet requested a `20260730` decision filename, while
the newer review instruction requested `20260731`. The reviewer correctly
followed the newer explicit path. This is a documentation-path erratum only;
no published artifact identity changed.
