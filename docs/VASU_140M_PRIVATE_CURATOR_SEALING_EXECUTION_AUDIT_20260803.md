# VASU-140M Private Curator Sealing Execution Audit

Date: 2026-08-03

## Scope

This audit records the one-shot conversion of the independently supplied private
curator intake into five sealed held-out candidate inventories. The execution
used the accepted implementation at repository commit
`08d34598dd103686be50b6121b80f88b63d05e64` in a clean detached worktree.

The run did not open the private key, decrypt a held-out payload, freeze a
production evaluation suite, authorize evaluation, or authorize training.

## Bound intake

- Intake schema: `vasu_140m_private_curator_intake_v1`
- Intake report SHA-256:
  `a568cc5627cd84a1d6989ae832cdd017edd6fb94011b3874f0ef2a5ed043928d`
- Intake receipt file SHA-256:
  `9c77388dcff102e675f048fd294e04df7fa48520b81d98ddb06d9113ab7f5bf1`
- Age recipient fingerprint SHA-256:
  `3413c58d2dae91a9c253642aba8583923af3fcd5d02893ad940ca6f21b5bd2da`
- Development overlap count: `0`
- Total unique records: `1500`

## Sealed candidate identities

| Dimension | Records | Inventory SHA-256 | Ciphertext SHA-256 |
|---|---:|---|---|
| Arithmetic | 1000 | `ccc2afe2dbf4414a523c4c3715b9e8defbdf146f9c39b0a98fcd98cc2279274f` | `dd26b6b07e0b0f1199d4f508c9e81f3958fcca891752c1e277efccb75ddeff8a` |
| Factuality | 200 | `3d1b43edc6c01f7c9de4f67b2b65b1db3a8a74658d43f9a49d7a3691d62982e1` | `693134d7b4628e63c2415d3f28f93597ea0ec5a53a9fba3ac08071e6c96487cc` |
| Manual review | 60 | `e364c07f9ef0e3356bdb27f61cc761b95213a277e9598754ad278c792e9cf4b6` | `075113c520916dfdff0eeab1b9a6cb7b78cc81a1482bfbe071727084235a9daf` |
| Repetition | 120 | `68a986f226e63f7c1015f3dcfebaf5a0444f4fb4d2743bd14b78fa2517706a0b` | `3b730912f87da0fc8f7651dd73d35c0161ee1a54c1b423ec0ab1a5c5076de571` |
| Robustness | 120 | `09bfe0eb1cac11561ec82a44b45b113dafc55aff9bcce923a9b9c7824e922783` | `cc407cc22135a876ee06cff569a6fa14200f85d747b9dbab621a5e80096b7e63` |

Detached sealing receipt:

- Embedded receipt identity:
  `c14fe696143e618ffdf4f501f3ba90ab8fe4f48fab93e96bfe99cd3e2ce52d18`
- Receipt file SHA-256:
  `e8ae88b9368b786882d9c9d87cd55929a58614f080fceab2f63b050e941e6d81`

## Validation

- All five manifests passed `validate_inventory_manifest_files` without held-out
  opening.
- Every payload has an Age v1 ciphertext header and matches its bound SHA-256
  and byte count.
- All provenance and contamination records match their manifest commitments.
- No `payload.jsonl` plaintext file exists in the candidate tree.
- The output contains exactly five encrypted payloads, five manifests, ten
  public indexes, and one detached receipt.
- `private_key_opened=false`.
- `production_suite_frozen=false`.
- `evaluation_run_authorized=false`.
- `training_authorized=false`.

## Compatibility

This evidence is additive. It changes no model architecture, tokenizer,
dataset, mask, checkpoint, optimizer or scheduler state, training configuration,
or existing evaluation result. Existing checkpoints and datasets remain
compatible.

## Remaining gates

These artifacts are candidate held-out inventories, not a frozen production
suite. Production development inventories, semantic review, source admission,
suite-freeze review, a compatible VASU-140M checkpoint, and a separately
authorized evaluation run remain required. Training remains unauthorized.
