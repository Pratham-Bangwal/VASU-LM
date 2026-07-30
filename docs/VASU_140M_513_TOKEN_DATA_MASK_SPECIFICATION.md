# VASU-140M 513-Token Data and Mask Specification

Status: frozen engineering specification; fixture-qualified; no production
release or training authorization.

Date: 2026-07-30

## Purpose and scope

`vasu_140m_v1` has a 512-token model context. Its fixed records therefore
contain 513 stored positions so the trainer can derive 512 inputs and their
next-token targets. Existing 257-token VASU-31M/60M releases remain valid for
their current workflows but are not silently repacked, extended, or relabeled
for VASU-140M.

The implementation in `vasu/data/vasu_140m_records.py` accepts only
caller-supplied logical examples. It has no default sources, production output
path, schedule builder, training configuration, or training entry point.

## Immutable identities

| Field | Frozen value |
| --- | --- |
| Record schema | `vasu.model-family-records.v1` |
| Model family | `vasu_140m_v1` |
| Model config SHA-256 | `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059` |
| Record specification SHA-256 | `1bbbeafe836104dc4326a3a00e0dfcef21a965bbc02f56408be4305ae16ce7e5` |
| Model context | 512 |
| Stored record width | 513 |
| Token dtype | `uint16` |
| Stored-mask dtype | `uint8` |
| Packing | `complete-example-sequential-input-order-v1` |
| Tokenizer SHA-256 | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` |
| Vocabulary | 32,000 |
| PAD / UNK / BOS / EOS | 0 / 1 / 2 / 3 |
| Required splits | `train`, `development`, `evaluation` |

Changing an identity-bearing field creates a new schema or specification
identity. It must never overwrite an artifact built under this contract.

## Logical-example contract

Each example has a non-empty stable ID, declared split, lowercase semantic
SHA-256, token IDs, stored mask, and target-start offset. The compiler requires
the tokenized prompt to remain an exact prefix of the tokenized prompt plus
response; a tokenizer boundary merge fails closed.

Every logical example:

- is complete and at most 513 tokens;
- contains no PAD token in real content;
- contains only IDs from the frozen 32,000-token vocabulary;
- ends in supervised EOS;
- has zero mask values from its first prompt token through the token before
  `target_start`;
- has one mask values from `target_start` through EOS;
- is never truncated, split between records, or carried between splits.

IDs and semantic hashes must be unique across all three splits. Every split is
packed independently. The input order is identity-bearing; no implicit shuffle
or randomness exists in the packer.

## Shifted-target semantics

For one stored record `tokens[0:513]` and `stored_mask[0:513]`, training derives:

```text
x         = tokens[:-1]
y         = tokens[1:]
loss_mask = stored_mask[1:]
```

The first stored mask value of every packed logical example is zero. For a
second example beginning at stored position `s`, `loss_mask[s - 1]` is
therefore zero: the synthetic transition from the prior example's EOS to the
new prompt's first token is never supervised. PAD tails contain token ID zero
and stored-mask zero. Response tokens and terminal EOS remain supervised.

## Deterministic fixture qualification

Run:

```powershell
python scripts/smoke_vasu_140m_record_spec.py
```

The smoke binds the actual tokenizer file and special IDs, compiles fixed
train/development/evaluation text fixtures, packs them twice, and requires
byte-identical logical, token, mask, and report hashes. The frozen report is
`evaluation/fixtures/vasu_140m_513_record_spec_v1.json`, with report SHA-256
`7329698aefc3ab4593b1bfba246487a83ed0210b4e3fbe96f72c8bb7b8c8e7e3`.

The report is self-describing: it records the family, model-config,
tokenizer, record-width, specification, and complete logical-input identities
in addition to per-split token and mask hashes. The qualification also derives
each 512-position training view independently from one record and proves that
no target array can join tokens from a following record. The reusable report
validator rejects unknown/missing fields, wrong family/config/tokenizer/width,
incomplete checks, malformed split metadata, authorization changes, and any
canonical report-hash mismatch. A separate frozen-fixture validator additionally
pins the canonical report identity above, so a modified report cannot be
accepted merely by recomputing its self-consistent hash.

This evidence proves record mechanics only. Fixture text is not production
data, and the smoke does not create a processed-data directory.

## Compatibility and non-authorization

- VASU-31M/60M checkpoints, record files, masks, and trainers are unchanged.
- VASU-140M model and optimizer states remain incompatible with earlier
  families because tensor shapes differ.
- The tokenizer identity and token IDs are unchanged.
- Exact-resume checkpoint semantics are unchanged; a future real-data runner
  must bind release and schedule identities separately.
- No raw source has been selected or approved by this specification.
- No production 513-token data, schedule, training config, authorization
  record, checkpoint, or optimizer update exists.

The generic contract's independent review is accepted. A separate immutable
source-specific production-release review remains mandatory before production
data construction. Training would still require a separate scientific plan,
preflight, and exact human authorization.
