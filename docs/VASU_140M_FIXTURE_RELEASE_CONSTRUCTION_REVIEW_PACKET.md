# VASU-140M Fixture Release Construction Review Packet

Status: independently accepted as fixture-only evidence; non-authorizing.

Date: 2026-07-30

## Decision requested

Accept or reject the fixture-only transactional construction layer as evidence
for a later production release-builder design. Acceptance must not be
interpreted as permission to construct the 996-example production release or
to train a model.

## Scope

`vasu/data/vasu_140m_fixture_release.py` exercises publication mechanics only:

- caller-supplied logical examples, with a hard maximum of 30;
- exact train/development/evaluation split isolation;
- the accepted 513-token `uint16` token and `uint8` shifted-mask contract;
- deterministic split-local packing and artifact hashes;
- an immutable manifest bound to the accepted plan and GPT-5.5 decision;
- non-overwriting, whole-directory atomic publication;
- cleanup after pre-publication failure;
- rejection of the planned production release and manifest paths;
- explicit `fixture_only=true`, `production_release_created=false`,
  `training_authorized=false`, and `training_permitted=false` evidence.

It has no source discovery, production plan execution, schedule generation,
training configuration, checkpoint selection, optimizer, or training entry
point.

## Materials

- `vasu/data/vasu_140m_fixture_release.py`
- `tests/test_vasu_140m_fixture_release.py`
- `vasu/data/vasu_140m_records.py`
- `configs/data/releases/vasu_140m_instruction_seed_v1.plan.json`
- [accepted plan decision](VASU_140M_INSTRUCTION_SEED_RELEASE_INDEPENDENT_REVIEW_DECISION.md)
- [completion audit](VASU_140M_FIXTURE_RELEASE_CONSTRUCTION_AUDIT_20260730.md)

## Review requirements

1. Confirm that no public function can target the planned production release.
2. Confirm that outputs are published only after every artifact and the
   hash-bound manifest are complete.
3. Confirm that an injected failure leaves neither the destination nor a
   staging directory.
4. Confirm that overwrite, wrong plan identity, wrong decision identity,
   oversized fixtures, split leakage, and artifact tampering fail closed.
5. Confirm that stored masks supervise response and EOS tokens only, exclude
   prompt and PAD tokens, and remain aligned through `stored_mask[1:]`.
6. Confirm that acceptance remains non-authorizing.

## Compatibility

This additive fixture layer changes no architecture, checkpoint tensor key,
tokenizer asset, production dataset, existing mask, schedule, or training
configuration. Existing checkpoints and datasets remain byte-compatible.

## Independent decision

GPT-5.5 independently accepted the exact fixture layer on 2026-07-30. See
[the decision record](VASU_140M_FIXTURE_RELEASE_CONSTRUCTION_INDEPENDENT_REVIEW_DECISION_20260730.md).
The decision does not authorize production release construction or training.
