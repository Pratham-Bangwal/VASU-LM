# VASU-140M Source-Admission Schema Implementation Audit — 2026-08-01

## Outcome

The accepted source-admission design now has a strict, metadata-only v1
validator. It validates an in-memory or JSON review package and creates no
source record, data, release, checkpoint, configuration, or authorization.

## Contract

The validator freezes `vasu_140m_v1`, its model-configuration identity, and the
unchanged tokenizer identity. Exact field sets cover the source-registry
binding, primary legal evidence, immutable acquisition recipe, document
lineage, versioned quality policy, pinned evaluation inventories, cross-source
deduplication, decision state, and canonical package SHA-256.

Admission and acquisition are separate: every package must keep
`acquisition.authorized=false` and `training_authorized=false`. Approval
requires reviewer identity, a timezone-aware review time, and zero unresolved
legal items. A repository-bound validation mode verifies the source-registry
file and every evaluation inventory against their declared paths and SHA-256,
then confirms the source ID and approval state against the generic registry.
Evaluation isolation and deduplication must both precede split assignment.

## Validation

- Focused source-admission and registry tests pass.
- Tests reject unknown fields, family/config/tokenizer changes, unsafe paths,
  missing evaluation inventories, post-split scans, authorization flags,
  unresolved approval evidence, timestamps without timezones, mismatched bound
  files, and modified-but-unhashed packages.
- No file under `configs/data/sources`, protected base-data paths, or
  `checkpoints/vasu_140m` was created or modified.

## Compatibility

The implementation is additive and does not change the generic registry
schema. Existing registry records, mixtures, datasets, masks, tokenizer,
checkpoints, and exact-resume behavior remain compatible.

GPT-5.5 independently accepted the implementation on 2026-08-01 in
`VASU_140M_SOURCE_ADMISSION_SCHEMA_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260801.md`.
The decision remains non-authorizing.
