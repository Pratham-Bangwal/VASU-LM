# VASU-140M Base-Pretraining Data-Release Plan Review Packet

Review whether this specification preserves VASU-140M data compatibility and
blocks unsafe source selection or release construction.

Read `AGENTS.md`, `docs/PROJECT_STATUS.md`,
`docs/VASU_140M_BASE_PRETRAINING_READINESS.md`,
`docs/VASU_140M_513_TOKEN_DATA_MASK_SPECIFICATION.md`,
`docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN.md`,
`docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN_AUDIT_20260731.md`,
`docs/DATASET.md`, and `data/processed/pretrain/fineweb_manifest.json`.

Confirm it selects no source, rejects automatic FineWeb/instruction-seed reuse,
requires source provenance and licensing, pins all contamination inventories,
requires cross-source/split deduplication and document-level split isolation,
and requires 513-token full-loss qualification with atomic publication.

Acceptance authorizes only this plan. It does not authorize source discovery,
acquisition, processing, data release construction, configuration, schedule,
optimizer, checkpoint, training, commit, or push.
