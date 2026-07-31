# VASU-140M Base-Pretraining Source-Admission Audit — 2026-08-01

The accepted data-release plan requires source provenance, licensing,
contamination, deduplication, document split isolation, and full-loss 513-token
qualification, but it intentionally selects no source. The existing
`vasu.data.sources` registry already provides strict provenance metadata and
approval states, so a new parallel registry would create divergence risk.

This design therefore binds future VASU-140M candidates to the existing
registry and adds a source-specific review package before approval. No source
record was added or modified. No data path, release path, checkpoint, schedule,
or configuration was created.

Validation will use the source-registry tests and the VASU-140M record-contract
smoke. The design preserves the frozen tokenizer and 513-token contract and
rejects both legacy FineWeb binaries and the published instruction seed.

GPT-5.5 independently accepted the design on 2026-08-01 in
`VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_INDEPENDENT_REVIEW_DECISION_20260801.md`.
This decision does not authorize a source record, acquisition, release, or
training.
