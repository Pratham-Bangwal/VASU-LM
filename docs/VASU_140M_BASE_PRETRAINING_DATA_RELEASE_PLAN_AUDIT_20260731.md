# VASU-140M Base-Pretraining Data-Release Plan Audit — 2026-07-31

The repository contains a frozen VASU-140M 513-token record contract and a
published response-masked instruction release, but no VASU-140M base-pretraining
source or data release. Existing FineWeb manifest records are 256-token
training views and cannot satisfy the 512-token VASU-140M contract without a
new source-specific release and review.

This plan therefore selects no source and creates no data. It adds only the
required evidence sequence: source admission, contamination/deduplication,
source-level split isolation, full-loss 513-token packing, qualification,
atomic publication, and independent review.

## Validation note

The VASU-140M record-contract tests passed. Five historical instruction-release
plan tests correctly fail in the published repository because they still assert
that `data/processed/vasu_140m/instruction_seed_v1` must be absent. That path
was intentionally created by the one authorized publication. This is
post-publication test debt outside this source-agnostic planning task; the
tests must be redesigned to preserve no-overwrite safety while validating the
published state, not weakened or deleted here.

GPT-5.5 independently accepted the plan in
`VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN_INDEPENDENT_REVIEW_DECISION_20260731.md`.
