# VASU-140M Base-Pretraining Data-Release Plan

Status: **specification-only; no source selected or data constructed.**

## Purpose

Define the release requirements for the first full-loss VASU-140M
base-pretraining corpus. It is intentionally source-agnostic: selecting a
source, downloading material, repacking the existing FineWeb binaries, or
creating a release all require later approval and review.

## Frozen compatibility boundary

| Item | Required value |
|---|---|
| Family | `vasu_140m_v1` |
| Model config SHA-256 | `29e9bafdffbbc632b1b6f006818b1470e6dbc20f21841aa33e625a0f04159059` |
| Tokenizer SHA-256 | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` |
| Stored record width | 513 (`uint16`) |
| Training view | `tokens[:-1]` inputs and `tokens[1:]` targets |
| Loss mask | full-loss text mask, with zero boundary/PAD targets |
| Required splits | train, development, evaluation |
| Output family | a new versioned `vasu_140m/base_pretraining/*` release only |

The published instruction seed v1 is response-masked instruction data and is
excluded. Existing FineWeb binaries use 257-token VASU-60M records and are not
approved, repacked, or relabeled by this plan.

## Source admission requirements

Every proposed source must have a versioned admission record before a byte is
processed. It must pin source owner, license and terms, immutable revision or
snapshot, acquisition method/date, raw-content hash, language/content scope,
known quality risks, permitted use, and deletion/retention constraints.

Potential sources are evaluated individually; this plan selects none. A source
is rejected if its provenance, license, reproducibility, or evaluation
independence cannot be established.

## Required release controls

1. **Document lineage:** retain stable source/document IDs and source hashes;
   generated chunks must identify their parent document and transformation.
2. **Normalization/filtering:** version the Unicode, boilerplate, language,
   quality, length, and safety filters; record rejection counts by reason.
3. **Contamination:** compare proposed text against every pinned VASU
   evaluation inventory at exact and word-ngram levels before split assignment.
   Quarantine any collision; do not weaken benchmark inventories.
4. **Deduplication:** run exact and near-duplicate checks within and across
   sources, then repeat across train/development/evaluation splits. Preserve
   reviewable candidate and disposition evidence.
5. **Split isolation:** deterministic source/document-level assignment prevents
   a parent document or semantic duplicate crossing splits.
6. **Tokenization/packing:** use the frozen tokenizer and independent
   513-token full-loss records. Validate dtype, range, PAD tails, EOS, shifted
   targets, boundaries, token/mask equality, and deterministic rebuild.
7. **Release atomicity:** qualify in a staging directory, write immutable
   manifests/hashes, publish atomically to a new output version, and prohibit
   overwrite.

## Required qualification evidence

Before publication, qualification must provide source and release manifests,
record/token/mask hashes, exact split counts, duplicate/contamination reports,
decoded round trips, complete serialized mask audits, deterministic rebuild
comparison, output-path absence, and a `training_authorized=false` result.
The builder must have its own implementation review and one-build publication
authorization; a passing release does not authorize training.

## Alternatives rejected

| Alternative | Reason |
|---|---|
| Reuse VASU-60M 257-token FineWeb binaries | Wrong record/context identity and missing 140M source-release review. |
| Use the instruction seed as pretraining text | Response-masked instruction lineage would confound base-pretraining scope. |
| Select external data informally | License, revision, source hash, and contamination boundaries would be unproven. |
| Source-agnostic admission plan first | Recommended; preserves choice and evidence gates. |

## Non-authorization

This plan does not authorize source discovery, acquisition, processing, data
release construction, schedule/configuration creation, checkpoint creation,
optimizer creation, base pretraining, instruction tuning, or training.
