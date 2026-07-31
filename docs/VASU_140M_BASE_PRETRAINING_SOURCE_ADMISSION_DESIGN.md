# VASU-140M Base-Pretraining Source-Admission Design

Status: **design-only; no source is selected, acquired, or approved.**

## Purpose

Define the admission boundary before any external source can enter a new
VASU-140M base-pretraining release. This design extends the repository's
metadata-only source registry; it does not create a parallel registry or
authorize construction.

## Immutable boundaries

| Boundary | Requirement |
| --- | --- |
| Model family | `vasu_140m_v1` |
| Tokenizer SHA-256 | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` |
| Record contract | New full-loss `uint16[513]` records and `uint8[513]` masks only |
| Existing 257-token FineWeb binaries | Not admissible as a VASU-140M release input |
| Published instruction seed v1 | Excluded; response-masked instruction lineage is not base text |
| Authorization | `training_authorized=false`; admission never authorizes acquisition, release, or training |

## Admission record

Each candidate must bind one exact record from the generic source registry.
The generic registry status and the VASU-140M admission decision are separate
facts: a generically `approved` source can remain `pending`, `blocked`, or
`rejected` for this model and release. The package must bind provider, dataset
card and homepage URLs, immutable revision, subset/split, access method,
license URL/name, commercial/attribution/redistribution/access policy,
expected size/counts, local raw and processed path plans, intended domain,
and known quality/contamination risks.

In addition, the candidate review package must contain:

1. primary license and terms evidence, including attribution, ShareAlike,
   deletion, personal-data, and redistribution obligations;
2. a reproducible acquisition recipe with exact revision/snapshot identifier,
   timestamp-independent endpoints where possible, expected shard inventory,
   and raw-byte hashing method;
3. a document-lineage proposal: stable provider/document IDs, source revision,
   shard/location metadata, and a deterministic transformation identifier;
4. a quality/filtering proposal with versioned normalization, language,
   boilerplate, safety, length, and malformed-text dispositions;
5. an evaluation-isolation plan covering every pinned VASU evaluation inventory
   at exact and word-ngram levels before split assignment; and
6. a cross-source duplicate plan using the compatible FineWeb document indexes
   where applicable, plus exact and near-duplicate evidence within the source.

Unknown policy values, mutable revisions, missing license evidence, or absent
document lineage block admission. A source with an incompatible license or
unresolvable evaluation-contamination risk is rejected rather than deferred.

## State transitions

`pending` → `approved` is permitted only after a human reviewer records a
timezone-aware decision, the bound generic registry record is already
`approved`, and the package resolves every required item. Generic approval
alone never implies VASU-140M approval. VASU-140M approval means eligible only
for a separately authorized bounded acquisition/preparation proposal. It does
not mean bytes may be downloaded or used in a release.

`pending` → `blocked` records a remediable access, revision, policy, or
evidence gap. `pending`/`blocked` → `rejected` records an incompatible license,
unrecoverable provenance problem, or unacceptable contamination/quality risk.
No rejected record can be recycled under a new ID without a materially new
source identity and a fresh review.

## Required independent evidence before approval

The prospective admission audit must show registry-schema validation, no
existing VASU-140M base release output, no selected checkpoint/configuration,
and `training_authorized=false`. It must explicitly show why the candidate is
not the legacy FineWeb binary or instruction-seed path. It must not claim raw
content, hashes, filtered counts, or deduplication results before acquisition.

## Alternatives

| Alternative | Decision |
| --- | --- |
| Reuse the generic source registry | Recommended; preserves one provenance authority. |
| Treat generic approval as VASU-140M approval | Rejected; model/release-specific gates remain separate. |
| Make an unreviewed source record `approved` | Rejected; approval must be evidence-backed. |
| Download to discover whether a source is admissible | Rejected; admission precedes acquisition. |
| Reuse legacy binaries as a shortcut | Rejected; wrong VASU-140M record lineage. |

## Non-authorization

This design does not authorize source discovery, registry mutation, acquisition,
processing, release construction, configuration, scheduling, checkpoint
creation, optimizer creation, base pretraining, instruction tuning, or training.
