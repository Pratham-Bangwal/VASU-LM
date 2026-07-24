# VASU Data Source Registry

## Capability-CPT scheduled sources

The additive capability-CPT path resolves three hash-bound training sources:
the post-step-200k FineWeb extension region, the approved Wikimedia factual
training split, and the canonical `verified_arithmetic_v1` packed training
records. FineWeb/Wikimedia validation and arithmetic development/evaluation
artifacts are references only and are rejected as schedule inputs. The legacy
physical FineWeb/Wikimedia 85/15 artifact is unchanged.

## FineWeb document recovery and index readiness

The versioned SQLite index supports normalized SHA-256 exact
lookup and deterministic word-5-gram MinHash/LSH candidates using normalization
version `vasu_cross_source_nfc_casefold_ws_v1`. The original
`data/raw/pretrain/fineweb_1m.jsonl` is document-level recoverable, although it
lacks original IDs and URLs; its stable local reference is its line number and
its provenance is explicitly incomplete. The completed original index contains
999,992 unique documents from 1,000,000 input lines and collapses eight
duplicate normalized hashes. SQLite integrity, manifest hash, deterministic
exact lookups, and normalization-version rejection passed.

The extension is not document-level recoverable from its local token binary
alone. Its dedup database contained reusable historical exact hashes and
extension source IDs, which enabled the 2026-07-17 production recovery against
the official Dataset Viewer pinned revision. All 379,247 retained source IDs
were recovered and reproduced their historical `SHA-256(text.strip())` values.
The recovered gzip JSONL artifact was then indexed with the same normalization
version as the original source. Combined coverage now includes both the
original and extension FineWeb indexes. Token binaries are still never treated
as document indexes.

## Purpose

The source registry is the provenance and approval layer for datasets proposed
in VASU data-mixture manifests. It records exactly which provider, dataset,
subset, split, revision, license, access route, local path plan, and quality
work applies to each mixture source ID.

This layer is metadata-only. Loading or validating the registry does not
download, tokenize, inspect, or train on dataset content.

## Approval states

The registry supports exactly four states:

| State | Meaning |
| --- | --- |
| `pending` | Metadata exists, but review or preparation planning is incomplete. |
| `approved` | The source is eligible for acquisition and preparation under the recorded conditions. |
| `rejected` | The source must not be used. |
| `blocked` | The source may become usable, but an access, license, revision, or risk issue prevents approval. |

An `approved` record requires known policy values, a reviewer, a timezone-aware
review timestamp, resolved approval notes, and both raw and processed local path
plans. Unknown boolean policy values are represented by JSON `null` and are
allowed only for `pending` or `blocked` sources.

Approval is not permission to bypass license obligations, attribution,
redistribution restrictions, quality filtering, deduplication, or contamination
review.

## Current registry

Registry files live under `configs/data/sources/`.

| Source ID | Dataset/subset | Status | Purpose |
| --- | --- | --- | --- |
| `fineweb_edu_original_train` | FineWeb-Edu `CC-MAIN-2013-20`, fixed local training region | approved | Existing control/general/educational source |
| `fineweb_edu_extension_2025_26` | FineWeb-Edu `CC-MAIN-2025-26`, validated extension | approved | Existing educational continuation source |
| `wikipedia_en_20231101_planned` | English Wikipedia `20231101.en` at `e6057dc557255a03c9c3c47ceab0eb44353b1bc5` | approved | Pinned factual source; bounded default preparation validated, training not started |
| `finemath_4plus_planned` | FineMath 4+ | blocked | Planned mathematics pilot; immutable revision and risk review required |
| `permissive_python_code_planned` | The Stack v2 permissive Python allowlist | blocked | Planned code pilot; access, per-file licensing, and redistribution review required |
| `vasu_verified_reasoning_v1_planned` | Locally generated verified reasoning v1 | pending | Planned reasoning pilot; generator and validation design incomplete |

The control and factual pilots pass the registry approval gate. The capability
pilot remains planning-valid but not training-ready because its mathematics,
code, and reasoning records are blocked or pending. Registry readiness means
metadata and preparation eligibility only; it does not mean that a physical
dataset exists.

## Approved Wikimedia factual source

The factual pilot selects the official `wikimedia/wikipedia` Hugging Face
distribution at immutable commit
`e6057dc557255a03c9c3c47ceab0eb44353b1bc5`, configuration `20231101.en`,
split `train`. The pinned card records 41 Parquet shards, 6,407,814 examples,
11,630,929,031 compressed download bytes, and 20,200,062,385 decoded bytes.
The provider does not publish a VASU-tokenizer count; 5 billion raw tokens is
therefore recorded only as a planning estimate and must be replaced with a
measured count during preparation.

This source was selected over:

- a Wikimedia XML dump, because old dated dump directories are not guaranteed
  to remain available, the XML/wikitext extraction is substantially heavier,
  and a full dump is less practical for limited local storage;
- the Wikimedia Enterprise Snapshot API, because it requires an account and
  its current snapshot identifiers are updated over time unless a returned
  version is separately captured;
- the newer Structured Contents beta, because its schema is evolving and is
  unnecessary for the bounded plain-text pilot.

The pinned Parquet distribution is practical on Windows because it is already
sharded and can be acquired and processed sequentially with resumable cache
state. The full 11.63 GB source need not coexist with all interim data: a future
preparer should process one verified shard at a time and stop after producing
the approved 50M-token pilot allocation.

### License and reuse obligations

The pinned dataset card declares `CC-BY-SA-3.0` and `GFDL`. Commercial reuse
and redistribution are permitted only while complying with the applicable
license. Preparation and any redistribution must retain article/source
provenance, provide author attribution through article/history URLs or an
equivalent compliant mechanism, link the license, identify modifications, and
honor ShareAlike for adaptations. Imported text and fair-use material can carry
additional restrictions, so visible source notices must be preserved and
exception material excluded or separately reviewed.

Primary evidence:

- [pinned dataset commit and statistics](https://huggingface.co/datasets/wikimedia/wikipedia/commit/e6057dc557255a03c9c3c47ceab0eb44353b1bc5)
- [CC BY-SA 3.0 official deed](https://creativecommons.org/licenses/by-sa/3.0/)
- [GNU Free Documentation License 1.3](https://www.gnu.org/licenses/fdl-1.3.html)
- [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
- [Wikimedia dump license and exception notice](https://dumps.wikimedia.org/legal.html)

### Quality and contamination plan

The future preparer must:

1. accept only `20231101.en` / `train` records and verify article/main
   namespace scope;
2. remove redirects and disambiguation pages when reliably identifiable;
3. clean markup, citation remnants, empty text, and boilerplate-heavy records;
4. normalize Unicode to NFC and reject undecodable records;
5. require at least 200 normalized characters and split records over 100,000
   characters at paragraph boundaries;
6. retain article ID, title, URL, source revision, shard, and attribution data;
7. remove exact duplicates using normalized-text hashes;
8. perform deterministic near-duplicate detection using documented
   MinHash/LSH parameters;
9. deduplicate against both existing FineWeb continuation sources;
10. scan normalized text and n-grams against every repository evaluation
    prompt before tokenization; and
11. record counts and reasons for every filtered or deduplicated record.

Windows-safe repository-relative path plan:

```text
data/raw/factual/wikimedia/20231101_en_e6057dc557255a03c9c3c47ceab0eb44353b1bc5/
data/interim/factual/wikimedia/20231101_en_e6057dc/
data/processed/pretrain/factual/wikimedia_20231101_en_e6057dc.bin
data/manifests/factual/wikimedia_20231101_en_e6057dc.json
```

No directory or data artifact above was created by the metadata review.

## File formats

A registry JSON file can contain either:

1. One source-record object; or
2. A source-family bundle with exactly one top-level `records` list.

The FineWeb file uses a bundle because its original and extension artifacts
have distinct source IDs, crawl configurations, revisions, local paths, and
content hashes. All fields are strict: missing and unknown fields are rejected.

## Python API

```python
from pathlib import Path

from vasu.data.mixtures import load_manifest
from vasu.data.sources import (
    get_source,
    load_source_record,
    load_source_registry,
    validate_manifest_sources,
)

registry = load_source_registry(Path("configs/data/sources"))
record = get_source(registry, "fineweb_edu_original_train")

manifest = load_manifest(Path("configs/data/vasu_60m_control_pilot.json"))

# Check IDs and provenance consistency while allowing planning records.
validate_manifest_sources(manifest, registry)

# Required readiness gate before acquisition, preparation, or training.
validate_manifest_sources(manifest, registry, require_approved=True)
```

`load_source_record()` intentionally loads only single-record files. Source
family bundles are loaded through `load_source_registry()`.

## Cross-manifest checks

For every mixture source, registry validation checks:

- exact source ID presence;
- dataset-card URL;
- pinned revision;
- license name;
- split;
- planned processed path;
- supported domain;
- commercial-use and attribution policy when known;
- content hash consistency when the mixture supplies a hash;
- approval state when `require_approved=True`.

This prevents a mixture from silently changing the dataset revision, license,
path, policy, or domain declared during source review.

## Filesystem behavior

The registry validates path plans as metadata. It does not require planned raw
or processed files to exist and does not resolve paths relative to a hidden
working directory. Physical file existence, size, hash verification, and
preparation outputs belong to a later explicitly authorized acquisition or
preparation workflow.

## Promotion workflow

Before changing a record to `approved`:

1. Pin an immutable dataset revision and intended subset/split.
2. Verify the applicable license and source-level exceptions.
3. Resolve commercial-use, attribution, redistribution, gating, and
   authentication policies.
4. Define raw and processed local paths.
5. Specify quality, deduplication, and contamination controls.
6. Record expected sizes and counts where available.
7. Review access and redistribution requirements.
8. Record the reviewer and timezone-aware review timestamp.
9. Remove unresolved markers from approval notes.
10. Add a SHA-256 content hash once a concrete prepared artifact exists.

No pending, blocked, or rejected source should pass the training-readiness gate.

## Wikimedia bounded preparation

The approved `wikipedia_en_20231101_planned` record is consumed by
`scripts/prepare_wikimedia_pilot.py`. The preparation config pins the exact
dataset revision, subset, split, and a single shard; dry-run validates this
registry and the factual mixture readiness gate before any acquisition.

The default pilot cannot exceed 10,000 source rows, 2,000 accepted parent
documents, 4,000 accepted chunks, 2,000,000 VASU tokens, one shard, or 1 GB
downloaded. Parent and chunk counters are explicit and independent; the legacy
`max_accepted_documents` key is interpreted only as a deprecated chunk-limit
alias. The pipeline produces a reviewable
document-level JSONL plus progress, preparation manifest, machine-readable
summary, and text summary. Resume verifies the configuration/source identity
and reconciles output to the last atomically committed byte offset.

FineWeb cross-source deduplication is available through the combined
original-plus-extension coverage manifest and compatible document indexes. The
Wikimedia broad-review artifact was compared against both indexes with zero
overlap candidates. The existing token binaries alone remain deliberately
insufficient for this purpose; the gate relies on recovered document text,
normalized exact hashes, compatible word-5-gram signatures, and provenance
metadata.

The pinned 420,296,449-byte shard is cached and verified against its
acquisition metadata and SHA-256. The corrected v2 smoke inspected 2 rows,
retained 20 distinct chunks and 17,237 tokens, and observed a maximum of 1,022
tokens per chunk. It recorded no encoding repairs or quality rejections and
passed output validation.

The apparent smoke mojibake came from displaying clean UTF-8 JSONL through an
incompatible Windows decoder. The preparer now also detects and conservatively
repairs only reversible source corruption, rejects replacement/control or
low-confidence cases, and preserves correct Unicode. Inline references,
citations, comments, and templates are removed with spaces before canonical
whitespace cleanup. Chunk-level hashes, deduplication, contamination checks,
token accounting, and parent/source-row provenance use the versioned
`wikimedia_pilot_document_v2` schema.

The earlier default artifact stopped at the historical 2,000-chunk limit with
1,036,527 tokens. It remains valid but is limit-bound. The refactored default
pilot completed at `token_limit` with 279 inspected rows, 262 accepted parent
documents, 3,751 accepted chunks, and 1,999,974 tokens. The maximum chunk was
1,024 tokens, no replacement characters were present, one high-confidence
FineWeb near overlap was rejected, and v3 output validation passed. No
Wikimedia model training has started.

Manual-quality review is tracked separately in
`data/manifests/factual/wikimedia_pilot_manual_review.json` and its bounded text
companion. The deterministic seed-42 selection contains 79 unique chunks from
62 parent articles, including 20 random chunks, size extremes, near-maximum
chunks, stream-spanning and distinct-parent samples, every warning/reference
candidate, and suspicious metadata. Fourteen overlapping selections were
deduplicated without losing their reasons. All classifications are currently
pending; source approval and structural validation do not yet imply that the
prepared pilot is training-ready.

The first 79-chunk review failed because it found systematic reference-section
leakage, subminimum fragments, malformed starts, list-heavy material, and
missing source/template values. Its reports remain archived under the old
dataset hash. The corrected v4 preparer now enforces reference appendix
exclusion, a strict 128-token minimum with safe sibling merging, explicit
boundary metadata, configurable list-density checks, and conservative
malformed-source rejection before accounting. The replacement artifact has
  SHA-256 `4c21e6b54cf747a82769951e97222cbe161296e7c0556ee14125df93cc1a4fd0`,
  304 parents, 3,480 chunks, and 1,999,700 tokens. The new 61-chunk review has
zero automatic precheck failures but remains entirely pending human review;
the source is still not authorized for model training.

### Broad-review evidence

Review mode deterministically chooses 500 unique, evenly distributed row
indices with a seed-local offset and reads only their Parquet row groups. It
uses separate `*_review` artifacts, flags reference-like sections, caps each
parent at five retained chunks, and reports diversity and quality warnings.

The current review inspected 15 rows and retained 50 chunks from 14 articles,
totalling 26,435 tokens. Token counts were 43/470.5/528.7/938 for
minimum/median/mean/maximum, and the largest article contributed 10%. Three
reference-section chunks were flagged. No encoding repair, quality rejection,
contamination match, or duplicate was recorded, and output validation passed.
One malformed date boundary was present in the source Parquet itself; it was
left unchanged because an automatic factual reconstruction would be unsafe.
The review artifact is not approved training data.

## Capability-CPT source use

The approved quarantined Wikimedia release `6aa10d73` is eligible for bounded
data planning, with license, attribution, source revision, contamination, and
hash evidence preserved. Its 1,915,008-token train split is insufficient for a
20M-token run at 15% or more without replacement; current capability-CPT plans
therefore cap it at 9%. The internally generated arithmetic source now has a
canonical, validated tokenized release at
`data/processed/capability/verified_arithmetic_v1/`. It contains one unique
training pass (80 logical examples packed into nine records) plus isolated
20-example development and evaluation JSONL splits. Its manifest binds the
unchanged tokenizer, generator/configuration, serializer, logical splits, and
artifacts by SHA-256. The release is eligible as a future scheduled-mixture
source, but `training_authorized=false`; no candidate is executable until the
N-source scheduler is implemented and separately authorized.
