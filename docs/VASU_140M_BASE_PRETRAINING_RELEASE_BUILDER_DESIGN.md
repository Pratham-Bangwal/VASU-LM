# VASU-140M Base-Pretraining Release Builder Design

Status: design plus fixture-qualified full-loss record foundation and
two-pass qualification-core implementation; production construction and
publication remain unauthorized.

## Root cause and scope

The accepted 513-token record module is intentionally instruction-oriented:
it masks a prompt prefix and supervises a response. Base pretraining instead
needs full-loss text while still masking artificial chunk/document boundaries
and PAD transitions. Reusing the instruction compiler would silently change
the scientific objective; treating a whole web document as one logical example
would also exceed the fixed record width.

The additive `vasu/data/vasu_140m_base_records.py` foundation therefore accepts
only already admitted, normalized, bounded text chunks. It masks the first
token of each chunk, supervises every later content token and EOS, and reuses
the accepted 513-token packer. A stateful isolation validator and generator
support one-pass record construction without retaining packed corpus bytes in
memory. The module performs no source discovery, acquisition, normalization,
deduplication, publication, scheduling, or training.

The additive `vasu/data/vasu_140m_base_release.py` implementation now covers
the deterministic qualification core. It validates a strict release
specification, recursively hash-binds semantic source-evidence envelopes,
streams source-separated token/mask/lineage artifacts through two independent
scratch builds, audits every serialized record, and returns a non-authorizing
qualification report. Fixture and production evidence are explicit, mutually
exclusive scopes; fixture evidence can never claim production eligibility.
The publication transaction and detached one-build authorization remain a
later implementation gate.

## Alternatives

| Alternative | Decision |
| --- | --- |
| Reuse prompt/response masking with an empty prompt | Reject: obscures the base-text objective and boundary semantics. |
| Concatenate all text before fixed-width slicing | Reject: destroys document/chunk lineage and supervises synthetic boundaries. |
| Combine all sources into one binary | Reject: prevents source-specific replay, auditing, and schedules. |
| Add a source-separated streaming builder above the frozen packer | Adopt: preserves lineage, scale, and compatibility. |

## Required input bundle

The future builder accepts one immutable release specification and no default
source paths. Before reading text, it must validate:

1. an independently accepted release specification and exact output version;
2. one approved VASU-140M source-admission package per source;
3. a separately authorized and completed acquisition receipt per source;
4. immutable raw-shard inventories and byte hashes;
5. normalized/filter manifests with transformation IDs and rejection counts;
6. quarantine decisions for licensing, quality, safety, and contamination;
7. exact and near-duplicate disposition evidence within and across sources;
8. a document-level split assignment with no parent/duplicate-group crossing;
9. exact per-source selection indexes, with no replacement sampling;
10. every frozen evaluation inventory commitment used by contamination scans;
11. the accepted tokenizer, family, model configuration, record specification,
    base-record implementation, and repository identities.

An approval state without matching files and hashes is insufficient. The
builder never downloads data and never repairs or guesses missing lineage.

## Deterministic logical order

Source order, source revision, selection-index order, parent document ID,
chunk index, transformation ID, and split are identity-bearing. A source may
have only one revision. A parent document and its duplicate group may occur in
only one split. Duplicate text hashes and duplicate chunk IDs fail closed.

The builder preserves source-separated artifacts. It does not invent a
training mixture or interleave sources; a later immutable experiment schedule
must consume these source streams explicitly. Realized source shares are
reported from supervised token counts rather than inferred from requested
weights.

## Full-loss record contract

Every admitted chunk:

- is non-empty and at most 512 content tokens plus EOS;
- contains no PAD, UNK, BOS, or EOS token in source content;
- retains source/revision/document/chunk/transformation lineage;
- stores mask zero at the first chunk token;
- stores mask one for all later content tokens and terminal EOS;
- is packed whole within one `uint16[513]` record;
- has PAD tail and synthetic cross-chunk targets masked in `uint8[513]`;
- derives 512-position training masks as `stored_mask[1:]`.

Chunking and normalization occur upstream under versioned, reviewed policies.
The release builder must not silently split, truncate, normalize, or reorder a
chunk.

## Output contract

The only production version in this design is:

```text
data/processed/vasu_140m/base_pretraining/v1/
  <source_id>/train.tokens.bin
  <source_id>/train.mask.bin
  <source_id>/train.lineage.jsonl
  <source_id>/development.tokens.bin
  <source_id>/development.mask.bin
  <source_id>/development.lineage.jsonl
  <source_id>/evaluation.tokens.bin
  <source_id>/evaluation.mask.bin
  <source_id>/evaluation.lineage.jsonl
  release.internal.json
data/manifests/vasu_140m/base_pretraining/v1.json
```

The manifests bind byte counts, record counts, real/PAD/supervised token
counts, source shares, every input identity, per-artifact SHA-256, logical
lineage SHA-256, builder/runtime identity, and `training_authorized=false`.
No schedule, experiment config, or checkpoint is part of the release.

## Phase 1: read-only qualification

Qualification requires a clean reviewed commit and absent production paths.
It performs two sequential builds in independent canonical scratch roots so
large corpora do not require simultaneous duplicate storage. Before each pass
it checks disk capacity for the measured source size, expected token/mask and
lineage output, safety margin, and failure evidence.

Each pass streams chunks and outputs; no corpus-wide in-memory materialization
is allowed. Qualification requires byte-identical logical, token, mask,
lineage, internal-manifest, and external-manifest-template hashes across both
passes. It performs:

- complete serialized dtype, width, token-range, token/mask length, PAD-tail,
  boundary-mask, EOS, shifted-target, and record-count audits;
- encode/decode/encode token round trips for every chunk;
- complete source/document/chunk/split lineage reconciliation;
- exact source share and rejection/quarantine accounting;
- independent source and split isolation checks;
- post-validation mutation detection before final qualification identity;
- scratch cleanup on success and evidence preservation on failure.

The qualification report binds the exact expected production bytes but writes
nothing to the production paths and keeps `publication_authorized=false` and
`training_authorized=false`.

## Phase 2: detached one-build authorization and publication

Publication requires a canonical authorization envelope outside the repository
with two distinct human identities: release owner and independent reviewer.
It binds the clean runtime commit, accepted implementation/post-commit
decisions, builder and tests, qualification report, all input hashes, all
expected output hashes, exact production paths, expiration, and one-use ID.

The publisher must acquire an exclusive lock, reject links/junctions and
unexpected filesystem boundaries, revalidate all inputs before and after the
build, stream into a same-volume staging directory, fsync/close handles,
validate every staged byte, write the internal manifest last, and atomically
rename the directory. It publishes the external manifest only afterward and
writes a consumed-envelope receipt without modifying the envelope.

No retry is automatic. A crash leaves either no production release, a complete
directory awaiting external-manifest recovery, or an explicitly quarantined
incomplete state. Recovery requires exact staged/output identities and a new
reviewed action. Reusing an envelope or overwriting any output fails closed.

## Threat model and required tests

Implementation review must prove wrong-family/tokenizer/record/admission/
acquisition/selection identities, source mutation, duplicate leakage, split
leakage, source-order changes, overwidth chunks, reserved tokens, truncated
files, swapped token/mask files, disk exhaustion, injected write/rename/fsync
failures, concurrent invocation, stale locks, Windows junctions/open handles,
post-validation mutation, authorization reuse, and partial publication all
fail without corrupting an existing artifact.

## Compatibility and non-authorization

This design and foundation are additive. Existing instruction releases,
VASU-31M/60M data, checkpoints, tokenizer, model architecture, training
configs, schedules, and optimizer states are unchanged. A future published
base release is compatible only with `vasu_140m_v1` and the frozen tokenizer.

No source is admitted or acquired by this design. No production data,
manifest, authorization envelope, schedule, config, checkpoint, optimizer
state, or training update is created or authorized.
