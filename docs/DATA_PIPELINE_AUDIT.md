# Data Pipeline Audit

Status: read-only assessment; no dataset, tokenizer, schedule, or loader
artifact was modified.

## Current contract

VASU uses a shared 32k byte-level BPE tokenizer, fixed 257-token stored
records, and next-token supervision over 256 positions. Packed masked sources
store a record-width mask and the trainer consumes `stored_mask[1:]`.
Arithmetic packing preserves complete EOS-terminated examples, explicitly pads
record tails, and disables cross-example and PAD-involved loss targets.

## Findings

1. Fixed-width records make step accounting, scheduling, and resume behavior
   reproducible. Changing width would change model context assumptions and
   requires a new data/model compatibility review.
2. The arithmetic release and proposed Candidate E pipeline make mask alignment
   explicit and validate prompt/PAD/cross-example exclusion. This is the right
   pattern for future supervised sources.
3. Deterministic schedules and manifest hashes protect source ordering and
   replay identity. The read-only lineage index now exposes their artifact
   identities across the repository.
4. The major remaining performance question is empirical: measure host-side
   loading, mask transfer, and device utilization separately before optimizing
   packers or introducing prefetching.

## Safe next measurements

- Profile batch construction and host-to-device transfer with the existing
  immutable FineWeb and masked-source manifests.
- Compare pinned-memory and worker settings using a no-update data-loader
  benchmark.
- Record throughput, CPU RAM, GPU allocation, and batch-shape invariants.

None of these measurements authorizes data regeneration, schedule changes, or
training. Existing tokenizer, data, checkpoint, and resume compatibility remain
unchanged.
