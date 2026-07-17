# FineWeb Extension Recovery Plan

## Status and classification

Recovery classification: **`source_id_reacquisition_possible`**.

This classification means the retained document IDs and historical hashes are
sufficient to attempt authoritative reacquisition and verify individual
documents. It does not claim that exact reconstruction has already been proven.
No extension data was downloaded or rebuilt during the original-index task.

## Discovered evidence

- Provider repository: `HuggingFaceFW/fineweb-edu`.
- Configuration: `CC-MAIN-2025-26`.
- Split: `train`.
- Pinned revision: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.
- Streaming order: provider order, `shuffle=false`.
- Recorded seed: 42, although no shuffle was applied.
- Rows seen: 379,450.
- Documents retained: 379,247.
- Documents excluded as original-source exact duplicates: 203.
- Target tokens: 500,000,000; actual tokens: 500,000,478.
- Tokenization: `assets/tokenizer.json`, SHA-256
  `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a`.
- Serialization: `text.strip()` followed by literal `\n[EOS]\n`.
- Historical exact hash: SHA-256 of UTF-8 `text.strip()`.
- Historical SQLite table: `fingerprints(fingerprint, origin, source_id)`.
- All 379,247 retained extension fingerprints have a preserved source ID.
- Output token binary SHA-256:
  `d62efa0521365fa634d1e8f34bf8c84764a8066a9cc53430f9fc4bb89cabf51e`.

## Missing evidence

- Local raw extension document text.
- A retained local provider Parquet/Arrow cache.
- Proof that current dataset-library streaming resolves byte-for-byte identical
  records despite the pinned dataset revision.
- A completed source-ID reacquisition trial.
- A newly reconstructed token binary matching the historical output hash.

These gaps prevent classification as `exact_reconstruction_possible`.

## Authoritative reacquisition source

Reacquisition must use `HuggingFaceFW/fineweb-edu`, configuration
`CC-MAIN-2025-26`, split `train`, pinned to revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Authentication was not required by
the historical script, but future provider or Hub policy changes may require a
Hugging Face token.

## Storage planning

The historical 500M-token target corresponds to a derived estimate of roughly
1.9 GB of raw JSONL text using the original source's measured byte/token ratio.
This is not a provider-published size. Allow at least 5-10 GB for streaming
cache, temporary records, hashes, metadata, and atomic outputs. Measure actual
network and disk use during a separately authorized bounded trial.

## Source-ID lookup and verification strategy

1. Open the historical SQLite database read-only and load the 379,247 extension
   `(source_id, fingerprint)` pairs.
2. Stream the exact pinned provider/configuration/split without shuffling.
3. Select rows whose provider ID occurs in the retained ID set.
4. Compute SHA-256 over UTF-8 `text.strip()` and require equality with the
   stored historical fingerprint.
5. Record missing IDs, duplicate IDs, hash mismatches, unexpected rows, and
   ordering differences.
6. Only after every retained ID matches may the reacquired text be used to
   build the missing document-level near-duplicate index coverage.

## Acceptance criteria

- All 379,247 retained source IDs are found exactly once.
- Every reacquired normalized text matches its historical fingerprint.
- The 203 historical original-source duplicates remain excluded.
- Provider order and retained-document order are reproduced and recorded.
- Tokenization with the unchanged tokenizer and separator policy produces the
  recorded token count and, for exact reconstruction, the historical binary
  SHA-256.
- All acquisition metadata, provider revision, hashes, counts, and failures are
  written atomically.

Any missing ID or hash mismatch blocks exact coverage and requires manual
investigation. Near-equivalent text is not an acceptable substitute.

## Risks

- Dataset streaming implementation or remote shard layout may have changed.
- Provider IDs may not support direct random access, requiring a scan.
- Dataset revision pinning may not fully pin external resources.
- Unicode or JSON decoding behavior may differ across library versions.
- Recreating approximate text would create false confidence in overlap checks.

## Why token-binary reconstruction is prohibited

The `uint16` token binary preserves token order but not authoritative document
boundaries or source provenance. Separator token patterns can occur in ordinary
content and cannot safely reconstruct original IDs, URLs, raw text, or exact
record boundaries. Decoding the token binary into guessed documents would
invent provenance and is therefore prohibited.
