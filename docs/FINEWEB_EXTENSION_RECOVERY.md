# FineWeb Extension Recovery Plan

## Bounded source-ID smoke result (2026-07-17)

The bounded recovery smoke passed. It selected exactly 100 source IDs by
sorting the 379,247 retained IDs and choosing evenly spread integer positions
from the first through the last ID. The official Hugging Face Dataset Viewer
`/filter` endpoint performed exact `id` predicates; every response reported
revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` in `x-revision`. No rolling
revision or Parquet/full-dataset scan fallback was used.

- IDs found: 100/100.
- Historical `SHA-256(text.strip())` matches: 100/100.
- Hash mismatches, missing IDs, duplicate provider IDs, malformed rows, and
  retrieval errors: zero.
- Dataset-server response bytes: 645,735.
- Accepted source-text bytes inspected: 498,312 (below the 20 MB limit).
- Smoke retrieval duration recorded by progress timestamps: 70.82 seconds.
- Reports:
  `data/manifests/pretrain/fineweb_extension_recovery_smoke.json` and
  `data/manifests/pretrain/fineweb_extension_recovery_smoke.txt`.
- Atomic resume state:
  `data/interim/pretrain/fineweb_extension_recovery_progress.json`.

The capability audit accessed only official read-only metadata/API endpoints:
the Hub dataset metadata endpoint for both the pinned revision and current
revision, plus Dataset Viewer `splits`, `rows`, `filter`, `parquet`, `size`, and
`statistics`. The `parquet` call listed 50 files (about 44.76 GB) but did not
download them; `statistics` was unavailable. The smoke JSON records every
revision-check and exact-filter URL used by the executable smoke itself.

No full source text is stored in any report. This evidence authorizes planning
a broader source-ID reacquisition, but the complete 379,247-record recovery and
the extension document index have not been run.

## Provider-friendly 1,000-ID benchmark (2026-07-17)

The benchmark selected one deterministic, evenly spread set of 1,000
historical IDs and used it for every strategy. Each strategy used an isolated,
initially empty local response cache. Cold timings therefore exclude local
cache hits; provider-side cache state cannot be cleared or measured and is not
claimed as cold. A second pass measured local-cache behavior separately and is
not interpreted as provider throughput.

Official Dataset Viewer documentation supports composite `OR` predicates,
filtered pagination, and up to 100 returned rows. Live checks confirmed exact
OR filters for 5, 10, and 25 IDs and the pinned `x-revision` header. SQL-style
`IN` membership is not documented and was rejected with HTTP 422. No
Dataset Viewer-specific fixed rate is documented and no server response-byte
parameter exists, so the client enforces its own byte limit and handles 429,
Retry-After, transient 5xx responses, timeouts, and bounded backoff.

| Strategy | Cold seconds | Requests | Transients/retries | Exact hashes |
| --- | ---: | ---: | ---: | ---: |
| Serial | 2,277.46 | 1,005 | 5 / 4 | 1,000/1,000 |
| Concurrency 2 | 512.34 | 1,001 | 1 / 1 | 1,000/1,000 |
| Concurrency 4 | 334.96 | 1,001 | 1 / 1 | 1,000/1,000 |
| Concurrency 8 | 257.71 | 1,002 | 2 / 2 | 1,000/1,000 |
| OR batch 5 | 172.66 | 200 | 0 / 0 | 1,000/1,000 |
| OR batch 10 | 86.45 | 100 | 0 / 0 | 1,000/1,000 |
| OR batch 25 | 37.33 | 40 | 0 / 0 | 1,000/1,000 |

All strategies had zero mismatches, missing IDs, duplicate provider records,
malformed records, revision fallbacks, and HTTP 429 responses. Batch 25 with
concurrency 1 is selected because it passed exact correctness with the fewest
provider requests and lowest complexity. Production should use a conservative
2 RPS ceiling (observed latency already kept throughput near 1.1 RPS), four
retries, Retry-After, exponential backoff with bounded seeded jitter, and an
atomic checkpoint every 1,000 completed documents.

The measured projection for all 379,247 IDs is 15,170 filter requests,
approximately 3.93 hours under benchmark conditions, 2.24 GB of downloaded
JSON responses, 2.03 GB of accepted raw text, and 4.06 GB of temporary working
storage. Plan for roughly 3.9-6 hours and at least 10 GiB free because provider
latency and rate limits can change. Reports are
`data/manifests/pretrain/fineweb_extension_recovery_benchmark.json` and `.txt`.
The complete recovery and extension index remain unstarted.

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
- Proof that all 379,247 current provider records reproduce their historical
  hashes; the bounded trial proves only its 100-record sample.
- A newly reconstructed token binary matching the historical output hash.

These gaps prevent classification as `exact_reconstruction_possible`.

## Authoritative reacquisition source

Reacquisition must use `HuggingFaceFW/fineweb-edu`, configuration
`CC-MAIN-2025-26`, split `train`, pinned to revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Authentication was not required by
the historical script, but future provider or Hub policy changes may require a
Hugging Face token.

## Storage planning

The 100-record smoke measured a mean 4,983.12 raw UTF-8 bytes per matched
document, which gives a derived full-recovery estimate of about 1.89 GB. This
is a sample projection, not a provider-published size. Allow at least 10 GiB
for temporary records, hashes, metadata, atomic outputs, and later indexing.

## Source-ID lookup and verification strategy

1. Open the historical SQLite database read-only and load the 379,247 extension
   `(source_id, fingerprint)` pairs.
2. Use the official Dataset Viewer exact-ID filter and require its
   `x-revision` header to equal the pinned revision on every request.
3. Refuse a rolling revision, unavailable filter index, or fallback scan of the
   approximately 44.76 GB / 50-file Parquet export.
4. Compute SHA-256 over UTF-8 `text.strip()` and require equality with the
   stored historical fingerprint.
5. Record missing IDs, duplicate IDs, hash mismatches, unexpected rows, and
   ordering differences.
6. Only after every retained ID matches may the reacquired text be used to
   build the missing document-level near-duplicate index coverage.

The smoke demonstrates bounded lookup at current provider behavior. At its
observed serial rate, a naive 379,247-request run would take about 74.6 hours;
this is a linear planning estimate, not a guarantee. The sampled mean implies
approximately 1.89 GB of raw text and 3.78 GB of temporary working storage.
Scaling the existing 4,876,034,048-byte / 999,992-document original index by
document count gives a rough 1.85 GB extension-index estimate; actual text and
shingle distributions may differ. Use resumable batches of roughly 1,000 and
retain at least 10 GiB free. Before full recovery, evaluate provider-friendly
batching or an official indexed export; never replace bounded lookup with an
unreviewed full scan.

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

- The provider's exact-ID filter index may become unavailable or rate-limited.
- A naive full run requires about 379,247 requests unless a reviewed batching
  mechanism reduces provider load.
- Dataset revision pinning must be checked on every response; current behavior
  reports the expected commit in `x-revision`.
- Unicode or JSON decoding behavior may differ across library versions.
- Recreating approximate text would create false confidence in overlap checks.

## Why token-binary reconstruction is prohibited

The `uint16` token binary preserves token order but not authoritative document
boundaries or source provenance. Separator token patterns can occur in ordinary
content and cannot safely reconstruct original IDs, URLs, raw text, or exact
record boundaries. Decoding the token binary into guessed documents would
invent provenance and is therefore prohibited.
