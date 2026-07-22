# FineWeb Extension Recovery Plan

## Production recovery, index, and factual gate result (2026-07-17)

The full retained-extension source recovery completed successfully against the
official Hugging Face Dataset Viewer exact-ID filter for
`HuggingFaceFW/fineweb-edu`, configuration `CC-MAIN-2025-26`, split `train`,
pinned to revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`.

Protected preflight evidence passed before acquisition:

- historical token binary SHA-256:
  `d62efa0521365fa634d1e8f34bf8c84764a8066a9cc53430f9fc4bb89cabf51e`;
- historical fingerprint database SHA-256:
  `de30a1a10ae513bd663fdd7655d6e0251db65b93929efb087071be4754c4b1a1`;
- historical fingerprint rows, source IDs, and fingerprints: 379,247;
- SQLite integrity: `ok`;
- no training process was started by the recovery workflow.

Production acquisition used the benchmark-selected 25-ID OR filter batches,
concurrency 1, a conservative 2 RPS ceiling, four retries, pinned-revision
enforcement, atomic progress, and no rolling-revision or full-Parquet fallback.
One Dataset Viewer behavior was handled explicitly: the exact
`{"error":"A query parameter is invalid"}` HTTP 422 response sometimes occurred
transiently for valid single-ID filters and was retried; generic 422 responses
remain permanent failures.

Final recovery result:

- status: `complete`;
- accepted records: 379,247;
- unique source IDs: 379,247;
- historical exact text-hash matches: 379,247;
- missing IDs, hash mismatches, duplicate provider IDs, malformed records,
  duplicate accepted IDs, and unrecovered retrieval errors: zero;
- accepted text bytes: 2,031,365,373;
- total characters: 2,021,516,867;
- provider requests: 15,617;
- provider retry count: 307;
- provider status counts: 15,281 HTTP 200, 26 HTTP 422, 289 HTTP 500, and
  21 HTTP 502;
- fallback events: 27;
- full source text is stored only in the compressed recovery artifact, not in
  the summary reports.

Recovered artifact:

```text
data/interim/pretrain/fineweb_extension_recovered.jsonl.gz
```

Artifact size: 797,342,910 bytes.

Artifact SHA-256:

```text
c90f9e21d9b73324b9165cf1fb7ffbc274fbba5ccba5ac22b7cbe48abb6d7f1e
```

Reports:

```text
data/manifests/pretrain/fineweb_extension_recovery_production.json
data/manifests/pretrain/fineweb_extension_recovery_production.txt
```

The recovered text was then indexed with normalization version
`vasu_cross_source_nfc_casefold_ws_v1`.

Extension index result:

- index path:
  `data/manifests/pretrain/fineweb_extension_document_index.sqlite3`;
- metadata path:
  `data/manifests/pretrain/fineweb_extension_document_index_metadata.json`;
- SHA-256:
  `d093770179204b45af1a5a824d5a5826a6f0626b3c8825aa81ac8d3557acf16e`;
- indexed documents: 379,247;
- rejected records: zero;
- duplicate hashes: zero;
- records with complete source IDs: 379,247;
- LSH bucket count: 3,033,976;
- SQLite integrity: `ok`.

Combined FineWeb coverage was written to:

```text
data/manifests/pretrain/fineweb_combined_coverage.json
```

It covers both `fineweb_original` and `fineweb_extension`, has no missing
coverage entries, and reports `training_ready: true` for downstream
cross-source deduplication gates.

The Wikimedia broad-review overlap check was rerun against the original and
extension indexes together. It found zero overlap candidates and wrote:

```text
data/manifests/factual/wikimedia_fineweb_combined_overlap.json
```

The factual pilot dry-run then passed with FineWeb cross-deduplication reported
as available and without downloading additional data. The default factual pilot
and any model training remain separately unauthorized.

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

No full source text is stored in any report. This bounded smoke is retained as
historical evidence and has been superseded by the completed production
recovery, extension index, combined coverage manifest, and factual-gate dry-run
recorded above.

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
This benchmark was later used to run the completed production recovery and
extension index recorded above.

## Status and classification

Recovery classification: **`exact_source_id_reacquisition_completed`**.

This classification means all retained document IDs were authoritatively
reacquired from the pinned provider revision and every recovered
`text.strip()` value matched its historical SHA-256 fingerprint. It does not
claim that the historical token binary was rewritten or replaced; the protected
binary remains unchanged.

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

## Historical missing evidence now resolved

- Local raw extension document text is now represented by the recovered gzip
  JSONL artifact above.
- Proof that all 379,247 provider records reproduce their historical hashes is
  recorded in the production recovery reports.
- Compatible document-level normalized exact/near index coverage is now
  recorded in the extension index metadata and combined coverage manifest.

No newly reconstructed token binary was produced, and none is required for the
document-deduplication gate because the original extension token binary remains
the protected training artifact.

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
