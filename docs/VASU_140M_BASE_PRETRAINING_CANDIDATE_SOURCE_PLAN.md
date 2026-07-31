# VASU-140M Base-Pretraining Candidate Source Plan

Status: **candidate plan only; sources are not admitted or acquired.**

## Decision question

Which bounded source strategy can produce the first reproducible VASU-140M
base checkpoint on local hardware while preserving document provenance,
evaluation isolation, and a credible path to later extension?

## Options compared

| Strategy | Strength | Blocking weakness | Decision |
| --- | --- | --- | --- |
| Legacy 257-token FineWeb binaries | Already local; 1.746B tokens | Wrong record lineage and no document-level release contract | Reject as direct input |
| Original FineWeb 1M raw text | Broad and large | Original provider IDs/URLs are incomplete | Reject for v1 admission |
| Recovered FineWeb-Edu extension only | 500,000,478 measured tokens; recovered source IDs and hashes | Narrow single crawl; web-rights/evaluation review still required | Candidate primary |
| Full pinned English Wikipedia only | Strong article IDs/URLs and factual coverage | Narrow encyclopedic distribution; attribution/ShareAlike obligations; acquisition required | Candidate secondary only |
| Fresh FineWeb-Edu sample-10BT | Large, documented educational corpus | Multi-gigabyte acquisition and new shard/legal review | Defer to extension release |
| FineWeb-Edu extension plus bounded Wikipedia | Broad/factual complement with known repository tooling | Two-source admission, cross-dedup, and mixture complexity | Recommended v1 strategy |

## Recommended candidate

Propose `vasu_140m_base_sources_v1` with two independently admitted sources:

1. **Primary:** `fineweb_edu_extension_2025_26`, pinned to repository revision
   `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9:CC-MAIN-2025-26`. Its recovered
   document source IDs reproduced all 379,247 historical text hashes, and its
   existing token stream measures 500,000,478 tokens. The token binary is not
   reused; only separately hash-verified document text may be retokenized.
2. **Secondary:** `wikipedia_en_20231101_planned`, pinned to
   `e6057dc557255a03c9c3c47ceab0eb44353b1bc5:20231101.en`. It provides article
   ID, URL, title, and text fields. A bounded allocation is acquired only after
   a separate admission and acquisition authorization.

The release target is a single non-repeating pass over eligible FineWeb-Edu
extension documents plus enough eligible Wikipedia documents to target 10% of
the final training tokens. Exact token totals and the realized share remain
unknown until both sources pass admission, contamination, deduplication, and
token measurement. No replacement sampling is permitted.

This is a first-stage base-pretraining corpus, not a claim of compute-optimal
training. Any later scale extension requires a new source/release version and
scientific plan; it cannot silently append legacy data.

## Required source-specific gates

### FineWeb-Edu extension

- Create a new VASU-140M admission package bound to the recovered document
  artifact, recovery manifest, combined document-index identity, and official
  dataset revision.
- Resolve the distinction between the dataset's ODC-By database license and
  rights/obligations attached to underlying web documents. Attribution and
  redistribution handling must be explicit.
- Re-run every frozen VASU-140M evaluation inventory against normalized text;
  existing historical checks are insufficient for the new suite.

### Wikipedia

- Bind the existing primary-source license review, CC-BY-SA/GFDL obligations,
  attribution metadata, shard inventory, and immutable revision.
- Authorize only bounded, resumable acquisition after admission.
- Preserve article provenance and source notices; exclude unresolved imported
  or fair-use material when reliable permission cannot be established.

### Combined release

- Exact and near-deduplicate within and across sources before document-level
  split assignment.
- Keep parent documents and semantic duplicates in one split.
- Freeze source proportions only after exact tokenizer measurement.
- Pack independent full-loss 513-token records with zero boundary/PAD targets.
- Publish only under a new non-overwriting `vasu_140m/base_pretraining/v1`
  identity after implementation, qualification, and one-build review.

## Primary evidence

- FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- FineWeb paper: https://arxiv.org/abs/2406.17557
- ODC-By 1.0: https://opendatacommons.org/licenses/by/1-0/
- Pinned Wikipedia dataset: https://huggingface.co/datasets/wikimedia/wikipedia/tree/e6057dc557255a03c9c3c47ceab0eb44353b1bc5/20231101.en
- CC BY-SA 3.0: https://creativecommons.org/licenses/by-sa/3.0/
- GFDL 1.3: https://www.gnu.org/licenses/fdl-1.3.html

## Compatibility and non-authorization

The proposal retains the frozen tokenizer and new 513-token full-loss record
contract. It does not modify or relabel existing binaries, datasets, masks,
manifests, checkpoints, or evaluation artifacts. It does not authorize registry
mutation, acquisition, processing, publication, configuration, or training.
