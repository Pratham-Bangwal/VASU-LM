# VASU-140M Prompt-Matrix Rejection Remediation

Status: author-side remediation complete; independent reconsideration pending.

## Root cause

The independent review correctly rejected the submitted matrix under its stated
criteria. It found two different classes of overlap:

1. 192 development/held-out normalized-answer overlaps; and
2. 2,847 exact FineWeb prompt candidates plus nine substantive semantic
   FineWeb candidates requiring source-document quarantine.

The first class is not sufficient evidence of task leakage. Prompt, item-ID,
semantic-family, and parent-document overlap counts are all zero. Arithmetic
and short-answer tasks necessarily reuse ordinary outputs such as integers.
The already-versioned VASU-140M short-answer policy therefore treats an answer
without its prompt as audit-only evidence. This remediation does not weaken
prompt, fragment, semantic-family, or parent-document isolation.

The second class is actionable contamination evidence. Exact prompt matches
and independently rejected semantic matches must exclude their complete source
documents from any future base-pretraining release.

## Remediation

The immutable rejection and its complete non-plaintext JSON evidence remain
preserved at commit `e130f60c84ab340ff80745f78ac7054fb634c783`.

The additive quarantine builder:

- verifies the independent report's embedded SHA-256;
- verifies that private-key and held-out-exposure flags remain false;
- resolves opaque `SHA-256(str(source_record_ordinal))` commitments;
- includes only independent quarantine dispositions;
- deduplicates multiple evaluation matches against the same source document;
- writes atomically and refuses overwrite; and
- keeps release-building, evaluation, and training authorization false.

Frozen output:

- path: `configs/data/exclusions/vasu_140m_fineweb_semantic_quarantine_20260804.json`
- independent quarantine candidates: 2,856
- unique FineWeb source documents: 2,659
- embedded quarantine SHA-256:
  `cfd31cf8ea68d27994b1d85caba163de1141a9ead1a8bc87c90ed39c7e67837b`
- file SHA-256:
  `f440bde55cb499a6294fc9739c65e0955e9d61460535f5dca0c9e761b78f3c58`

## Scientific disposition

The rejected prompt matrix is not silently promoted. A fresh independent
review must confirm both of the following before source-admission review:

1. answer-only overlap is audit evidence under the existing short-answer
   policy and is not prompt leakage when all stronger isolation dimensions are
   zero; and
2. the 2,659-document quarantine fully and exactly resolves every independently
   rejected FineWeb candidate.

Likelihood remains deferred until accepted source admission and document-level
train exclusions. No model, tokenizer, dataset, checkpoint, optimizer,
schedule, training configuration, or authorization record changed.
