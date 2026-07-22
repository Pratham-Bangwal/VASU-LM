# Wikimedia factual-pilot release policy

This policy governs promotion of a regenerated Wikimedia factual-pilot artifact
to a training-ready VASU data release. It does not authorize model training.

## Required release evidence

Every candidate must preserve the following immutable, hash-bound artifacts:

- the prepared `documents.jsonl` SHA-256;
- its preparation manifest and summary;
- the global known-defect audit in JSON and text form;
- the deterministic manual-review JSON and text reports;
- the review configuration and random seed;
- the source revision, tokenizer identity, and preparation configuration.

The global diagnostic scan covers all defect families recorded in completed
manual-review archives. Automatic-reject findings must be enforced by the final
chunk-quality gate before chunk counting, token accounting, deduplication, or
JSONL writing. Ambiguous findings must be explicitly marked for manual review;
they may not disappear into an unclassified bucket.

## Promotion gates

A Wikimedia candidate is eligible for approval only when:

1. output validation passes against the recorded dataset hash;
2. the global scan reports zero unexplained matches;
3. the generated deterministic review reports zero automatic precheck failures;
4. human review is complete and contains zero critical rejects;
5. every harmless minor issue has an explicit review note;
6. tokenizer, source revision, deduplication evidence, and contamination evidence
   match the preparation manifest;
7. the approved dataset hash and approved review reports are copied to immutable,
   hash-qualified release paths.

Harmless minor issues may be accepted with notes. A minor issue must not hide a
missing value, formula, quotation, measurement, identifier, list boundary, or
other defect capable of changing factual meaning.

## Remediation limit and freeze

After the global audit, only one final remediation-and-regeneration cycle is
allowed. The deterministic sample generated from that result is the final review
sample. It must not be replaced merely to obtain more favorable selections.

A subsequent sample or regeneration is permitted only when the frozen review
contains a critical reject. Such an exception must document the critical defect,
the generalized rule, the prior dataset hash, and the replacement hash.

Once human review approves the candidate, the dataset hash and review reports
become immutable release artifacts. Changes require a new release identity and
must never overwrite the approved evidence.

## Current status

The frozen final human gate rejected 12 exact chunks from candidate
`ffcbc25f4863f519744212f809ee600bdc7f4a0d5c2d02a0833e1bc4cec6014d`.
Those decisions remain immutable. They are implemented by the versioned
quarantine manifest
`data/manifests/factual/wikimedia_quarantine_ffcbc25f.json`, not by changing
quality heuristics or editing generated JSONL records.

The deterministic quarantine transformation verifies the frozen input hash,
preserves retained JSONL records byte-for-byte and in their original order,
does not renumber chunk IDs, and fails on missing or duplicate IDs. Its approved
release is
`data/processed/pretrain/factual/wikimedia_pilot_release/documents.jsonl`, with
SHA-256
`6aa10d73669ca90ad20f867f14a6368d2191b38094f02aa1679ebaf122962de7`.
It contains 402 parents, 3,597 chunks, and 1,993,564 tokens after quarantining
12 chunks and 6,133 tokens across 9 affected parents. No parent was fully
removed. Existing-policy validation reports zero automatic-reject findings and
zero unexplained findings. The 24 remaining diagnostic findings are the
previously reviewed, non-automatic class.

Release approval records dataset eligibility only; it does not authorize or
start training. The release manifest therefore keeps `training_authorized`
false.
