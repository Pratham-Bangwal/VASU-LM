# VASU-140M Base-Pretraining Contamination Scanner Qualification Audit

Date: 2026-08-01

Status: **author-side fixture qualification rebased to the committed evaluation-v2 inventory chain; non-authorizing.**

## Root Cause

The source-admission contract requires exact, fragment, and semantic evaluation
isolation, but a hash list alone cannot find a benchmark prompt embedded inside
a longer source document. Short accepted answers are especially vulnerable:
hashing an entire source document will never equal the hash of a one-word
answer. A production admission decision therefore needs a reviewed pre-split
scanner that understands each commitment's exact word-span length.

## Implemented Behavior

`vasu/data/vasu_140m_contamination_scan.py`:

- scans caller-provided source documents before any train/development split;
- consumes the inventory contract's prompt, accepted-answer, eight-word, and
  semantic commitments without opening held-out content;
- computes rolling exact spans at every committed word length;
- applies NFC normalization, whitespace collapse, and Unicode case folding so
  capitalization or spacing changes do not evade matching;
- detects full prompts, embedded fragments, and short answers independently;
- accepts externally produced MinHash/LSH candidates only when each candidate
  has exactly one explicit human `clear` or `reject` decision;
- hash-binds the exact ordered candidate and human-decision lists in addition
  to the externally generated semantic-search evidence;
- rejects duplicate or unresolved document/item/candidate identities,
  non-finite similarities, source substitution, incomplete review, and report
  mutation;
- emits only source-document hashes, findings, outcomes, counts, and bound
  semantic-search evidence; and
- keeps source admission, release construction, and training authority false.

The implementation does not acquire data, write a release, split documents,
run a model, create an optimizer, or expose a training path.

## Synthetic Qualification

The fixture scans three in-memory documents. It rejects one document containing
an embedded ten-word prompt plus its eight-word fragment and a second document
containing lowercase `mars` against the capitalized accepted answer `Mars`.
One unrelated document remains eligible after an explicit semantic-clear
decision. The result has three exact findings, two rejected documents, one
eligible document, and no semantic rejection.

## Frozen Identities

- Committed inventory-builder anchor: `84529a8f22a3c053c8327d8f33de662211de240e`
- Implementation SHA-256: `c1005d4b26d651b1fce211f3425c63f0562ebed1a28cf1c18d4a7018123d0a8c`
- Tests SHA-256: `6b2934e69d462f0ecba4bea9013b2179f34b54e9b3456c5c29d058d409e932e0`
- Smoke SHA-256: `be0db71104553227441e5cb5e3ee362f61ab266c8c5692859458ed54c943ae74`
- Fixture SHA-256: `f5c6d2a41d1802c492a7296509c74187e571999f789dbbbfa181f40115a5b887`
- Qualification report SHA-256: `b856613ca9526feae0700807039a3a6c2c9f8d5dcd513533856a5340c3240993`
- Contamination inventory SHA-256: `f2ff9b6f7f5b1cbfb79ed8fda08fa8dbf884bb4c224087de804c2568e7afb260`

## Validation

- `python -m pytest tests\test_vasu_140m_contamination_scan.py -q`: 8 passed.
- Ruff passed for the implementation, tests, and smoke.
- The smoke reproduced the frozen fixture exactly.
- The combined inventory, builder, scanner, and plan suite passed: 73 passed,
  1 skipped. The skipped Windows directory-symlink test requires privileges
  unavailable in the current process; link/junction rejection remains in code.

## Remaining Gates

The scanner now runs against the committed inventory contract, corrected
construction plan, and plan-bound fixture builder. Real source scanning still
requires production inventories, acquired source documents, externally
generated semantic-search evidence, and completed human decisions. A passing
scan is evidence for a later source-admission review; it is not source approval
by itself.

## Compatibility and Non-Authorization

The package is additive. Model architecture, tokenizer, existing datasets and
masks, checkpoints, optimizer/scheduler state, Trainer behavior, and exact
resume remain unchanged.

No real evaluation inventory, held-out plaintext, source data, source
admission, acquisition, release, schedule, configuration, optimizer,
authorization record, checkpoint, model execution, or training run was
created, modified, authorized, or started.
