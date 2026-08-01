# Repository Line-Ending Reproducibility Audit

Date: 2026-08-01

Status: **author-side remediation evidence; independent review pending.**

## Root Cause

The accepted VASU-140M scoring package was committed as
`926add62666b1f54d9aaac8d1c8ba1fda56108c2`. A clean detached checkout using
the machine-wide Windows Git setting `core.autocrlf=true` converted tracked LF
text to CRLF. That changed the byte hashes of `assets/tokenizer.json`, scoring
implementations, tests, smoke, and fixture. Two tokenizer-identity tests failed
and the frozen scoring fixture no longer reproduced in that checkout.

The Git blobs themselves are correct LF bytes. A second clean detached checkout
with conversion disabled passed all 96 scoring/schema regression tests. Its
scoring fixture differed only because the smoke correctly reports the new
post-commit repository identity rather than the pre-commit parent identity.

## Alternatives

1. Rely on each contributor to set `core.autocrlf=false`. This is not portable
   and cannot protect CI or new contributors.
2. Normalize line endings before hashing. This weakens byte-level artifact
   identity and could hide unintended mutations.
3. Commit a repository-owned `.gitattributes` policy. This makes checkout
   behavior deterministic while preserving exact byte hashes.

The third option is selected.

## Remediation

`.gitattributes` now applies `text=auto eol=lf` repository-wide, retains CRLF
only for Windows batch/command files, and explicitly disables text conversion
for model, token, array, image, PDF, and archive artifacts.

`tests/test_repository_line_endings.py` verifies:

- representative tokenizer, scoring, fixture, script, test, and policy files
  resolve to LF checkout semantics;
- representative `.bin` and `.pt` research artifacts disable text handling;
  and
- currently checked-out hash-bound files contain no CRLF bytes.

An inventory of all tracked candidate text blobs found zero committed CRLF
sequences and zero NUL bytes. The policy therefore preserves existing Git blob
content rather than renormalizing scientific artifacts.

## Compatibility

Model architecture, tokenizer vocabulary/content, datasets, masks, checkpoints,
optimizer/scheduler state, evaluation semantics, and exact resume are unchanged.
The tokenizer and scoring hashes remain the accepted LF identities. Existing
binary artifacts are explicitly protected from conversion.

## Non-Authorization

This remediation does not authorize data acquisition, release construction,
model or checkpoint execution, configuration or schedule creation, optimizer
updates, evaluation publication, authorization records, or training. It does
not amend or replace commit `926add6`.
