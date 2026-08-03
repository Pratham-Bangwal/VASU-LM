# VASU-140M Private Curator Sealing Implementation

Date: 2026-08-03

Status: implementation qualified; real sealing not yet executed.

## Purpose

The private intake report now proves that 1,500 independently authored records
exist outside the repository. This package provides the next fail-closed step:
convert those records to evaluation-v2 held-out tasks entirely in memory,
encrypt each dimension's payload with the accepted Age X25519 public recipient,
and persist only ciphertext plus public provenance and contamination indexes.

## Security and integrity behavior

The implementation:

- re-runs the private intake validator and matches every file hash/count to the
  committed intake receipt;
- binds the exact public Age recipient and its SHA-256 fingerprint;
- never names, reads, or opens the private key;
- converts and validates held-out tasks in memory;
- sends plaintext payload bytes to `age` through standard input rather than a
  plaintext temporary file;
- requires the Age v1 ciphertext header;
- creates five public manifests with `opening_authorized=false`,
  `production_suite_frozen=false`, and all evaluation/training flags false;
- publishes the all-five candidate directory atomically and refuses overwrite;
- removes only its isolated staging directory after a pre-promotion failure;
- emits a detached sealing receipt bound to the intake, runtime commit,
  recipient, inventory identities, and counts.

The production CLI requires a clean worktree before execution:

```powershell
python scripts\seal_vasu_140m_private_curator_inputs.py
```

## Qualification

Fixture tests cover successful ciphertext-only construction, bound public file
validation, plaintext absence, overwrite rejection, source mutation,
recipient-fingerprint substitution, injected encryption failure, staging
cleanup, and non-Age output rejection. The read-only smoke additionally
reproduces the real intake identity and confirms the Age executable is present
without invoking sealing.

Frozen evidence:
`evaluation/fixtures/vasu_140m_private_curator_sealing_qualification_v1.json`.

## Compatibility and authorization

This package changes no model, tokenizer, dataset, mask, checkpoint,
optimizer/scheduler state, training configuration, or exact-resume behavior.
Implementation qualification does not authorize held-out opening, suite
freezing, model execution, source admission, data release, or training.
