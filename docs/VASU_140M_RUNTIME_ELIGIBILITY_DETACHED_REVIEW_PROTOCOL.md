# VASU-140M Runtime Eligibility Detached Review Protocol

Date: 2026-07-30

Status: review procedure only; non-authorizing

## Purpose

The final runtime commit must equal clean `HEAD` when a future detached
authorization envelope is validated. Writing a runtime-review decision into
the repository would advance `HEAD` and invalidate the commit just reviewed.

The final runtime review must therefore return its decision in chat or another
detached record. It must not create or modify a repository file.

## Procedure

1. Commit this protocol and its read-only smoke.
2. Confirm the repository is clean.
3. Run:

   ```powershell
   python scripts\smoke_vasu_140m_runtime_eligibility.py
   ```

4. Give the exact emitted JSON and current commit to GPT-5.5.
5. GPT-5.5 independently reruns the smoke and focused validation.
6. GPT-5.5 returns a detached accept/reject decision without modifying Git.
7. Preserve the detached decision outside the repository.
8. Only after acceptance may a human separately decide whether to create one
   detached, expiring authorization envelope.

## Review requirements

The reviewer must verify:

- emitted runtime commit equals clean `HEAD`;
- `050d1fa` remains an ancestor;
- gate, tests, original smoke, and production builder hashes are exact;
- the accepted post-commit qualification still reproduces;
- production release, manifest, and receipt paths are absent;
- all authorization and training flags are false;
- the runtime report self-hash is correct.

The reviewer must not create a decision document in `D:\VASU`.

## Non-authorization

Runtime eligibility review does not authorize envelope creation or publication.
Even an accepted detached decision requires a separate explicit human approval
to create one exact envelope, followed by a separate explicit instruction to
invoke publication. Training remains outside this authority.

## Compatibility

This procedure changes no model, checkpoint, tokenizer, dataset, mask, release
byte, optimizer state, or training configuration.
