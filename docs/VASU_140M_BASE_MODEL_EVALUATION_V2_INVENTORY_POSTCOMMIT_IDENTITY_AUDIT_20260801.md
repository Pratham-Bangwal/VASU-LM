# VASU-140M Evaluation-v2 Inventory Post-Commit Identity Audit

Date: 2026-08-01

Status: **clean-checkout author evidence; independent review pending.**

## Purpose

Prove that the independently accepted inventory contract remains byte-identical
and behaviorally reproducible after its isolated commit.

## Commit Lineage

- Accepted package base: `f2dbda855a4255c06ad082dd49c62cb0adab4e11`
- Inventory contract commit: `858c8e9d1a8a977d2205065b3157e73ad34c3cf6`

The package base is the direct parent of the inventory contract commit.

## Clean-Checkout Qualification

A temporary detached worktree at the inventory commit was created with the
default Windows Git configuration (`core.autocrlf=true`). Initial status was
clean. The line-ending, inventory, scoring, schema, framework, metrics, and
comparison suite passed: 129 passed in 3.97 seconds. Ruff passed. The smoke
completed all twelve temporary inventories and the worktree was removed.

## Preserved File Identities

- Implementation SHA-256:
  `1ead769715f72faa13be981bf92199d81db30952bc17a02812d51c8055b4905a`
- Tests SHA-256:
  `0ddb706517a313ddaf2d0a6d8b93c89352357ef1175df2fa157b618c372db606`
- Smoke SHA-256:
  `de52fe868e6ad0a78ed0d7cca9293ab504dc5cf6e16de037ef4b63ed9adfd44d`
- Accepted pre-commit fixture SHA-256:
  `4dcdfaf5b04d4864e00d0244dde178135b987c37bc1a923280e2133ef7111dc5`
- Frozen post-commit fixture SHA-256:
  `aad4c03bdefde2afd00792fbe1adeafadf6bba9a6f8f5c0f3f8a9d40a123ce8e`
- Post-commit qualification SHA-256:
  `5a17ef13f059ba39bc2c4045d8f176585e52b7abc08e64e360bf8950fdef1e76`

## Expected Identity Delta

Exactly four top-level fixture fields differ from accepted pre-commit evidence:

- `repository_commit`, now the inventory contract commit;
- `development_inventory_sha256s`;
- `held_out_inventory_sha256s`; and
- `qualification_sha256`.

Every generated inventory manifest includes `repository_commit`, so all twelve
inventory self-identities must change when the parent changes. This is expected
lineage binding rather than an implementation change. Implementation, tests,
smoke, counts, access outcomes, cleanup, and non-authorization flags remain
identical.

## Compatibility and Non-Authorization

Model architecture, tokenizer semantics, existing datasets and masks,
checkpoints, optimizer/scheduler state, configurations, schedules, evaluation
outputs, and exact resume remain unchanged.

No production prompt inventory, real encryption key, held-out plaintext, suite
freeze, source acquisition, data construction, model/checkpoint execution,
optimizer update, authorization record, or training run was created or
authorized.
