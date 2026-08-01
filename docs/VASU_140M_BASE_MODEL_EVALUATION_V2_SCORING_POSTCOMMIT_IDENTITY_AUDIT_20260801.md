# VASU-140M Evaluation-v2 Scoring Post-Commit Identity Audit

Date: 2026-08-01

Status: **clean-checkout author evidence; independent review pending.**

## Purpose

Prove that the independently accepted scoring implementation remains identical
after its isolated commit and after the repository line-ending remediation,
including under the machine's default Windows `core.autocrlf=true` behavior.

## Commit Lineage

- Accepted scoring commit:
  `926add62666b1f54d9aaac8d1c8ba1fda56108c2`
- Line-ending policy commit / reviewed runtime commit:
  `92656ac0c8b8652fca3758a156e97a717d6b89a3`
- The scoring commit is an ancestor of the reviewed runtime commit.

## Clean-Checkout Result

A detached worktree at `92656ac0c8b8652fca3758a156e97a717d6b89a3`
was created with the default system Git configuration. Its initial status was
clean. The checkout preserved the accepted LF identities for the tokenizer,
task/scorer implementation, statistics implementation, tests, smoke, and
pre-commit fixture.

The combined line-ending, scoring, schema, framework, metrics, and comparison
suite passed: 99 passed in 3.50 seconds. Ruff passed. The temporary worktree was
removed after validation.

## Exact Identities

- Tokenizer SHA-256:
  `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a`
- Task/scorer SHA-256:
  `7777bd7d6a488c21e3937b5b05e1e46153e7fddbcb67784c47dc9e53a5087bf5`
- Statistics SHA-256:
  `5d4c07d01fe7b833a2b6daf06fb562061ed0731160ffbd0e031eba3527568d71`
- Tests SHA-256:
  `c71190f99822d1c5a887aa64bc1b22c5c2b57f465124ee34053d7ec2a8a62b34`
- Smoke SHA-256:
  `6e43664e5eb1d8f39916ff63f1425ebbb3f339ee69ec6dceb0ec21dcb979e074`
- Pre-commit fixture SHA-256:
  `0ff1801e11d96036a7c645b752754955eda0e4dc5a640c7c57ad1f2e918cfe26`
- Post-commit fixture SHA-256:
  `186ad21ae60e637f73b7815b9f607df921fcc4559bbacfc71b0d1ac609d3db96`
- Post-commit compact evidence SHA-256:
  `5a681598e81d253d140cf66e30344de97eb352205e2802d543cd263cfb628f1f`
- Post-commit qualification SHA-256:
  `de5937040c96320a30796dbfd431eadda9f18ba69707f3db415f4c1c856179e0`

## Expected Identity Delta

Compared with the accepted pre-commit fixture, exactly three fields change:

- `repository_commit`: `eafd4d7...` to `92656ac...`;
- `evidence_sha256`: `0790f5e...` to `5a68159...`; and
- `qualification_sha256`: `5fbb82d...` to `de59370...`.

All scoring inputs, observations, rows, summaries, bootstrap settings,
implementation/test/smoke hashes, safety flags, and non-authorization fields
remain identical.

## Compatibility and Non-Authorization

No model architecture, tokenizer semantics, dataset, mask, checkpoint,
optimizer/scheduler state, configuration, schedule, evaluation output, or exact
resume behavior changed. No prompt content, held-out content, model, checkpoint,
or optimizer was opened or executed.

This evidence does not authorize inventory construction, suite freezing,
held-out opening, source acquisition, data construction, model execution,
checkpoint creation, evaluation publication, optimizer updates, authorization
records, or training.
