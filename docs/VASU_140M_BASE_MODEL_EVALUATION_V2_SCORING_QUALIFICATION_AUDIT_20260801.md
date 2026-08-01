# VASU-140M Base-Model Evaluation v2 Scoring Qualification Audit — 2026-08-01

Status: pre-commit prompt-free fixture evidence; non-authorizing.

## Root cause

The accepted evaluation-v2 schema proved suite/result metadata integrity, but
it did not yet prove the task contracts, individual scorers, uncertainty
calculation, stratified reporting, robustness pairing, degeneration metrics,
or manual-review selection. Freezing real inventories before those behaviors
were qualified would make prompt identities depend on unproven scoring code.

## Implementation

`evaluation/framework/vasu_140m_base_v2_tasks.py` adds strict, exact-field task
and observation contracts plus deterministic scorers for all six dimensions:

- token-weighted likelihood with finite log-likelihood and perplexity checks;
- factual multiple-choice/cloze ranking reported separately for raw and
  length-normalized likelihood, with ties exposed instead of silently broken;
- strict arithmetic parsing with correct/incorrect/malformed/unanswered,
  truncation, and prompt-leakage outcomes;
- token-level unique-token, repeated n-gram, EOS, empty-output, truncation, and
  terminal-loop metrics;
- paired robustness correctness, consistency, empty-output visibility, and
  paired correctness deltas; and
- manual-review records that preserve blank human judgments and rubric
  dimensions without inferring a semantic score.

`evaluation/framework/vasu_140m_base_v2_statistics.py` adds deterministic
bootstrap intervals, exact dimension/mode summaries, complete strata,
category-stratified manual sampling, a self-hashed qualification report, and a
compact frozen evidence identity that binds the full report. It deliberately
has no aggregate capability score.

The smoke contains prompt/text hashes plus small synthetic scorer observations;
it contains no prompt inventory or protected evaluation content. It covers 18
tasks and all nine required dimension/mode pairs. It does not open held-out
files, invoke a model, open a checkpoint, freeze a production suite, or create
an evaluation/training authorization.

## Frozen pre-commit identities

- Parent commit: `eafd4d708415116663a5c1b1909cc94a6e7b00b3`
- Existing suite/result schema SHA-256:
  `b3d889168196b3c880eb99f9868f14105d6b4b3b5cc84010add96ffb77997c49`
- Task/scorer implementation SHA-256:
  `7777bd7d6a488c21e3937b5b05e1e46153e7fddbcb67784c47dc9e53a5087bf5`
- Statistics/qualification implementation SHA-256:
  `5d4c07d01fe7b833a2b6daf06fb562061ed0731160ffbd0e031eba3527568d71`
- Tests SHA-256:
  `c71190f99822d1c5a887aa64bc1b22c5c2b57f465124ee34053d7ec2a8a62b34`
- Smoke SHA-256:
  `6e43664e5eb1d8f39916ff63f1425ebbb3f339ee69ec6dceb0ec21dcb979e074`
- Frozen compact fixture SHA-256:
  `0ff1801e11d96036a7c645b752754955eda0e4dc5a640c7c57ad1f2e918cfe26`
- Full qualification SHA-256:
  `5fbb82d7eabd6dc025b595776c1dafa814c117785af6f7e1821663b82d59c32c`
- Compact evidence SHA-256:
  `0790f5ef1fd301968f8fbb3e430e32c213128b63365ca7719139fa33e21e0f1b`
- Task inventory SHA-256:
  `82447a39cbbabda6259735befdba1b0719073c4daeb387801305329e554383ef`
- Observation inventory SHA-256:
  `2dc6f263355270d8fe1d19fc71bddc319779068d5a52b5ae0baa0db167121599`
- Score rows SHA-256:
  `7ef49b1ad20d0b01e56d6a505beb637a7f9ac5383e13c28eb09cfa2b5a3b0323`

## Validation

- 29 focused scoring/qualification tests passed.
- Ruff passed for both implementations, tests, and smoke.
- The compact smoke evidence reproduced the frozen fixture exactly.
- The complete smoke report remained JSON-valid and bound by the frozen
  qualification SHA-256.
- The scoring tests plus existing evaluation-v2 schema, capability framework,
  metrics, and comparison regressions passed together: 96 passed in 3.71
  seconds on the first combined run and 2.63 seconds on the final replay.
- `git diff --check` must report no whitespace error; existing CRLF conversion
  warnings in concurrent modified documentation are not package failures.

## Remaining gates

This is scorer qualification, not a frozen production suite. Versioned
development inventories, sealed held-out inventories, source provenance,
contamination commitments, scorer-file bindings, a production suite manifest,
clean post-commit identity, model runtime, checkpoint evaluation, and held-out
opening remain separate review gates.

## Compatibility and non-authorization

The implementation is additive. Existing VASU-31M/60M evaluation outputs,
model architecture, tokenizer, datasets, masks, checkpoints, optimizer and
scheduler state, training configurations, schedules, and exact-resume behavior
are unchanged.

No prompt inventory, held-out content, production suite, model execution,
checkpoint access, evaluation result, data artifact, configuration, schedule,
optimizer, authorization record, checkpoint, or training run was created or
authorized.
