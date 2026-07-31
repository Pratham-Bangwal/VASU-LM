# VASU-140M Base-Pretraining Release Foundation Audit — 2026-08-01

Status: pre-commit implementation/design evidence; non-authorizing.

## Evidence produced

- Full-loss base chunks retain source, revision, parent-document, document hash,
  transformation, chunk ID/index, split, text hash, tokens, and masks.
- Only the chunk's first token is unsupervised; later content and EOS are
  supervised. PAD and cross-chunk transitions remain masked after shifting.
- Parent documents cannot cross splits, exact text cannot repeat, source
  revisions cannot drift, and parent chunk indexes cannot collide.
- Stateful cross-split validation and a single-pass record generator preserve
  these guarantees without materializing the future corpus in memory.
- The accepted 513-token packer is reused without modifying its instruction
  contract or frozen specification.
- Deterministic fixture evidence covers two synthetic sources and all three
  splits using the real frozen tokenizer.
- The accompanying builder design is source-separated, streaming, two-pass,
  atomic, non-overwriting, authorization-gated, and training-inert.

## Identities

- Parent commit: `2d73fcb766c76dffb7ceb8b615d7162b79262176`
- Implementation SHA-256:
  `56dadc8d958ce2b0ca5c6bf4ccdc10f231502d3207b2e118734a944e0adfd6bb`
- Tests SHA-256:
  `ea119d2337d39724439bf2f8d4b62abdf697dbd86f37cfe5990eca420079e6b`
- Smoke SHA-256:
  `7358af5ca8ff672b5d7d253668b9511f53f59c415f95a9d4f250dc614a661962`
- Fixture file SHA-256:
  `04f31585ca8ea731c55980d4e0dc5898ebe7b320b5d7c088f14eb2d259a45266`
- Fixture report SHA-256:
  `7c4e4710cc70bb39124f8b1445516aac801401bc2c263175b6472dba125aa578`

## Validation

- Base-record plus frozen-record tests: 34 passed.
- Ruff on implementation, tests, and smoke: passed.
- Smoke JSON validation and exact frozen-fixture comparison: passed.
- No source discovery, network access, data acquisition, processed release,
  manifest, schedule, config, checkpoint, optimizer, or training action occurred.

## Remaining gates

The production builder, source-specific acquisition/preparation pipelines,
admission packages, release specification, real-corpus qualification, detached
authorization protocol, publication, and post-publication validation remain
unimplemented or unexecuted. Acceptance of this foundation authorizes only a
clean commit and later builder implementation review.
