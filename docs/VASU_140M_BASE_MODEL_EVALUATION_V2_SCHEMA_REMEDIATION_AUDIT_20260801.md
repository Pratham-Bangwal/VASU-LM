# VASU-140M Base-Model Evaluation v2 Schema Remediation Audit — 2026-08-01

Status: remediated pre-commit evidence; non-authorizing.

## Prior review history preserved

GPT-5.5 initially rejected the first schema package because its frozen
qualification was bound to commit `2d73fcb` while review ran at `2e5a492`.
The exact rejected fixture is preserved as
`evaluation/fixtures/vasu_140m_base_evaluation_v2_schema_qualification_rejected_2d73fcb.json`.
The implementation decision path later recorded acceptance of the prior
`2e5a492` package. It is not rewritten or treated as acceptance of the current
`51943a7` fixture. The current identity therefore receives a new remediation
decision path.

## Additional defect found before re-review

An adversarial probe demonstrated that the first result validator accepted a
likelihood shard labeled as sampled generation and held-out even though no
held-out opening decision was bound. Result mode and split were validated only
against global enumerations, not the dimension or evaluation stage.

## Remediation

- Results now declare one exact `evaluation_stage`.
- Development results must carry a canonical empty held-out authorization.
- Held-out results require a decision ID plus path and SHA-256 binding.
- Every output shard and dimension report must match the declared stage.
- Direct-likelihood, greedy, sampled, and manual modes are restricted to their
  exact dimensions.
- Arithmetic, repetition, and robustness results require separate greedy and
  sampled coverage.
- Dimension reports now bind split and mode, and their coverage must match the
  output shards.
- `validate_result_against_suite` requires the exact suite and tokenizer
  identities.
- Resume identity now includes evaluation stage and held-out decision identity.

## Current identities

| Artifact | SHA-256 |
| --- | --- |
| Implementation | `b3d889168196b3c880eb99f9868f14105d6b4b3b5cc84010add96ffb77997c49` |
| Tests | `313126c35430372832a9f19f463d15e01a3b054ad0a7f6c15027cf7813174319` |
| Smoke | `e675ab631742377f1c457dd73ffb22ca98e5b34ecd3497c98e06c998b1ae5fb0` |
| Current fixture | `4080eddbfe2d5c9355ee8de438a5594053e6b59b990a1d7196534eccfed908deb` |

- Parent commit: `51943a7a326b836755b2edaa7b7fd26ba0da366d`
- Qualification SHA-256:
  `d82bb1408d5b28ff9599a535c4a94c6b0a9c8e6db638bc03be48556787e178d8`

## Validation

- 20 focused schema tests passed.
- 67 schema and existing evaluation-framework tests passed together.
- Ruff passed for implementation, tests, and smoke.
- The current smoke reproduced the remediated fixture exactly.
- Held-out files remained unopened; no real prompt, scorer result, model,
  checkpoint, production result, data artifact, or training action was created.

## Compatibility

This is a pre-commit schema package, so the stricter result shape supersedes
only synthetic uncommitted result examples. Existing evaluation files,
checkpoints, tokenizer assets, datasets, masks, training code, and exact-resume
state remain unchanged.

## Non-authorization

This remediation does not freeze prompts, open held-out data, run a model,
open or create a checkpoint, publish a result, construct data, create a
configuration or schedule, create optimizer state, authorize training, or
train.
