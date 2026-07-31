# VASU-140M Base-Model Evaluation v2 Schema Implementation Audit — 2026-08-01

Status: pre-commit implementation evidence; non-authorizing.

## Scope completed

- Added strict, exact-field suite and result manifest validators.
- Bound `vasu_140m_v1`, its frozen configuration fingerprint, and the frozen
  tokenizer SHA-256.
- Required development and sealed held-out inventories for likelihood,
  factuality, arithmetic, repetition, robustness, and manual review.
- Separated direct-likelihood and raw/paired-continuation interfaces.
- Bound greedy and sampled generation profiles, bootstrap settings,
  contamination ordering, scorers, repository commit, checkpoint, runtime,
  output shards, per-dimension metrics, and resume identity.
- Rejected aggregate capability scores, unknown fields, unsafe paths,
  incomplete dimension coverage, early held-out access, non-finite metrics,
  identity mutation, wrong-family checkpoints, and training authorization.
- Added prompt-free synthetic builders, 16 adversarial tests, a read-only
  smoke, and frozen pre-commit qualification evidence.

## Frozen evidence

- Parent commit: `2d73fcb766c76dffb7ceb8b615d7162b79262176`
- Module SHA-256:
  `994f8087365489548b4c8d739fcfe6eafe41efac2484c64eb6c5373828bab24a`
- Test SHA-256:
  `afa66469fe9284d1e5350b1531a29e9ceda1327d31a6114d5863604ae2c55b9f`
- Smoke SHA-256:
  `e675ab631742377f1c457dd73ffb22ca98e5b34ecd3497c98e06c998b1ae5fb0`
- Frozen fixture file SHA-256:
  `443ee3a5decdf2c491a609613d9851ebb031c67d412ca53baf76f0f02e20439e`
- Qualification SHA-256:
  `8870976dc9f7c76e5da3eacdb374eedc1a4b1a2fa74af0d41789c4ff2be38b14`

The smoke reproduced the frozen fixture exactly. It verified development-file
bindings but did not open held-out files, freeze real prompts, invoke a model,
open a checkpoint, publish results, construct data, or authorize training.

## Remaining gates

This package validates manifest identities and structure. Dimension-specific
inventory record schemas, real prompt provenance, scorer implementations,
contamination qualification, and frozen suite identities remain separate
review-gated work. After implementation acceptance and commit, the schema
qualification must be rerun against the clean final commit and identity-reviewed.
