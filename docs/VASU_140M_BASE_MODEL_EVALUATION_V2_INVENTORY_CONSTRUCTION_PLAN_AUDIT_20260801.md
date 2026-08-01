# VASU-140M Base Evaluation v2 Inventory Construction Plan Audit

Date: 2026-08-01

Status: **author-side plan evidence; independent review pending; construction is not authorized.**

## Purpose

Freeze the scientific construction decision before any production prompt or
held-out key is created. The plan separates ten externally authored prompt
inventories from likelihood holdouts that can only be sampled from admitted
source documents after acquisition.

## Frozen Coverage

| Dimension | Development | Held-out | Scope |
| --- | ---: | ---: | --- |
| Likelihood | 512 | 512 | per admitted source |
| Factuality | 200 | 200 | total |
| Arithmetic | 1,000 | 1,000 | total |
| Repetition | 120 | 120 | total |
| Robustness | 120 pairs | 120 pairs | total |
| Manual review | 60 | 60 | total |

Likelihood uses whole-document isolation before 513-token packing. With the
currently proposed two-source strategy, the eventual release must therefore
reserve 1,024 development and 1,024 held-out likelihood records in total,
while still reporting each source independently.

The other five dimensions produce ten prompt inventories before source
admission. Semantic families, parent documents, and derived variants cannot
cross development/held-out boundaries.

## Source Roles

- `factual_cpt_v2.json` is a development seed and fact-family audit source.
  It cannot derive held-out prompts.
- `arithmetic_v2.py` is the deterministic verified generator. Development and
  held-out construction must use disjoint template families and operand ranges.
- `ultrachat_promotion_v1.json` supplies topic/stratum ideas only. Its
  instruction prompts cannot be copied into the base-model suite or used to
  derive held-out prompts.

All production items require public provenance, exact contamination
commitments, eight-word fragment commitments, and semantic review.

## Accepted Dependency Bindings

The plan now hash-binds all four independent decisions that establish its
scoring and inventory foundations: scoring qualification, scoring post-commit
identity, inventory-contract qualification, and inventory post-commit
identity. File validation opens and hashes each exact decision document. A
missing, substituted, renamed, or mutated decision therefore invalidates the
plan before construction review.

## Held-Out Boundary

The plan selects Age X25519 but deliberately binds no recipient fingerprint.
Key creation, held-out construction, and opening all remain unauthorized.
Plaintext held-out content is forbidden inside the repository, and an
independent curator remains required.

## Identities

- Parent commit: `216013a9175a5cdff165765e60641ce5b1dae8c5`
- Plan identity: `4bd36c7d7b862e782614347f9c837e3fbaf12c7eaff0d80b7df146c6d71db784`
- Plan file SHA-256: `9efd4b18c4b606d1d9d10a0028ac7f42c1d111522350951fffb54a0d926f6662`
- Validator SHA-256: `c95c883ab318f5ad79e7208039ad37e03860b082a928471554cac229d12139b6`
- Tests SHA-256: `a136f45319d49baf32efdab9b56e9880923fb0f8581421fada033fb9993cb888`
- Smoke SHA-256: `6052d498d69424086adf4c06f4980add83a79bff5f82c5c405b426e518716b9f`
- Fixture SHA-256: `a8ea22d607421a2d9fb9434523ba32243664dae9fc125bdc9ae6538aea038190`

## Validation

`python -m pytest tests\test_vasu_140m_base_v2_inventory_plan.py -q`

- 18 passed.

Ruff passed for the validator, tests, and smoke. The smoke reproduced
`evaluation/fixtures/vasu_140m_base_v2_inventory_plan_qualification_v1.json`
exactly.

## Dependency Order

The plan hash-binds the accepted scoring and inventory-contract implementations
plus all four qualification and post-commit decision documents. Those
dependencies are complete. Independent review of this plan remains required
before any construction package can rely on it.

## Compatibility and Non-Authorization

This package changes no model, tokenizer, dataset, mask, checkpoint, trainer,
or existing evaluation result. It creates no prompt, held-out ciphertext/key,
data release, model result, checkpoint, optimizer state, or training path.
`construction_authorized`, `evaluation_run_authorized`, and
`training_authorized` remain false.
