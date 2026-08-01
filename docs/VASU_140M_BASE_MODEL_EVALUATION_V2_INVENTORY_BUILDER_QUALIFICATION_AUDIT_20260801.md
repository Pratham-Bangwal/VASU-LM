# VASU-140M Base Evaluation v2 Inventory Builder Qualification Audit

Date: 2026-08-01

Status: **author-side fixture qualification rebased to the corrected construction plan; independent implementation review pending.**

## Purpose

The inventory contract defines valid files, while the construction plan defines
scientific coverage. Neither package previously supplied a deterministic way to
turn reviewed authoring records into mutually bound payload, provenance,
contamination, and manifest artifacts. Implementing that logic directly in a
future production authoring script would create an unreviewed contamination and
lineage boundary.

This package therefore qualifies the builder with one synthetic fixture record
for each of the five prompt-based development dimensions. It deliberately has
no production mode and does not construct likelihood or held-out inventories.
The fixture report now binds the exact validated construction-plan identity;
fixtures cannot be reported against a substituted suite or repository commit.

## Implemented Behavior

`evaluation/framework/vasu_140m_base_v2_inventory_builder.py`:

- accepts only the five prompt dimensions: factuality, arithmetic, repetition,
  robustness, and manual review;
- accepts only development records carrying explicit `human_approved=true`;
- binds each record to a declared semantic family and parent document;
- constructs exact task inputs and payload content hashes for each dimension;
- constructs public provenance records with immutable revision and license
  evidence;
- constructs NFC, whitespace-collapsed, case-folded prompt/answer hashes with
  exact scan-span word counts, rolling eight-word fragment hashes, and a
  declared-family/content semantic commitment;
- enforces unique sorted IDs, prompt identities, semantic families, and parent
  documents inside the fixture inventory;
- hash-binds the scorer, payload, provenance, contamination, commitments, byte
  counts, and inventory identity;
- writes an isolated staging directory and promotes it without overwrite;
- rejects unsafe paths, path escapes, link/junction traversal, stale staging,
  existing output, scorer substitution, unapproved records, and tampered bytes;
  and
- validates every promoted bundle through the separately implemented inventory
  contract before returning its manifest.

The public API is named `build_fixture_inventory`; there is no argument or
alternate function that can set `fixture_only=false`. The fixture report keeps
production construction, held-out construction, key creation, model execution,
evaluation authorization, and training authorization false.

## Frozen Identities

- Construction-plan parent commit: `216013a9175a5cdff165765e60641ce5b1dae8c5`
- Construction-plan identity: `3a0b17aa613af4fd1bec4a76e12bbc545505a289719d394dc62e27e8e10b1494`
- Implementation SHA-256: `7f4654a071706421161c95f19b75012d9e14582dc28e817367628857ceb56c11`
- Test SHA-256: `67db4f2ec385825aaedd987b386c06213cd9a14d43d851aea5ba12421fc3874e`
- Smoke SHA-256: `b97a5b60c9964ffd836b8e2a6c2585342113e5e7ecd41dee2cf6e1ec9d966e3b`
- Fixture SHA-256: `636cc7655cb1260a779c0c3beeecfe867cbf25363765b0b8866488c0b2818758`
- Qualification report SHA-256: `2e046f7947ecd194c3c2bdefb19644aeb9768c1ec8cdbc4e8d884efc1cccd2e7`

The five synthetic inventory identities are frozen in
`evaluation/fixtures/vasu_140m_base_v2_inventory_builder_qualification_v1.json`.

## Validation

Focused tests:

```powershell
python -m pytest tests\test_vasu_140m_base_v2_inventory_builder.py -q
```

- 16 passed;
- 1 directory-symlink test skipped because the current unprivileged Windows
  environment did not permit creating the link;
- link and junction rejection was inspected directly in the implementation.

Broader evaluation/source regression:

```powershell
python -m pytest tests\test_vasu_140m_base_v2_inventory_builder.py tests\test_vasu_140m_base_v2_inventory.py tests\test_vasu_140m_base_v2_inventory_plan.py tests\test_vasu_140m_base_v2_scoring.py tests\test_vasu_140m_source_admission.py -q
```

- 97 passed;
- 1 skipped as described above.

Ruff passed for the implementation, tests, and smoke. The smoke reproduced the
frozen fixture exactly and removed every temporary fixture bundle.
`git diff --check` found no whitespace error and emitted only existing
LF-to-CRLF warnings for unrelated modified files.

## Dependency Order

Review and commit remain ineligible until:

1. evaluation-v2 scoring qualification is independently accepted and committed;
2. the inventory contract is rebased, independently accepted, and committed;
3. the construction plan is rebased and independently accepted; and
4. this package is rebased to those clean identities and requalified.

This audit records useful author-side evidence but does not bypass that order.

## Compatibility

The implementation is additive. It changes no model architecture, tokenizer,
dataset, mask, checkpoint, Trainer, optimizer, scheduler, existing benchmark,
or evaluation result. All existing artifacts remain compatible.

## Non-Authorization

Only five one-record synthetic fixture bundles were created in auto-removed
temporary directories. No production prompt inventory, likelihood holdout,
held-out plaintext/ciphertext, encryption key, source acquisition, dataset,
checkpoint, model evaluation, optimizer update, authorization record, or
training run was created or started.
