# VASU-140M Development-Inventory Promotion Execution Independent Review Decision

Status: accepted; non-authorizing.

Reviewer: GPT-5.5 independent review

Review date: 2026-08-04

Result commit reviewed: `8088172e4dfefa80cc4ff2d17dce78b01e369e4c`

Execution commit: `b6a6f55d2c1d3b99cd2ea6f4c5e56869fd03852d`

## Decision

Accept.

The completed one-shot development-inventory promotion is accepted as a
byte-preserving, non-authorizing promotion of the five public development
inventories. These five promoted inventories may now be used as development
inventory evidence in separately reviewed source-admission package
construction.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

None.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_DEVELOPMENT_INVENTORY_PROMOTION_EXECUTION_AUDIT_20260804.md`
- `docs/VASU_140M_DEVELOPMENT_INVENTORY_PROMOTION_EXECUTION_PACKAGE_INDEPENDENT_REVIEW_DECISION_20260804.md`
- `evaluation/candidates/vasu_140m_base_v2_development_v1/promotion_receipt.json`
- All manifests and bound files under
  `evaluation/candidates/vasu_140m_base_v2_development_v1`
- The original source bundles under
  `evaluation/fixtures/vasu_140m_assistant_authored_internal_v1`
- `evaluation/framework/vasu_140m_development_promotion.py`
- `evaluation/framework/vasu_140m_base_v2_inventory.py`

## Exact Identities and Counts

Receipt SHA-256:
`3852c57e4704efc213cc0fe34a86a58f37fa60bc43a21dae3b3c6bc9aea85818`.

| Dimension | Records | Source inventory SHA-256 | Promoted inventory SHA-256 |
| --- | ---: | --- | --- |
| arithmetic | 1000 | `fa9855f439220e811f5cf5de5978eeae358f8e6eeb3cecc33f61315c880c4eaf` | `fd6f75ebed75ee0fcb2bfc36fde29a1057a2e67a162eec7438d0ef727d0266bf` |
| factuality | 200 | `eb84e4904ed2da8f8309bbb9df4d557490a6dc6f3c1c75ebae72a1f0749a9045` | `958257c55a0fdd4782470324cc5d1128d70ae5ef9dbf2d029c67f4e88fdb4008` |
| manual_review | 60 | `955df5ef3e306b975756f076fe9572b0433127d993ca40cb4b4c7303d6e39bad` | `ee214389a90b189432ad5bb10bf14903c62cc22226fa97b646a23a31174e3cea` |
| repetition | 120 | `68ca327e0fe5b7d8e60420d5723478833ae77c50fea908f44121aa47691b629f` | `a9acb73757acf8321e15dc0b4f1eb0936e5c983976b125d39cbc6cf419676a03` |
| robustness | 120 | `fc37fbc37e2019025406af8e6bdd3bb5d8748a904031e7c68b68bef13cdf75a2` | `48d4396b1883af7fa345ce445ea2d0e726126ab625add9dc55e417a2da54460c` |

All promoted manifests bind suite
`vasu-140m-base-evaluation-v2-development-candidate-v1`, inventory IDs of the
form `vasu-140m-base-evaluation-v2-development-candidate-v1-<dimension>`, the
execution commit `b6a6f55d2c1d3b99cd2ea6f4c5e56869fd03852d`, and final
repository-relative candidate paths.

## Byte Preservation

Payload, provenance, and contamination files for all five promoted dimensions
are byte-identical to the corresponding accepted fixture source files. Existing
fixture files and source inventory identities remain unchanged.

## Manifest and Receipt Validation

Every final-path promoted manifest passed `validate_inventory_manifest_files`.
Each promoted manifest has:

- `split=development`
- `fixture_only=false`
- `production_suite_frozen=false`
- `evaluation_run_authorized=false`
- `training_authorized=false`

The receipt binds all five source inventory identities and all five promoted
inventory identities. Its canonical self-hash reproduces the expected receipt
SHA-256. Exactly five promoted dimension directories and one receipt exist under
the output root. No staging directory remains.

## Validation Results

- `git rev-parse HEAD`
  - `8088172e4dfefa80cc4ff2d17dce78b01e369e4c`
- Direct manifest, receipt, source/promoted byte comparison, identity, count,
  and staging validation
  - Passed.
- `python -m pytest tests\test_vasu_140m_development_promotion.py tests\test_vasu_140m_base_v2_inventory.py tests\test_vasu_140m_source_admission_v4.py -q`
  - `38 passed in 1.83s`
- `python -m ruff check evaluation\framework\vasu_140m_development_promotion.py scripts\promote_vasu_140m_development_inventories.py tests\test_vasu_140m_development_promotion.py`
  - `All checks passed!`
- `git diff --check`
  - Passed; Git reported existing CRLF normalization warnings for
    `docs/CHANGELOG.md`, `docs/PROJECT_STATUS.md`, and `docs/ROADMAP.md`.
- `git status --short`
  - Existing unrelated dirty documentation files were present before review and
    preserved.

## Compatibility

The promotion is additive. It does not alter fixture bundles, sealed held-out
payloads, tokenizer identity, scoring contracts, datasets, masks, checkpoints,
model architecture, exact resume, source-admission validators, optimizer or
scheduler state, or training state.

## Non-Authorization

No retry or second promotion was found. No held-out payload or key was accessed.
No source was admitted. No likelihood inventory was constructed. No evaluation
suite was frozen or run. No data release, checkpoint access, checkpoint
creation, configuration, schedule, optimizer state, authorization record, or
training action was authorized or performed.
