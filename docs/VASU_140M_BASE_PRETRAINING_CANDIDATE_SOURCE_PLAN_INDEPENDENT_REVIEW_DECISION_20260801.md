# VASU-140M Base-Pretraining Candidate Source Plan Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5 independent review

## Decision

Accept the VASU-140M candidate source plan for later source-specific admission
packages.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a pre-existing dirty author-side worktree:
  `docs/CHANGELOG.md`, `docs/PROJECT_STATUS.md`, and `docs/ROADMAP.md` were
  already modified, with several unrelated VASU-140M planning and
  implementation files untracked. This review modified only this decision file.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_DATA_RELEASE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_SOURCE_ADMISSION_SCHEMA_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN_REVIEW_PACKET.md`
- `docs/DATA_SOURCE_REGISTRY.md`
- `configs/data/sources/fineweb_edu.json`
- `configs/data/sources/wikimedia.json`
- `data/manifests/pretrain/fineweb_combined_coverage.json`
- `data/manifests/pretrain/fineweb_extension_document_index_metadata.json`
- `data/manifests/pretrain/fineweb_document_index_metadata.json`

## Rationale

The plan correctly rejects direct reuse of existing 257-token VASU-60M
FineWeb binaries as VASU-140M base-pretraining input. It also rejects the
original FineWeb 1M raw text for v1 admission because its document index shows
zero provider IDs, zero URLs, and incomplete provenance for all indexed
documents.

The recovered FineWeb-Edu extension is treated only as a candidate primary.
Repository evidence supports that candidacy: all 379,247 retained extension
source IDs were recovered, matched their historical text hashes, and were
indexed with complete source IDs and no duplicate hashes. The plan still
requires a new VASU-140M admission package, explicit web-rights review beyond
the ODC-By database license, renewed contamination review, and retokenization
from hash-verified document text rather than token-binary reuse.

Wikipedia is treated only as a bounded candidate secondary. The registry and
plan preserve its immutable revision, article-level provenance expectations,
CC-BY-SA/GFDL attribution, license-link, change-notice, ShareAlike, source
notice, fair-use exclusion, and bounded acquisition requirements.

The proposed 90/10 target remains conditional on measured eligible tokens
after source admission, filtering, contamination checks, cross-source
deduplication, and tokenizer measurement. Replacement sampling is explicitly
prohibited. Both sources require separate admission before acquisition or
release construction.

The plan keeps contamination scanning, cross-source deduplication,
document-level split isolation, 513-token full-loss packing, atomic
publication, and one-build authorization as separate later gates. It does not
create or approve a registry mutation, acquisition, data release, checkpoint,
schedule, configuration, optimizer state, or training run.

## Validation Results

- `python -m pytest tests\test_data_source_registry.py tests\test_vasu_140m_source_admission.py -q`
  passed: 45 tests passed.
- `git diff --check` passed with line-ending warnings only for pre-existing
  modified documentation files.
- `git status --short` showed pre-existing modified documentation files and
  unrelated untracked planning/implementation artifacts, plus this decision
  file after creation.
- Targeted path checks found no VASU-140M source-registry mutation file, no
  VASU-140M base-pretraining processed data or manifest, no VASU-140M
  training configuration, no schedule, and no `checkpoints/vasu_140m` path.

## Compatibility Conclusion

The candidate source plan is documentation-only and is compatible with the
existing source registry, tokenizer, 513-token VASU-140M record contract,
existing VASU-60M FineWeb binaries, existing manifests, checkpoints, schedules,
and evaluation artifacts. It does not relabel legacy artifacts or change
current consumers.

## Non-Authorization

This acceptance authorizes only later source-specific admission-package work.
It does not authorize source selection as final, registry mutation, source
discovery, acquisition, data processing, data construction, publication,
configuration creation, schedule creation, checkpoint creation, optimizer
creation, authorization-record creation, base pretraining, instruction tuning,
training, commit, or push.
