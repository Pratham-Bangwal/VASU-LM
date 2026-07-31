# VASU-140M Source-Admission v2 Remediation Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the VASU-140M source-admission v2 remediation for an isolated
remediation commit and later separately reviewed source-specific admission
packages.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree with unrelated
  modified and untracked VASU-140M packages. This was not treated as a blocker
  because the remediation is internally consistent, non-authorizing, and the
  requested focused tests pass.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_SOURCE_ADMISSION_DESIGN.md`
- `docs/VASU_140M_SOURCE_ADMISSION_SCHEMA_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN.md`
- `docs/VASU_140M_BASE_PRETRAINING_CANDIDATE_SOURCE_PLAN_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V2_REMEDIATION_AUDIT_20260801.md`
- `docs/VASU_140M_SOURCE_ADMISSION_V2_REMEDIATION_REVIEW_PACKET.md`
- `vasu/data/vasu_140m_source_admission.py`
- `vasu/data/sources/__init__.py`
- `vasu/data/sources/registry.py`
- `tests/test_vasu_140m_source_admission.py`
- `tests/test_data_source_registry.py`
- `configs/data/sources/fineweb_edu.json`
- `configs/data/sources/wikimedia.json`

## Rationale

The v1 defects are genuine. The accepted design explicitly required generic
registry approval and VASU-140M admission state to remain separate, but v1
forced the VASU-140M `decision.state` to equal the generic source
`approval_status`. That would either reject an honest pending VASU-140M
package for a generically approved source, or encourage accidental inheritance
of generic approval as model/release-specific approval.

The bundled-record defect is also genuine. `configs/data/sources/fineweb_edu.json`
is a valid strict registry bundle containing both `fineweb_edu_original_train`
and `fineweb_edu_extension_2025_26`; the old single-record loader rejects that
bundle by design. A VASU-140M package must be able to bind one exact record in
that valid multi-record registry file.

The v2 schema resolves the state issue by using
`source_record.registry_approval_status` for generic registry state while
leaving `decision.state` as the VASU-140M admission decision. Pending,
blocked, and rejected VASU-140M decisions can bind a generically approved
record. A VASU-140M approval still requires the generic source state to be
`approved`, reviewer identity, timezone-aware review time, zero unresolved
legal items, and every model/release-specific legal, acquisition, lineage,
quality, evaluation-isolation, deduplication, and authorization-false gate.

Legacy v1 packages fail closed because the accepted schema ID is now
`vasu_140m_base_source_admission_v2`. The repository-bound validator uses the
new public `load_source_records` helper, validates either a single-record file
or a bundle with the same strict record validation, and then requires the
requested source ID to resolve exactly once. Missing, ambiguous, or duplicate
resolution fails closed through that exact-count check and generic registry
duplicate-ID validation.

The real bundled `fineweb_edu_extension_2025_26` record is hash-bound through
`configs/data/sources/fineweb_edu.json` SHA-256
`d72ffec39d8d0350291ab7c88dda3b01a621e5cfa09b4582a211ca1a122bdaf7`.
Read-only inspection resolved the bundle IDs as
`fineweb_edu_original_train` and `fineweb_edu_extension_2025_26`, with exactly
one extension match and generic approval status `approved`.

No source-specific admission package, registry mutation, acquisition,
processed data release, configuration, schedule, checkpoint, optimizer,
authorization, or training route was introduced.

## Commands and Exact Results

- `python -m pytest tests\test_vasu_140m_source_admission.py tests\test_data_source_registry.py tests\test_vasu_140m_records.py -q`
  passed: 65 passed in 0.88s.
- `python -m ruff check vasu\data\vasu_140m_source_admission.py vasu\data\sources\registry.py vasu\data\sources\__init__.py tests\test_vasu_140m_source_admission.py`
  passed: all checks passed.
- `git diff --check`
  completed with CRLF line-ending warnings only for existing modified files.
- `git status --short`
  showed the known concurrent author-side modified and untracked packages,
  with no commit or push performed.

Additional read-only checks:

- Current HEAD:
  `51943a7a326b836755b2edaa7b7fd26ba0da366d`
- Implementation SHA-256:
  `0dd31243c60e0402eba7ccda143b92c96f20a0ed7ea3ac0e1574a96cdc72cb55`
- Registry loader SHA-256:
  `325b848bfd0c43e775cce34be9752c018832c5454cc9023a29dc2475a66f65ae`
- Source API SHA-256:
  `0048c9e96679a15083879a381742c64438fe767fd562654273cb3159c2400b3a`
- Source-admission test SHA-256:
  `f25611c00955dbfb11325d61935e037232f559471c7053232c87097f0e9697e6`
- FineWeb-Edu registry bundle SHA-256:
  `d72ffec39d8d0350291ab7c88dda3b01a621e5cfa09b4582a211ca1a122bdaf7`
- Wikimedia registry record SHA-256:
  `c1a81bff01499f1a843a7fb9f966bc2d9e13637d8d6afac3f8ecd9d3478a34a1`
- Checked absent:
  `data/raw/vasu_140m/base_pretraining`,
  `data/processed/vasu_140m/base_pretraining`,
  `data/manifests/vasu_140m/base_pretraining`,
  `configs/training/vasu_140m_base_pretraining.json`,
  `configs/schedules/vasu_140m_base_pretraining.json`,
  `checkpoints/vasu_140m/base_pretraining`,
  `authorizations/vasu_140m_source_admission`, and
  `authorizations/vasu_140m_base_pretraining`.

## Compatibility Conclusion

The remediation preserves the generic registry schema and strict validation
behavior. It only exposes validated multi-record loading as a public helper and
updates the VASU-140M admission contract to v2. Existing registry records,
existing source consumers, datasets, masks, tokenizer assets, checkpoints,
schedules, configurations, optimizer state, and exact-resume behavior remain
compatible. Uninstantiated v1 VASU-140M admission packages are intentionally
incompatible and fail closed.

## Non-Authorization

This acceptance authorizes only committing the v2 remediation and later
preparing separate pending source-specific admission packages. It does not
authorize source selection as final, registry mutation, source approval,
source discovery, acquisition, data construction, release publication,
configuration creation, schedule creation, checkpoint creation, optimizer
state creation, authorization-record creation, base pretraining, instruction
tuning, training, commit, push, or any source-specific approval.
