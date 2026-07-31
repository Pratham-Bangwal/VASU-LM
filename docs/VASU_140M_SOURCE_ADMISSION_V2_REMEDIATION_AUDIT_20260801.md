# VASU-140M Source-Admission v2 Remediation Audit — 2026-08-01

## Outcome

The accepted v1 source-admission validator contained two contract defects that
became concrete when the accepted candidate source plan was mapped to the
repository registry. The versioned v2 proposal corrects both defects without
creating or approving a source admission package.

## Root cause

First, v1 required `decision.state` to equal the generic source record's
`approval_status`. The bundled FineWeb-Edu extension and planned Wikipedia
records are generically approved for their earlier bounded uses, but the new
VASU-140M legal, contamination, acquisition, and release-specific decisions
must remain pending. Conflating those states would either reject an honest
pending package or falsely inherit generic approval as VASU-140M approval.

Second, repository-bound validation used the single-record loader. The
FineWeb-Edu extension is correctly stored inside the multi-record
`configs/data/sources/fineweb_edu.json` bundle, so v1 could not bind that exact
candidate even though the generic registry accepted it.

## Remediation

- Version the contract as `vasu_140m_base_source_admission_v2`; v1 packages
  fail closed and require explicit regeneration.
- Rename the bound field to `registry_approval_status` so its generic scope is
  unambiguous.
- Keep the VASU-140M `decision.state` independent. Pending, blocked, and
  rejected VASU-140M decisions can bind a generically approved record.
- Require generic registry approval as a prerequisite for VASU-140M approval;
  generic approval is never sufficient by itself.
- Expose a read-only `load_source_records` helper that validates either a
  single-record file or a bundle, then require the exact source ID to resolve
  exactly once.
- Add regression coverage for v1 rejection, state separation, invalid
  candidate approval over a non-approved generic record, and the real bundled
  FineWeb-Edu extension record.

## Exact implementation identities

| Artifact | SHA-256 |
| --- | --- |
| `vasu/data/vasu_140m_source_admission.py` | `0dd31243c60e0402eba7ccda143b92c96f20a0ed7ea3ac0e1574a96cdc72cb55` |
| `vasu/data/sources/registry.py` | `325b848bfd0c43e775cce34be9752c018832c5454cc9023a29dc2475a66f65ae` |
| `vasu/data/sources/__init__.py` | `0048c9e96679a15083879a381742c64438fe767fd562654273cb3159c2400b3a` |
| `tests/test_vasu_140m_source_admission.py` | `f25611c00955dbfb11325d61935e037232f559471c7053232c87097f0e9697e6` |

## Validation

- 65 source-admission, generic-registry, and 513-token record tests passed.
- Ruff passed for every changed Python file.
- Repository-bound validation resolved
  `fineweb_edu_extension_2025_26` exactly once from its real bundled file.
- No registry JSON, external source bytes, processed data, manifest,
  checkpoint, schedule, configuration, optimizer, or authorization artifact
  was created or modified.

## Compatibility

The generic registry file schema is unchanged. The new public loader exposes
existing validated bundle behavior without changing registry construction.
Existing datasets, masks, tokenizer assets, checkpoints, schedules, and
resume state remain byte-compatible. Only uninstantiated VASU-140M v1
admission packages are intentionally incompatible; none exist in the
repository, and v2 fails closed on the old schema ID.

## Non-authorization

This remediation does not authorize either candidate source, registry
mutation, discovery, acquisition, processing, release construction,
publication, configuration, schedule, checkpoint, optimizer, authorization
record, base pretraining, instruction tuning, or training.
