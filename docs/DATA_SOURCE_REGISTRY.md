# VASU Data Source Registry

## Purpose

The source registry is the provenance and approval layer for datasets proposed
in VASU data-mixture manifests. It records exactly which provider, dataset,
subset, split, revision, license, access route, local path plan, and quality
work applies to each mixture source ID.

This layer is metadata-only. Loading or validating the registry does not
download, tokenize, inspect, or train on dataset content.

## Approval states

The registry supports exactly four states:

| State | Meaning |
| --- | --- |
| `pending` | Metadata exists, but review or preparation planning is incomplete. |
| `approved` | The source is eligible for acquisition and preparation under the recorded conditions. |
| `rejected` | The source must not be used. |
| `blocked` | The source may become usable, but an access, license, revision, or risk issue prevents approval. |

An `approved` record requires known policy values, a reviewer, a timezone-aware
review timestamp, resolved approval notes, and both raw and processed local path
plans. Unknown boolean policy values are represented by JSON `null` and are
allowed only for `pending` or `blocked` sources.

Approval is not permission to bypass license obligations, attribution,
redistribution restrictions, quality filtering, deduplication, or contamination
review.

## Current registry

Registry files live under `configs/data/sources/`.

| Source ID | Dataset/subset | Status | Purpose |
| --- | --- | --- | --- |
| `fineweb_edu_original_train` | FineWeb-Edu `CC-MAIN-2013-20`, fixed local training region | approved | Existing control/general/educational source |
| `fineweb_edu_extension_2025_26` | FineWeb-Edu `CC-MAIN-2025-26`, validated extension | approved | Existing educational continuation source |
| `wikipedia_en_20231101_planned` | English Wikipedia `20231101.en` | pending | Planned factual pilot |
| `finemath_4plus_planned` | FineMath 4+ | blocked | Planned mathematics pilot; immutable revision and risk review required |
| `permissive_python_code_planned` | The Stack v2 permissive Python allowlist | blocked | Planned code pilot; access, per-file licensing, and redistribution review required |
| `vasu_verified_reasoning_v1_planned` | Locally generated verified reasoning v1 | pending | Planned reasoning pilot; generator and validation design incomplete |

The control pilot is currently registry-approved. The factual and capability
pilots are planning-valid but are not training-ready because they reference
pending or blocked records.

## File formats

A registry JSON file can contain either:

1. One source-record object; or
2. A source-family bundle with exactly one top-level `records` list.

The FineWeb file uses a bundle because its original and extension artifacts
have distinct source IDs, crawl configurations, revisions, local paths, and
content hashes. All fields are strict: missing and unknown fields are rejected.

## Python API

```python
from pathlib import Path

from vasu.data.mixtures import load_manifest
from vasu.data.sources import (
    get_source,
    load_source_record,
    load_source_registry,
    validate_manifest_sources,
)

registry = load_source_registry(Path("configs/data/sources"))
record = get_source(registry, "fineweb_edu_original_train")

manifest = load_manifest(Path("configs/data/vasu_60m_control_pilot.json"))

# Check IDs and provenance consistency while allowing planning records.
validate_manifest_sources(manifest, registry)

# Required readiness gate before acquisition, preparation, or training.
validate_manifest_sources(manifest, registry, require_approved=True)
```

`load_source_record()` intentionally loads only single-record files. Source
family bundles are loaded through `load_source_registry()`.

## Cross-manifest checks

For every mixture source, registry validation checks:

- exact source ID presence;
- dataset-card URL;
- pinned revision;
- license name;
- split;
- planned processed path;
- supported domain;
- commercial-use and attribution policy when known;
- content hash consistency when the mixture supplies a hash;
- approval state when `require_approved=True`.

This prevents a mixture from silently changing the dataset revision, license,
path, policy, or domain declared during source review.

## Filesystem behavior

The registry validates path plans as metadata. It does not require planned raw
or processed files to exist and does not resolve paths relative to a hidden
working directory. Physical file existence, size, hash verification, and
preparation outputs belong to a later explicitly authorized acquisition or
preparation workflow.

## Promotion workflow

Before changing a record to `approved`:

1. Pin an immutable dataset revision and intended subset/split.
2. Verify the applicable license and source-level exceptions.
3. Resolve commercial-use, attribution, redistribution, gating, and
   authentication policies.
4. Define raw and processed local paths.
5. Specify quality, deduplication, and contamination controls.
6. Record expected sizes and counts where available.
7. Review access and redistribution requirements.
8. Record the reviewer and timezone-aware review timestamp.
9. Remove unresolved markers from approval notes.
10. Add a SHA-256 content hash once a concrete prepared artifact exists.

No pending, blocked, or rejected source should pass the training-readiness gate.
