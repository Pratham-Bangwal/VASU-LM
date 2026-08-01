# VASU-140M Source-Admission v3 Hardening Audit

Date: 2026-08-01

Status: **author-side evidence validated against the committed contamination scanner; both candidate sources remain blocked.**

## Root Cause

The accepted v2 admission package hash-bound an entire generic registry file,
but its file validator cross-checked only source ID and approval status. The
package's declared license, policy booleans, immutable revision, and access
method could therefore disagree with the selected record while still passing.

The evaluation-isolation list also accepted one arbitrary path/hash pair. An
otherwise approved package was not required to bind all externally authored
prompt dimensions across development and held-out splits, and the bound file
was not required to be a non-fixture inventory manifest.

These are semantic identity gaps at a production-data admission boundary, not
formatting issues.

## v3 Remediation

`vasu_140m_base_source_admission_v3` now:

- records the five generic-registry policy booleans explicitly;
- compares license name/URL, every policy boolean, immutable revision, and
  access method against the exact hash-bound source record;
- requires dimension, split, and inventory ID for every evaluation binding;
- permits an incomplete matrix only while the source remains pending, blocked,
  or rejected;
- requires exactly the 10 unique combinations of five prompt dimensions and
  two splits before VASU-140M approval;
- separately requires likelihood development/held-out documents to be created
  after acquisition, isolated at document level, excluded from training, and
  bound into the later release manifest;
- opens every approved inventory as JSON, checks its ID/dimension/split, and
  requires `fixture_only=false`; and
- retains `acquisition.authorized=false` and `training_authorized=false`.

v1 and v2 packages fail closed. No VASU-140M source-specific admission package
has been approved, so the version bump invalidates no production admission.

## Candidate Evidence

The smoke resolves both candidates exactly once from their real registry files
and constructs in-memory blocked dossiers:

- `fineweb_edu_extension_2025_26` package SHA-256:
  `eadac665b718fe5603e3c7b00c9d2016cbccc761972cb8311f6e5e2c8828636c`;
- `wikipedia_en_20231101_planned` package SHA-256:
  `9a25f3600d1215885e122054009b09038bf5df4bf75d6319d5b2a6e7453420ba`.

Both remain blocked because the complete production-candidate evaluation
matrix does not exist. FineWeb-Edu also retains unresolved underlying web
content-rights and Common Crawl terms review. Wikipedia retains unresolved
imported/fair-use exclusion and operational attribution/ShareAlike review.

Primary-source checks used for this audit:

- FineWeb-Edu dataset card: `https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu`
- ODC-By 1.0: `https://opendatacommons.org/licenses/by/1-0/`
- Common Crawl terms: `https://commoncrawl.org/terms-of-use`
- Wikimedia Wikipedia dataset card: `https://huggingface.co/datasets/wikimedia/wikipedia`
- CC BY-SA 3.0: `https://creativecommons.org/licenses/by-sa/3.0/`
- GNU license index/GFDL: `https://www.gnu.org/licenses/`

The ODC-By text explicitly distinguishes database rights from independent
rights in database contents. Common Crawl's terms state that crawled content
may be subject to third-party terms and rights. These findings justify a
blocked rather than approved FineWeb-Edu dossier. This is engineering
provenance analysis, not legal advice.

## Frozen Identities

- Committed contamination-scanner anchor: `908a3aaa9bdae18688fa26e0966e4d84c5944301`
- Implementation SHA-256: `00a36e79658d00d7c79568981814ac44e0800e3dbc540f9e40e9440e1d594855`
- Tests SHA-256: `2ef1133bd9eea2bda8af2c4de2d0c034c3f064af78bfd6d72e4250d8a66305c4`
- Smoke SHA-256: `25cb5fedd5f58f99ab5723ca00a6c8700b300290d6c8105c042f52c304bb2fc4`
- Fixture SHA-256: `e4fbc751cced77473cd40bd297c9a0a04642ae80046f7fbe7b05167fd5767479`

## Validation

`python -m pytest tests\test_vasu_140m_source_admission.py tests\test_data_source_registry.py -q`

- 61 passed.

`python -m ruff check vasu\data\vasu_140m_source_admission.py tests\test_vasu_140m_source_admission.py scripts\smoke_vasu_140m_source_admission_v3.py`

- passed.

The smoke reproduced
`evaluation/fixtures/vasu_140m_source_admission_v3_qualification_v1.json`
exactly. It downloaded no bytes and created no data release.

## Compatibility

Model architecture, tokenizer assets, record packing, existing datasets,
masks, checkpoints, optimizer state, schedules, and trainer behavior are
unchanged. Only unapproved VASU-140M admission-package drafts using v1/v2 need
regeneration under v3.

## Non-Authorization

This remediation does not approve either source. It does not authorize
acquisition, registry mutation, processing, release construction, schedule or
configuration creation, checkpoint creation, optimizer updates, or training.
No such action occurred.
