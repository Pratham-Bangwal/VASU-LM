# VASU-140M Base-Release Qualification Core Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the VASU-140M base-release qualification core for an isolated commit
and later clean post-commit identity review.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree with unrelated
  modified documentation and untracked packages. This was not treated as a
  blocker because the qualification core's identities, tests, and frozen
  fixture reproduced exactly and the reviewed package remained internally
  non-authorizing.
- `git diff --check` reported CRLF conversion warnings for pre-existing
  modified documentation files only; it reported no whitespace errors.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_BUILDER_DESIGN.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_FOUNDATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_PRETRAINING_RELEASE_FOUNDATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/VASU_140M_BASE_RELEASE_QUALIFICATION_CORE_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_RELEASE_QUALIFICATION_CORE_REVIEW_PACKET.md`
- `vasu/data/vasu_140m_base_records.py`
- `vasu/data/vasu_140m_base_release.py`
- `tests/test_vasu_140m_base_release.py`
- `scripts/smoke_vasu_140m_base_release_qualification.py`
- `evaluation/fixtures/vasu_140m_base_release_qualification_fixture_v1.json`

## Rationale

The qualification core validates exact, self-hashed source-evidence envelopes
before reading source streams, including admission, acquisition receipt, raw
inventory, normalization manifest, quarantine decision, deduplication evidence,
split assignment, and selection index. It recursively hash-checks the bound
subject and decision files, enforces fixture-versus-production scope, requires
publication and training authority to remain false, and rejects existing
protected production release and manifest paths.

The builder streams source-separated `uint16[513]` token files, `uint8[513]`
mask files, and JSONL lineage files without corpus-wide packed-record
materialization. Serialized artifacts are fully audited for token/mask length,
record width, binary masks, PAD tails, EOS endings, shifted target boundaries,
complete lineage, artifact hashes, internal-manifest identity, external
manifest-template identity, and unbound-file absence.

The global stream validator rejects duplicate chunks, duplicate exact text,
parent-document split leakage, source-revision drift, and duplicate parent
chunk indexes. Two independent scratch builds are required to be byte-identical.
Disk reservation, stale scratch directories, symlink/junction traversal,
fsync failure preservation, post-validation mutation detection, and
nondeterminism all have fail-closed coverage in the tests.

The frozen fixture reproduced exactly with qualification SHA-256
`dc54bf95522018ec46b5541247280d5784ebce4587179b07822f604207af0200`. The
smoke remains fixture-only and reports `production_release_created=false`,
`publication_authorized=false`, and `training_authorized=false`.

## Commands and Exact Results

- `git rev-parse HEAD`
  returned `561666f06d81bf2710098c3b84220cdda6026650`.
- `git status --short`
  showed concurrent author-side modified and untracked files, including this
  package's untracked implementation, test, smoke, audit, packet, and fixture.
- `python -m pytest tests\test_vasu_140m_base_release.py tests\test_vasu_140m_base_records.py tests\test_vasu_140m_records.py tests\test_vasu_140m_release_plan.py -q`
  passed: `60 passed in 64.09s`.
- `python -m ruff check vasu\data\vasu_140m_base_release.py tests\test_vasu_140m_base_release.py scripts\smoke_vasu_140m_base_release_qualification.py`
  passed: `All checks passed!`.
- Frozen fixture comparison:
  `$observed = (python scripts\smoke_vasu_140m_base_release_qualification.py | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)`;
  `$expected = (Get-Content evaluation\fixtures\vasu_140m_base_release_qualification_fixture_v1.json -Raw | ConvertFrom-Json | ConvertTo-Json -Depth 100 -Compress)`;
  `if ($observed -ne $expected) { throw "base-release qualification fixture mismatch" }`
  passed: `qualification fixture matched`.
- `git diff --check`
  completed with CRLF conversion warnings only for pre-existing modified
  documentation files and no whitespace errors.

## Identity Comparison

- Parent commit: expected and observed
  `561666f06d81bf2710098c3b84220cdda6026650`.
- Implementation SHA-256: expected and observed
  `5d90b394231d509741bcaabd38b935b0cb909615b98b24ca5268b20521952ecf`.
- Test SHA-256: expected and observed
  `1dc16b1617b8a502bc3e84e7352ea801a81afd1bc723d8a4a69298d1ad0a5dd4`.
- Smoke SHA-256: expected and observed
  `958cc9435df632a1dfbe8c5ab16b417480f2f5d70f762672d4b8612f77f5111e`.
- Fixture SHA-256: expected and observed
  `da0eb48b0d6bb75632b92df724b6703007f310b1ca5468e1a5844c0ecb3fc915`.
- Qualification SHA-256: expected and observed
  `dc54bf95522018ec46b5541247280d5784ebce4587179b07822f604207af0200`.
- Expected internal manifest SHA-256:
  `469dc5e8045ae521f947b75ba94ec6f0c4ca753318e8bade8195b59a12c87643`.
- Expected external manifest template SHA-256:
  `b5ada0b694a0dcc6f45b0cbd69b03f930cda5785abdebbebb5eea9b96c65f8dd`.

## Compatibility Conclusion

The package is additive. It does not modify existing VASU-31M/60M datasets,
the published VASU-140M instruction release, tokenizer assets, model
families, checkpoint formats, schedules, training configurations, optimizer
state, or exact-resume behavior. It is fixture-qualified evidence for a later
production release-builder path, not production data qualification itself.

## Non-Authorization

This acceptance authorizes only an isolated commit and a later clean
post-commit identity review. It does not authorize source admission, source
selection, source acquisition, production data construction, production
publication, schedules, configurations, optimizers, checkpoints,
authorization records, training, commit, or push.
