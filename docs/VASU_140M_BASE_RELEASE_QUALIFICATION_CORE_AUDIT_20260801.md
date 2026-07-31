# VASU-140M Base-Release Qualification Core Audit — 2026-08-01

Status: pre-commit fixture qualification evidence; non-authorizing.

## Root cause

The accepted base-record foundation proved full-loss 513-token packing but did
not construct source-separated files, reconcile complete evidence, perform
independent scratch rebuilds, or audit serialized release bytes. Consequently,
it could not yet support a reproducible production data-release gate.

## Implementation

`vasu/data/vasu_140m_base_release.py` adds a strict qualification core with:

- immutable family, model, tokenizer, record, builder, repository, release,
  source-order, evaluation-inventory, and output-path identities;
- a self-hashed source-evidence envelope for admission, completed acquisition,
  raw inventory, normalization, quarantine, deduplication, split assignment,
  and no-replacement selection;
- recursive subject and independent-decision file hash checks;
- explicit fixture-versus-production scope that rejects substitution;
- normalized-text SHA checks and complete encode/decode/encode validation;
- global chunk/text/document/split/revision isolation;
- one-pass, source-separated `uint16[513]` token, `uint8[513]` mask, and JSONL
  lineage output without materializing corpus-wide packed records;
- exact source/split counts, source-share fractions, input bindings, artifact
  byte counts/hashes, logical-order identity, and manifest identities;
- two sequential byte-identical scratch builds with at least 10 GB free disk;
- stale-directory, link/junction, disk, fsync, mutation, nondeterminism, and
  incomplete-lineage failure handling; and
- success cleanup, failure-evidence preservation, and explicit false
  publication/training authority.

The smoke uses two synthetic sources and all three splits. It writes only under
a temporary directory on the repository volume and removes it on success. It
does not use or create real source admissions, acquisition receipts, base-data
outputs, manifests, schedules, configurations, checkpoints, or optimizer state.

## Frozen pre-commit evidence

- Parent commit: `561666f06d81bf2710098c3b84220cdda6026650`
- Implementation SHA-256:
  `5d90b394231d509741bcaabd38b935b0cb909615b98b24ca5268b20521952ecf`
- Tests SHA-256:
  `1dc16b1617b8a502bc3e84e7352ea801a81afd1bc723d8a4a69298d1ad0a5dd4`
- Smoke SHA-256:
  `958cc9435df632a1dfbe8c5ab16b417480f2f5d70f762672d4b8612f77f5111e`
- Frozen fixture SHA-256:
  `da0eb48b0d6bb75632b92df724b6703007f310b1ca5468e1a5844c0ecb3fc915`
- Fixture qualification SHA-256:
  `dc54bf95522018ec46b5541247280d5784ebce4587179b07822f604207af0200`
- Byte-identical pass SHA-256:
  `e686195cb9f8352f76bec2eee1550612fdfa7147d91e8e3cc6e5eaa51dfe900b`
- Expected internal manifest SHA-256:
  `469dc5e8045ae521f947b75ba94ec6f0c4ca753318e8bade8195b59a12c87643`
- Expected external manifest template SHA-256:
  `b5ada0b694a0dcc6f45b0cbd69b03f930cda5785abdebbebb5eea9b96c65f8dd`

## Validation

- `python -m pytest tests\test_vasu_140m_base_release.py -q`: 19 passed.
- Focused base-release/base-record/release-plan regression excluding the
  historically published instruction-release publisher suite: 60 passed in
  65.55 seconds.
- Ruff passed for implementation, tests, and smoke.
- Smoke reproduced
  `evaluation/fixtures/vasu_140m_base_release_qualification_fixture_v1.json`
  exactly.
- `git diff --check` reported only known CRLF conversion warnings.

The older `tests/test_vasu_140m_production_release.py` currently has unrelated
post-publication/time-dependent failures: it still expects the published
instruction-seed output to be absent and its synthetic authorization dates are
expired at the current date. This package neither changes nor weakens that
historical instruction-release path.

## Remaining production gates

This is not real-corpus qualification. Production evidence envelopes, admitted
source bytes, scale/disk measurement, real Windows junction/open-handle
execution, publication transaction, detached authorization, and recovery
remain separately reviewed gates.

## Compatibility and non-authorization

The implementation is additive. Existing VASU-31M/60M checkpoints, optimizer
states, tokenizer, datasets, masks, instruction release, model architecture,
training configurations, and exact-resume behavior are unchanged.

No production base release, external manifest, authorization envelope,
schedule, training configuration, checkpoint, optimizer, or training run was
created or authorized.
