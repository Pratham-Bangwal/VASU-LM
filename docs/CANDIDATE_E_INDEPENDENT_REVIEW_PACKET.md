# Candidate E Independent Review Packet

Status: ready for independent review; not authorized for data construction or
training.

## Decision requested

Approve or reject moving from the completed **preparation package** to a
separately reviewed Candidate E logical-data specification. This packet does
not approve generation of that data, any mixture/schedule/configuration, or a
training run.

## Requirement-to-evidence audit

| Requirement | Evidence | Result |
| --- | --- | --- |
| Deterministic, metadata-derived steps | `vasu/data/arithmetic_steps.py`; focused serializer tests | Pass |
| Same prompt; only target serialization differs | `PROMPT_PREFIX_FORMAT` and `response_text` in `vasu/data/arithmetic_step_supervision.py` | Pass |
| Prompt/PAD/cross-example targets unsupervised | Compiler/packer invariants and `test_arithmetic_step_supervision.py` | Pass |
| Verified response and EOS supervision | Compiler invariant, packed provenance validation, and focused tests | Pass |
| Paired source equality and split isolation | `validate_matched_logical_splits`; its tests reject mismatched IDs and semantic overlap | Pass |
| Immutable paired artifacts and provenance | `vasu/data/candidate_e_release.py` builds atomically from caller-supplied splits, hashes every artifact, and validates both manifests | Pass |
| No implicit data creation | The release constructor has no generator/default source paths; repository scan found no Candidate E processed release | Pass |
| No Candidate E schedule/config/training | Repository scan found no Candidate E configuration; all manifests written by the constructor set `training_authorized: false` | Pass |
| Tokenizer boundary and record capacity | [reference tokenizer audit](CANDIDATE_E_REFERENCE_TOKENIZER_AUDIT.md): 34,000 reference records compiled; maximum treatment length 59 of 257 | Pass (reference envelope only) |
| Matched budget and evaluation/safety gates | [budget and evaluation protocol](CANDIDATE_E_BUDGET_AND_EVALUATION_PROTOCOL.md) plus `validate_matched_budget` and focused tests | Pass |

Automated evidence: after the release constructor was added, the full
repository test suite completed with **1032 passed, 8 skipped**. Focused tests
also exercised temporary paired-release construction and subsequent validation.

## Required reviewer decisions before the next boundary

1. Approve a new logical-data specification: stable-ID namespace, counts,
   seeds, operand ranges, and held-out template families. These cannot reuse
   arithmetic-v2 held-out identities.
2. Confirm the fixed `Question ... Response` prefix, `Answer:` control target,
   and bounded verified-step treatment target are the desired sole data-format
   difference.
3. Accept equal processed model tokens/updates as the matching rule, with the
   higher treatment supervision exposure treated as the experimental variable.
4. Approve the pre-registered promotion, retention, malformed-output, and
   repetition gates unchanged.
5. After (1)--(4), require a new release-review record before invoking the
   constructor, then a separate schedule/configuration/training authorization.

## Explicit non-authorizations

- No Candidate E processed-data directory exists.
- No Candidate E schedule, configuration, authorization registry entry, or
  checkpoint exists.
- No training process has been started, resumed, or approved.

Existing checkpoints, data releases, tokenizer, inference behavior, and
Candidate D results remain unchanged.
