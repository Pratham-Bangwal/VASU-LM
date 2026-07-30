# VASU-140M 513-Token Independent Review Decision

Status: accepted by independent reviewer; non-authorizing.

Date prepared: 2026-07-30

## Materials

- [frozen specification](VASU_140M_513_TOKEN_DATA_MASK_SPECIFICATION.md)
- [independent review packet](VASU_140M_513_TOKEN_INDEPENDENT_REVIEW_PACKET.md)
- `vasu/data/vasu_140m_records.py`
- `tests/test_vasu_140m_records.py`
- `evaluation/fixtures/vasu_140m_513_record_spec_v1.json`
- [requirement-level completion audit](VASU_140M_513_TOKEN_MILESTONE_AUDIT_20260730.md)

## Primary engineering disposition

The implementation is accepted as **engineering-complete and
independent-review-ready**. Evidence establishes tokenizer identity,
513-position shape/dtypes, shifted-mask behavior, prompt/PAD/cross-example
exclusion, response/EOS supervision, split isolation, overlength rejection,
and deterministic rebuild. No production release was created.

This disposition is deliberately not represented as an independent scientific
decision. Project policy recommends GPT-5.5 or another genuinely separate
reviewer. Scientific validity is stronger than claiming independence from a
second pass by the authoring agent.

## Review history

The first independent pass returned the contract to engineering because the
generic report validator proved only internal hash consistency, while the
review packet claimed enforcement of the exact checked-in fixture identity. A
self-consistent modified report could therefore be rehashed and accepted.

Engineering remediation now separates generic report validation from exact
frozen-fixture validation. The smoke path pins report SHA-256
`7329698aefc3ab4593b1bfba246487a83ed0210b4e3fbe96f72c8bb7b8c8e7e3`,
and an adversarial regression test proves that a modified-and-rehashed report
is rejected. The fresh independent review accepted the remediated contract.

## Independent decision form

The independent reviewer must select exactly one:

- **Accept for production-release specification:** the generic contract may be
  referenced by a new, source-specific immutable release plan.
- **Reject:** record the failed requirement and return the contract to
  engineering.

Either choice remains non-authorizing. Acceptance does not select sources,
construct data, create a schedule/configuration, authorize training, or
promote a checkpoint.

Reviewer: GPT-5.5 independent review

Review date: 2026-07-30

Decision: Accept for production-release specification

Rationale: Independent inspection found the remediated contract preserves the
generic self-consistency validator while adding an exact frozen-fixture
qualification gate pinned to report SHA-256
`7329698aefc3ab4593b1bfba246487a83ed0210b4e3fbe96f72c8bb7b8c8e7e3`.
The focused regression test demonstrates that a modified report can still pass
generic validation after recomputing its self-hash, but is rejected by
`validate_frozen_fixture_report`. The smoke rebuild remains deterministic and
matches the checked-in fixture; tokenizer, checkpoint, dataset, mask, and
exact-resume compatibility are preserved because the change is fixture-only
and creates no production data, schedule, training configuration,
authorization, checkpoint, optimizer update, commit, or push.
