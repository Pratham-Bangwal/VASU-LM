# VASU-140M Production Release Post-Commit Identity Review Packet

Status: independently accepted; non-authorizing.

Date: 2026-07-30

## Decision requested

Accept or reject the clean-commit qualification identity. This is a narrow
read-only transition review after the implementation acceptance and commit.
Acceptance would make the evidence eligible for later authorization-package
design; it would not create or approve an authorization record.

## Exact identities

- implementation commit:
  `c014716ec38ef8f08842356fc016359dc5a233d7`;
- implementation file SHA-256:
  `0efb0189b4b0c5a8621da9053d56db57d95ebf5b0f1bb59753952ec8afa5d1e2`;
- post-commit qualification SHA-256:
  `ed6f64b9d5cb97925bc68186b4ec403de6e25dc484e33cb103ed05ee977f5cd6`;
- assignment SHA-256:
  `59481237acd2164f96dbbdc2b837ca8cabb00c496b1b6c42cb260b44bbd2b394`;
- external-manifest-template SHA-256:
  `eb72f0602431a7ab47ba4392e47b61315766a7efc689d85e836e81cf8bb47717`.

## Materials

- `evaluation/fixtures/vasu_140m_instruction_seed_v1_production_qualification_postcommit_c014716.json`
- `scripts/smoke_vasu_140m_postcommit_qualification.py`
- `vasu/data/vasu_140m_production_release.py`
- `docs/VASU_140M_PRODUCTION_RELEASE_BUILDER_IMPLEMENTATION_INDEPENDENT_REVIEW_DECISION_20260730.md`
- `evaluation/fixtures/vasu_140m_instruction_seed_v1_production_qualification.json`

## Required comparisons

1. Confirm HEAD is the exact implementation commit and the implementation file
   hash is unchanged from the accepted pre-commit review.
2. Confirm the plan, three earlier decisions, assignment, logical-record,
   token, and mask hashes are unchanged.
3. Confirm only commit-bound internal-manifest, external-template, and
   qualification hashes changed.
4. Reproduce the post-commit report exactly with the smoke command.
5. Confirm 996 decoded examples and 898/48/50 split counts.
6. Confirm the production release, external manifest, and receipt directory
   remain absent.

## Restrictions

Do not create an authorization record, call the publisher, construct production
artifacts, create a schedule or training config, select a checkpoint, commit,
push, or train. Record the independent decision in a new versioned document.

GPT-5.5 independently accepted the exact transition on 2026-07-30. See
[the decision record](VASU_140M_PRODUCTION_RELEASE_POSTCOMMIT_IDENTITY_INDEPENDENT_REVIEW_DECISION_20260730.md).

After this evidence was committed, `HEAD` advanced through documentation-only
descendants. The smoke therefore requires `c014716` to remain an ancestor and
the implementation file SHA-256 to remain exact; it does not incorrectly
require documentation commits to keep `HEAD` equal to `c014716`.
