# Repository Line-Ending Reproducibility Review Packet

Status: **independent GPT-5.5 review requested; non-authorizing.**

## Requested Decision

Accept or reject the repository-owned line-ending policy needed for exact
cross-checkout identities. Acceptance authorizes only a later isolated commit
of `.gitattributes`, its regression test, this audit, the packet, and the
decision document.

## Required Evidence

Read:

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/REPOSITORY_LINE_ENDING_REPRODUCIBILITY_AUDIT_20260801.md`
- `.gitattributes`
- `tests/test_repository_line_endings.py`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `evaluation/framework/vasu_140m_base_v2_tasks.py`
- `evaluation/framework/vasu_140m_base_v2_statistics.py`
- `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json`
- `assets/tokenizer.json`

## Required Questions

1. Does the audit correctly identify checkout conversion rather than changed
   Git blob content as the failure cause?
2. Is repository-owned `eol=lf` stronger and more portable than a local Git
   setting or hash-time normalization?
3. Are binary research artifacts protected from text conversion?
4. Does the policy preserve the accepted tokenizer and scoring blob identities?
5. Is the test strict enough to prevent removal or weakening of the policy?
6. Does the remediation leave model, tokenizer semantics, data, checkpoints,
   exact resume, and training authorization unchanged?

## Required Commands

```powershell
git config --show-origin --get core.autocrlf
git check-attr text eol -- assets/tokenizer.json evaluation/framework/vasu_140m_base_v2_tasks.py evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json data/processed/fineweb_edu/train.bin checkpoints/vasu_60m/milestones/fineweb_step_200000.pt
python -m pytest tests\test_repository_line_endings.py tests\test_vasu_140m_base_v2_scoring.py tests\test_vasu_140m_base_evaluation_v2.py -q
python -m ruff check tests\test_repository_line_endings.py
git diff --check
git status --short
```

Independently compare the SHA-256 of each committed blob at `HEAD` with its
current working-tree file for the tokenizer, two scoring implementations,
scoring test, smoke, and fixture.

## Decision Output

Create only:

`docs/REPOSITORY_LINE_ENDING_REPRODUCIBILITY_INDEPENDENT_REVIEW_DECISION_20260801.md`

Report the decision, severity-grouped findings, evidence, exact commands and
results, compatibility, and explicit non-authorization. Do not modify code,
tests, policies, fixtures, shared docs, data, checkpoints, configurations,
schedules, authorization records, or training state. Do not commit or push.
