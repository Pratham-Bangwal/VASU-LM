# Repository Line-Ending Reproducibility Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the repository line-ending reproducibility remediation for a later
isolated line-ending-policy commit.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The review was performed in a concurrent author-side worktree with unrelated
  modified files and many untracked VASU-140M packages. This was not treated
  as a blocker because the line-ending policy, test, audit, and hash-bound
  blob comparisons are self-contained.
- `git diff --check` reported LF-normalization warnings for existing modified
  documentation files only; it reported no whitespace errors.
- The representative `data/processed/fineweb_edu/train.bin` path named in the
  packet is absent in this checkout. The `.bin` rule still resolves to
  `text: unset` through `git check-attr`, so binary normalization is disabled
  for that artifact class.

## Root-Cause Assessment

The audit's root cause is supported. Commit
`926add62666b1f54d9aaac8d1c8ba1fda56108c2` contains the accepted scoring
package, and the machine-wide Git configuration reports `core.autocrlf=true`.
Without a repository-owned checkout policy, Windows checkout conversion can
change working-tree bytes for LF hash-bound text even when the committed Git
blobs remain correct. The direct blob-versus-working-tree comparison confirms
that the committed blobs are the accepted LF byte identities and that the
current remediation preserves those bytes in the working tree.

Repository-owned `.gitattributes` with `text=auto eol=lf` is stronger and more
portable than relying on each contributor's local Git settings. Hash-time
normalization would weaken byte-level scientific identities, while this policy
preserves exact bytes at checkout.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/REPOSITORY_LINE_ENDING_REPRODUCIBILITY_AUDIT_20260801.md`
- `docs/REPOSITORY_LINE_ENDING_REPRODUCIBILITY_REVIEW_PACKET.md`
- `.gitattributes`
- `tests/test_repository_line_endings.py`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `evaluation/framework/vasu_140m_base_v2_tasks.py`
- `evaluation/framework/vasu_140m_base_v2_statistics.py`
- `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json`
- `assets/tokenizer.json`
- Commit `926add62666b1f54d9aaac8d1c8ba1fda56108c2`

## Commands and Exact Results

- `git status --short`
  showed concurrent author-side modified and untracked files, including
  untracked `.gitattributes`, the line-ending audit, packet, and regression
  test.
- `git show --stat --oneline --decorate --name-status 926add62666b1f54d9aaac8d1c8ba1fda56108c2`
  showed `926add6 (HEAD -> main) feat(evaluation): qualify VASU-140M scoring`
  with the accepted scoring audit, decision, packet, fixture, scorer,
  statistics, smoke, and tests added.
- `git show --format=fuller --no-patch 926add62666b1f54d9aaac8d1c8ba1fda56108c2`
  reported author and commit date `Sat Aug 1 22:13:37 2026 +0530`.
- `git config --show-origin --get core.autocrlf`
  returned `file:C:/Program Files/Git/etc/gitconfig	true`.
- `git check-attr text eol -- assets/tokenizer.json evaluation/framework/vasu_140m_base_v2_tasks.py evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json data/processed/fineweb_edu/train.bin checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`
  returned `text: auto`, `eol: lf` for the tokenizer and scoring text files;
  `text: unset` for the `.bin` and `.pt` artifact paths.
- `python -m pytest tests\test_repository_line_endings.py tests\test_vasu_140m_base_v2_scoring.py tests\test_vasu_140m_base_evaluation_v2.py -q`
  passed: `52 passed in 0.57s`.
- `python -m ruff check tests\test_repository_line_endings.py`
  passed: `All checks passed!`.
- `git diff --check`
  completed with LF-normalization warnings only for existing modified
  documentation files and no whitespace errors.

## Git-Blob Versus Working-Tree Identity Comparison

Binary-safe comparison used `git cat-file blob HEAD:<path>` for committed
blobs and direct `Path.read_bytes()` for working-tree files.

| Path | Git blob SHA-256 | Working-tree SHA-256 | Result |
| --- | --- | --- | --- |
| `assets/tokenizer.json` | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` | Match; no CRLF |
| `evaluation/framework/vasu_140m_base_v2_tasks.py` | `7777bd7d6a488c21e3937b5b05e1e46153e7fddbcb67784c47dc9e53a5087bf5` | `7777bd7d6a488c21e3937b5b05e1e46153e7fddbcb67784c47dc9e53a5087bf5` | Match; no CRLF |
| `evaluation/framework/vasu_140m_base_v2_statistics.py` | `5d4c07d01fe7b833a2b6daf06fb562061ed0731160ffbd0e031eba3527568d71` | `5d4c07d01fe7b833a2b6daf06fb562061ed0731160ffbd0e031eba3527568d71` | Match; no CRLF |
| `tests/test_vasu_140m_base_v2_scoring.py` | `c71190f99822d1c5a887aa64bc1b22c5c2b57f465124ee34053d7ec2a8a62b34` | `c71190f99822d1c5a887aa64bc1b22c5c2b57f465124ee34053d7ec2a8a62b34` | Match; no CRLF |
| `scripts/smoke_vasu_140m_base_v2_scoring.py` | `6e43664e5eb1d8f39916ff63f1425ebbb3f339ee69ec6dceb0ec21dcb979e074` | `6e43664e5eb1d8f39916ff63f1425ebbb3f339ee69ec6dceb0ec21dcb979e074` | Match; no CRLF |
| `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json` | `0ff1801e11d96036a7c645b752754955eda0e4dc5a640c7c57ad1f2e918cfe26` | `0ff1801e11d96036a7c645b752754955eda0e4dc5a640c7c57ad1f2e918cfe26` | Match; no CRLF |

The tokenizer identity also matches the VASU-140M constant
`TOKENIZER_SHA256 = 04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a`.

## Binary-Artifact Protection

`.gitattributes` applies deterministic LF checkout to text by default and
explicitly unsets text normalization for binary research artifact classes:
`*.bin`, `*.npy`, `*.npz`, `*.pt`, `*.safetensors`, images, PDFs, and archives.
The required `git check-attr` command confirmed `text: unset` for the packet's
representative `.bin` and `.pt` paths. The existing
`checkpoints/vasu_60m/milestones/fineweb_step_200000.pt` file remains present;
no binary artifact was modified.

## Compatibility Impact

The remediation changes checkout policy only. It does not change tokenizer
semantics, model architecture, datasets, masks, manifests, checkpoints,
optimizer or scheduler state, training configurations, evaluation scoring
logic, or exact-resume behavior. It preserves the accepted LF byte identities
for the tokenizer and VASU-140M scoring package while protecting binary
scientific artifacts from line-ending normalization.

## Non-Authorization

This acceptance authorizes only a later isolated line-ending-policy commit.
It does not authorize source acquisition, dataset construction, evaluation
execution, checkpoint creation, optimizer updates, configuration creation,
schedule creation, authorization records, training, commit, or push.
