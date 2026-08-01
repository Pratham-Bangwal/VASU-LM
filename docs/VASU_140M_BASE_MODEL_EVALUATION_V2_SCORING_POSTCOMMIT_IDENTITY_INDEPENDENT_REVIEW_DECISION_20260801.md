# VASU-140M Base-Model Evaluation v2 Scoring Post-Commit Identity Independent Review Decision

Status: accepted; non-authorizing.

Review date: 2026-08-01

Reviewer: GPT-5.5 independent review

## Decision

Accept the clean post-commit identity of the VASU-140M evaluation-v2 scoring
package at runtime commit `92656ac0c8b8652fca3758a156e97a717d6b89a3`.

## Findings

### Blocking

None.

### High

None.

### Medium

None.

### Low

- The main author-side worktree remains concurrently modified and contains
  unrelated untracked VASU-140M packages. This was not treated as a blocker
  because validation ran in a separate clean detached worktree at the exact
  runtime commit, and the temporary worktree was removed afterward.
- The post-commit review packet and post-commit fixture are author-side
  evidence files outside runtime commit `92656ac`; the clean checkout generated
  compact evidence that matched the main-worktree frozen post-commit fixture
  exactly.

## Evidence Examined

- `AGENTS.md`
- `docs/PROJECT_STATUS.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_POSTCOMMIT_IDENTITY_REVIEW_PACKET.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_POSTCOMMIT_IDENTITY_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_AUDIT_20260801.md`
- `docs/VASU_140M_BASE_MODEL_EVALUATION_V2_SCORING_QUALIFICATION_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `docs/REPOSITORY_LINE_ENDING_REPRODUCIBILITY_INDEPENDENT_REVIEW_DECISION_20260801.md`
- `.gitattributes`
- `evaluation/framework/vasu_140m_base_v2_tasks.py`
- `evaluation/framework/vasu_140m_base_v2_statistics.py`
- `tests/test_vasu_140m_base_v2_scoring.py`
- `scripts/smoke_vasu_140m_base_v2_scoring.py`
- `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json`
- `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_postcommit_92656ac.json`
- Commit `926add62666b1f54d9aaac8d1c8ba1fda56108c2`
- Runtime commit `92656ac0c8b8652fca3758a156e97a717d6b89a3`

## Commands and Exact Results

- `git merge-base --is-ancestor 926add62666b1f54d9aaac8d1c8ba1fda56108c2 92656ac0c8b8652fca3758a156e97a717d6b89a3`
  returned exit code 0; `ancestor=true`.
- `git worktree add --detach $env:TEMP\vasu-scoring-postcommit-92656ac-review 92656ac0c8b8652fca3758a156e97a717d6b89a3`
  created a detached worktree with `HEAD is now at 92656ac chore(repo): enforce reproducible line endings`.
- In the detached worktree, `git rev-parse HEAD` returned
  `92656ac0c8b8652fca3758a156e97a717d6b89a3`.
- In the detached worktree, `git status --short` returned no output.
- In the detached worktree, `git config --show-origin --get core.autocrlf`
  returned `file:C:/Program Files/Git/etc/gitconfig	true`.
- In the detached worktree, `git check-attr text eol -- assets/tokenizer.json evaluation/framework/vasu_140m_base_v2_tasks.py evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json data/processed/fineweb_edu/train.bin checkpoints/vasu_60m/milestones/fineweb_step_200000.pt`
  returned `text: auto`, `eol: lf` for the tokenizer and scoring text files,
  and `text: unset` for the `.bin` and `.pt` artifact paths.
- In the detached worktree,
  `python -m pytest tests\test_repository_line_endings.py tests\test_vasu_140m_base_v2_scoring.py tests\test_vasu_140m_base_evaluation_v2.py tests\test_capability_framework.py tests\test_evaluation_metrics.py tests\test_evaluation_comparison.py -q`
  passed: `99 passed in 4.27s`.
- In the detached worktree,
  `python -m ruff check tests\test_repository_line_endings.py evaluation\framework\vasu_140m_base_v2_tasks.py evaluation\framework\vasu_140m_base_v2_statistics.py tests\test_vasu_140m_base_v2_scoring.py scripts\smoke_vasu_140m_base_v2_scoring.py`
  passed: `All checks passed!`.
- In the detached worktree, compact fixture replay against
  `D:\VASU\evaluation\fixtures\vasu_140m_base_v2_scoring_qualification_postcommit_92656ac.json`
  passed: `post-commit scoring fixture matched`.
- Pre-commit versus post-commit compact fixture comparison reported:
  `diff_fields=evidence_sha256,qualification_sha256,repository_commit` and
  `fixture_delta_expected=true`.
- In the detached worktree,
  `python scripts\smoke_vasu_140m_base_v2_scoring.py | python -m json.tool > $null`
  passed: `full post-commit smoke JSON valid`.
- In the detached worktree, `git diff --check` returned no output.
- In the detached worktree, final `git status --short` returned no output.
- `git worktree remove --force C:\Users\acer\AppData\Local\Temp\vasu-scoring-postcommit-92656ac-review`
  completed, and the path no longer existed.
- In the main worktree, `git worktree list` showed only `D:/VASU  92656ac [main]`.

## Clean-Checkout Status

The temporary detached checkout used the default Windows Git configuration,
including system `core.autocrlf=true`, and remained clean before and after all
validation commands. The checkout preserved deterministic LF identities for
hash-bound text via `.gitattributes`.

## Exact Identity Comparison

Clean-checkout SHA-256 values at runtime commit `92656ac`:

| Artifact | SHA-256 |
| --- | --- |
| `assets/tokenizer.json` | `04942e101a4a01f87f7e492ad9e463d299a559e784b650fdedd1763a017d195a` |
| `evaluation/framework/vasu_140m_base_v2_tasks.py` | `7777bd7d6a488c21e3937b5b05e1e46153e7fddbcb67784c47dc9e53a5087bf5` |
| `evaluation/framework/vasu_140m_base_v2_statistics.py` | `5d4c07d01fe7b833a2b6daf06fb562061ed0731160ffbd0e031eba3527568d71` |
| `tests/test_vasu_140m_base_v2_scoring.py` | `c71190f99822d1c5a887aa64bc1b22c5c2b57f465124ee34053d7ec2a8a62b34` |
| `scripts/smoke_vasu_140m_base_v2_scoring.py` | `6e43664e5eb1d8f39916ff63f1425ebbb3f339ee69ec6dceb0ec21dcb979e074` |
| `evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_v1.json` | `0ff1801e11d96036a7c645b752754955eda0e4dc5a640c7c57ad1f2e918cfe26` |

Post-commit compact evidence generated in the clean checkout reported:

- `repository_commit`: `92656ac0c8b8652fca3758a156e97a717d6b89a3`
- `evidence_sha256`: `5a681598e81d253d140cf66e30344de97eb352205e2802d543cd263cfb628f1f`
- `qualification_sha256`: `de5937040c96320a30796dbfd431eadda9f18ba69707f3db415f4c1c856179e0`
- `fixture_only=true`
- `prompt_content_included=false`
- `production_suite_frozen=false`
- `held_out_opened=false`
- `model_invoked=false`
- `checkpoint_opened=false`
- `evaluation_run_authorized=false`
- `training_authorized=false`

Compared with the accepted pre-commit fixture, the post-commit fixture differs
only in:

- `repository_commit`: `eafd4d708415116663a5c1b1909cc94a6e7b00b3` to
  `92656ac0c8b8652fca3758a156e97a717d6b89a3`;
- `evidence_sha256`: `0790f5ef1fd301968f8fbb3e430e32c213128b63365ca7719139fa33e21e0f1b`
  to `5a681598e81d253d140cf66e30344de97eb352205e2802d543cd263cfb628f1f`;
- `qualification_sha256`: `5fbb82d7eabd6dc025b595776c1dafa814c117785af6f7e1821663b82d59c32c`
  to `de5937040c96320a30796dbfd431eadda9f18ba69707f3db415f4c1c856179e0`.

All scorer inputs, observations, score rows, summaries, bootstrap settings,
implementation/test/smoke hashes, and non-authorization flags remained
unchanged.

## Fixture Reproducibility

The detached runtime checkout reproduced
`evaluation/fixtures/vasu_140m_base_v2_scoring_qualification_postcommit_92656ac.json`
exactly from `python scripts\smoke_vasu_140m_base_v2_scoring.py --evidence-only`.
The full smoke report was valid JSON.

## Compatibility

The runtime commit preserves the accepted scoring implementation and adds only
repository line-ending policy around the already accepted LF identities. It
does not change model architecture, tokenizer semantics, datasets, masks,
checkpoints, optimizer or scheduler state, training configurations, evaluation
scoring semantics, existing evaluation outputs, or exact-resume behavior.

## Non-Authorization

This acceptance does not authorize inventory construction, held-out opening,
source acquisition, data construction, model execution, checkpoint creation or
access, evaluation publication, configuration creation, schedule creation,
optimizer updates, authorization records, training, commit, or push.
